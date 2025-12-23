"""
Spectral-Spatial Linear Transformer for Hyperspectral Image Classification

This model combines efficiency and high accuracy by explicitly modeling the distinct
structures of spectral and spatial correlations. The architecture factorizes attention
into linear spatial attention for token interactions and a learnable spectral gating
mechanism for band correlations, allowing ultra-low FLOPs and latency.

Key Features:
- Linear spatial attention for efficient token mixing
- Spectral gating mechanism for band correlations
- Single global attention block for long-range interactions
- Band-weighted pooling for interpretable feature aggregation
- Ultra-low FLOPs and latency while maintaining high accuracy
"""

import torch
import torch.nn as nn
from einops import rearrange


# ============================================================
# 1. Band-weighted pooling (replaces weak spectral MLP)
# ============================================================
class BandWeightedPooling(nn.Module):
    """
    Learnable spectral-band weighting for global token aggregation.
    Provides explicit spectral inductive bias.

    Args:
        dim: Embedding dimension
    """
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, C)
        Returns:
            Aggregated tensor of shape (B, C)
        """
        # x: (B, N, C)
        w = torch.softmax(self.weights, dim=0)
        return (x * w).sum(dim=1)


# ============================================================
# 2. Spectral-Spatial Factorized Linear Attention (CORE NOVELTY)
# ============================================================
class SpectralSpatialLinearAttention(nn.Module):
    """
    Linear attention with explicit spectral gating.
    Spatial mixing via linear attention, spectral mixing via channel gate.

    This is the core novelty: factorizing attention into spatial (via linear attention)
    and spectral (via gating) components for efficiency and interpretability.

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        # Spectral gate (explicit inductive bias for hyperspectral data)
        self.spectral_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, C)
        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: (B, N, num_heads, head_dim)

        # Linear spatial attention (O(N) complexity instead of O(N²))
        k = k.softmax(dim=1)  # Softmax over spatial dimension
        context = torch.einsum("bnhd,bnhv->bhdv", k, v)  # (B, num_heads, head_dim, head_dim)
        out = torch.einsum("bnhd,bhdv->bnhv", q, context)  # (B, N, num_heads, head_dim)
        out = out.reshape(B, N, C)

        # Spectral gating (explicit spectral channel modeling)
        gate = self.spectral_gate(x)
        out = out * gate

        return self.proj(out)


# ============================================================
# 3. One global attention block (accuracy stabilizer)
# ============================================================
class GlobalAttentionBlock(nn.Module):
    """
    Single full-attention block to restore long-range interactions.
    Cost is minimal (only one block), accuracy gain is large.

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, C)
        Returns:
            Output tensor of shape (B, N, C)
        """
        h = self.norm(x)
        out, _ = self.attn(h, h, h)
        return x + out


# ============================================================
# 4. Transformer Block
# ============================================================
class SpectralSpatialViTBlock(nn.Module):
    """
    Transformer block using Spectral-Spatial Linear Attention.

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpectralSpatialLinearAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, C)
        Returns:
            Output tensor of shape (B, N, C)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 5. FINAL MODEL
# ============================================================
class SpectralSpatialLinearTransformer(nn.Module):
    """
    Spectral-Spatial Linear Transformer for Hyperspectral Image Classification

    This architecture achieves a strong balance of speed, accuracy, and scientific insight:
    - Linear attention blocks for efficient spatial mixing (O(N) complexity)
    - Spectral gating for explicit band correlation modeling
    - Single global attention block for long-range context
    - Band-weighted pooling for interpretable aggregation

    Args:
        image_size: Size of input spatial patches (e.g., 5 for 5x5)
        patch_size: Size of patches for embedding (e.g., 1 for per-pixel)
        num_channels: Number of spectral bands (e.g., 144 for Houston)
        num_classes: Number of classification classes (e.g., 15 for Houston)
        embed_dim: Embedding dimension (must be divisible by num_heads)
        depth: Number of linear transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension
    """
    def __init__(
        self,
        image_size=5,
        patch_size=1,
        num_channels=103,
        num_classes=9,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Calculate number of patches
        num_patches_h = (image_size - patch_size) // patch_size + 1
        num_patches_w = (image_size - patch_size) // patch_size + 1
        num_patches = num_patches_h * num_patches_w

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Linear attention blocks (efficient)
        self.blocks = nn.ModuleList([
            SpectralSpatialViTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Single global attention block (accuracy booster)
        self.global_block = GlobalAttentionBlock(embed_dim, num_heads)

        # Spectral pooling + classifier
        self.spectral_pool = BandWeightedPooling(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, labels=None):
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W)
            labels: Optional ground truth labels of shape (B,)

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Patchify: (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w)
        x = self.patch_embed(x)

        # Flatten patches: (B, embed_dim, num_patches_h, num_patches_w) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embedding
        x = x + self.pos_embed

        # Linear transformer blocks (efficient)
        for blk in self.blocks:
            x = blk(x)

        # Global attention refinement (accuracy booster)
        x = self.global_block(x)

        # Spectral pooling: (B, num_patches, embed_dim) -> (B, embed_dim)
        x = self.spectral_pool(x)

        # Normalization and classification
        x = self.norm(x)
        logits = self.head(x)

        # Calculate loss if labels provided
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


# Alias for compatibility
newFastViT = SpectralSpatialLinearTransformer


if __name__ == "__main__":
    # Test the model
    print("Testing Spectral-Spatial Linear Transformer...")

    # Houston dataset parameters
    batch_size = 4
    num_channels = 144  # Houston spectral bands
    image_size = 5      # 5x5 spatial patches
    num_classes = 15    # Houston classes

    # Create model
    model = SpectralSpatialLinearTransformer(
        image_size=image_size,
        patch_size=4,
        num_channels=num_channels,
        num_classes=num_classes,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0
    )

    # Create dummy input
    x = torch.randn(batch_size, num_channels, image_size, image_size)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    output = model(x, labels)

    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params / 1e6:.2f}M")

    print("\n✅ Model test passed!")
