# model.py
# Optimized version preserving original class names and API.
# - Same external API as your original file (forward returns {"logits"} or {"loss","logits"})
# - Latency & FLOP focused internals (depthwise-separable stem, efficient linear attention, layerscale)
# - Optional AMP and torch.compile in get_latency helper

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


# -------------------------
# Small helpers
# -------------------------
def _make_divisible(x, d=8):
    return int((x + d - 1) // d * d)


# Depthwise-separable patch embed (replaces direct Conv2d for large input channels)
class _DepthwiseSeparablePatchEmbed(nn.Module):
    def __init__(self, in_ch, out_dim, patch_size=1):
        super().__init__()
        # depthwise conv with stride=patch_size to create non-overlapping patches
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=patch_size, stride=patch_size,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_dim)

        # init
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.pw.weight, nonlinearity='linear')

    def forward(self, x):
        # x: B, C, H, W
        x = self.dw(x)           # B, C, H', W'
        x = self.pw(x)           # B, out_dim, H', W'
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1).contiguous()  # B, N, C
        x = self.norm(x)
        return x


# -------------------------
# SpectralAttention (kept name) - small MLP spectral mixer
# -------------------------
class SpectralAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # smaller hidden dim for faster inference
        hidden = max(8, dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True)
        )
        # LayerNorm to stabilize spectral output
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: B, C (we expect pooled tokens)
        return self.mlp(self.norm(x))


class HybridLinearAttention(nn.Module):
    """
    Hybrid attention: 90% linear attention + 10% global full attention.
    Adds RoPE, keeps same API and shapes as your current module.
    """
    def __init__(self, dim, heads=8, dim_head=64, global_ratio=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.global_ratio = global_ratio

        inner_dim = dim_head * heads

        # QKV projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # Output projection
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(dim_head)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        # Project
        q = self.to_q(x).view(b, n, h, self.dim_head)
        k = self.to_k(x).view(b, n, h, self.dim_head)
        v = self.to_v(x).view(b, n, h, self.dim_head)

        # Apply RoPE
        q = self.rotary_emb.apply_rotary_pos_emb(q)
        k = self.rotary_emb.apply_rotary_pos_emb(k)

        # Split global vs linear tokens
        g = max(1, int(n * self.global_ratio))     # number of global tokens
        q_global, k_global, v_global = q[:, :g], k[:, :g], v[:, :g]

        q_local, k_local, v_local = q[:, g:], k[:, g:], v[:, g:]

        # ----------------------------
        # 1) FULL ATTENTION for global tokens
        # ----------------------------
        attn_scores = torch.einsum("bghd,bghd->bghg", q_global, k_global) * self.scale
        attn = attn_scores.softmax(dim=-1)
        global_out = torch.einsum("bghg,bghd->bghd", attn, v_global)

        # ----------------------------
        # 2) LINEAR ATTENTION for local tokens
        # ----------------------------
        k_local_sm = F.softmax(k_local, dim=1)
        kv = torch.einsum("blhd,blhe->bhde", v_local, k_local_sm)
        local_out = torch.einsum("blhd,bhde->blhe", q_local, kv)

        # ----------------------------
        # 3) Concatenate back
        # ----------------------------
        out = torch.cat([global_out, local_out], dim=1)
        out = out.reshape(b, n, h * self.dim_head)

        return self.to_out(out)


# -------------------------
# newViTBlock (kept name) - uses EfficientAttention above + LayerScale + small MLP
# -------------------------
class newViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, layerscale=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        # smaller mlp by default if dim small; keep bias True for numeric stability
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True)
        )

        # LayerScale parameters
        if layerscale:
            self.gamma_1 = nn.Parameter(1e-3 * torch.ones(dim))
            self.gamma_2 = nn.Parameter(1e-3 * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x):
        # Attention
        attn_out = self.attn(self.norm1(x))
        if self.gamma_1 is not None:
            x = x + attn_out * self.gamma_1
        else:
            x = x + attn_out

        # MLP
        mlp_out = self.mlp(self.norm2(x))
        if self.gamma_2 is not None:
            x = x + mlp_out * self.gamma_2
        else:
            x = x + mlp_out
        return x


# -------------------------
# newFastViT (kept name) - main model class (API preserved)
# - I preserved constructor signature and forward return format
# - Added optional flags to trade latency/accuracy; defaults keep original behavior but optimize internals
# -------------------------
class newFastViT(nn.Module):
    def __init__(self, image_size=5, patch_size=1, num_channels=103, num_classes=9,
                 embed_dim=768, depth=6, num_heads=12, mlp_ratio=4.,
                 token_reduction_factor: int = 1,
                 use_spectral: bool = True,
                 qkv_bias: bool = False):
        """
        Kept external API (names & defaults) but optimized internals for latency:
        - depthwise-separable patch embedding
        - efficient linear attention implementation
        - optional token_reduction_factor (>=1) to reduce token count (trades accuracy for latency)
        - spectral attention remains but lighter
        """
        super().__init__()

        # keep dims divisible
        embed_dim = _make_divisible(embed_dim, 8)
        self.token_reduction_factor = max(1, int(token_reduction_factor))
        self.patch_size = patch_size
        self.use_spectral = use_spectral

        # Depthwise separable patch embed
        self.patch_embed = _DepthwiseSeparablePatchEmbed(num_channels, embed_dim, patch_size=patch_size)

        # Positional embedding: store max tokens for the default image_size and patch_size,
        # but allow interpolation in forward
        seq_len = (image_size // patch_size) ** 2
        reduced_seq = (seq_len + self.token_reduction_factor - 1) // self.token_reduction_factor
        self.pos_embed = nn.Parameter(torch.zeros(1, reduced_seq, embed_dim))

        # Blocks
        self.blocks = nn.ModuleList([
            newViTBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, layerscale=True)
            for _ in range(depth)
        ])

        # Lighter spectral attention (pool + small mlp)
        if use_spectral:
            self.spectral_attention = SpectralAttention(embed_dim)
        else:
            self.spectral_attention = nn.Identity()

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.head.weight, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def _reduce_tokens(self, x):
        # x: B, N, C
        r = self.token_reduction_factor
        if r == 1:
            return x
        B, N, C = x.shape
        # pad if necessary
        pad = (r - (N % r)) % r
        if pad:
            # simple pad by repeating first tokens (cheap)
            pad_tensor = x[:, :pad, :].contiguous()
            x = torch.cat([x, pad_tensor], dim=1)
            N = x.shape[1]
        # reshape and mean
        x = x.view(B, N // r, r, C).mean(dim=2)
        return x

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        """
        x: B, C, H, W
        returns: {"logits": logits} or {"loss": loss, "logits": logits}
        """
        # 1) patch embed
        x = self.patch_embed(x)  # B, N, C

        # 2) token reduction (optional)
        x = self._reduce_tokens(x)  # B, N', C

        # 3) pos embedding (interpolate if length mismatch)
        if x.shape[1] != self.pos_embed.shape[1]:
            # interpolate pos embeddings over token dimension
            p = F.interpolate(self.pos_embed.permute(0, 2, 1), size=x.shape[1], mode='linear', align_corners=False)
            p = p.permute(0, 2, 1)
            x = x + p
        else:
            x = x + self.pos_embed

        # 4) transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # 5) spectral attention (global)
        # keep spectral on pooled representation to be cheap
        pooled = x.mean(dim=1)  # B, C
        if self.use_spectral:
            pooled = self.spectral_attention(pooled)  # B, C (mlp)
        # 6) head
        logits = self.head(self.norm(pooled))

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    # -------------------------
    # Latency helper (preserves your earlier style but improved)
    # -------------------------
    @torch.no_grad()
    def get_latency(self, input_shape=(1, 103, 145, 145), device='cuda', repeats=200, warmup=30,
                    use_amp: bool = True, use_compile: bool = True):
        """
        Measure average latency (ms) for forward() over `repeats` runs after `warmup`.
        - use_amp: if True and device is CUDA, uses torch.cuda.amp.autocast during runs (simulates mixed precision path)
        - use_compile: if True and torch.compile available, attempt to compile the model once before timing
        """
        self.eval()
        self.to(device)

        # optional compile (wrap model ref returned by torch.compile)
        compiled_model = self
        if use_compile and hasattr(torch, "compile"):
            try:
                compiled_model = torch.compile(self)
            except Exception:
                compiled_model = self

        x = torch.randn(input_shape, device=device)
        # Warmup
        for _ in range(warmup):
            if use_amp and device.startswith('cuda'):
                with torch.cuda.amp.autocast():
                    _ = compiled_model(x)
            else:
                _ = compiled_model(x)

        torch.cuda.synchronize()
        import time
        t0 = time.time()
        for _ in range(repeats):
            if use_amp and device.startswith('cuda'):
                with torch.cuda.amp.autocast():
                    _ = compiled_model(x)
            else:
                _ = compiled_model(x)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / repeats * 1000.0
        print(f"Average latency ({repeats} runs): {elapsed:.3f} ms")
        return elapsed


# -------------------------
# Quick smoke test when run directly
# -------------------------
if __name__ == "__main__":
    # match your earlier default signature
    model = newFastViT(image_size=145, patch_size=5, num_channels=103, num_classes=9,
                       embed_dim=384, depth=4, num_heads=8, mlp_ratio=2.0,
                       token_reduction_factor=2, use_spectral=True)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(1, 103, 145, 145)
    out = model(x)
    print("Output keys:", out.keys())
    print("Logits shape:", out["logits"].shape)

    # Uncomment to run latency test on GPU (requires CUDA)
    # model.get_latency(input_shape=(1,103,145,145), device='cuda', repeats=200, warmup=30)
