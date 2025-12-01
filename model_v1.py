import torch
import torch.nn as nn
from einops import rearrange

# ---------------------------
# Helpers
# ---------------------------
def make_divisible(x, d=8):
    return int((x + d - 1) // d * d)

# Depthwise-separable conv patch embed (much cheaper than huge conv)
class DepthwiseSeparablePatchEmbed(nn.Module):
    def __init__(self, in_ch, out_dim, patch_size=1):
        super().__init__()
        # depthwise conv
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=patch_size, stride=patch_size, groups=in_ch, bias=False)
        # pointwise
        self.pw = nn.Conv2d(in_ch, out_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: B, C, H, W
        x = self.dw(x)
        x = self.pw(x)                # B, out_dim, H', W'
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x

# Token reducer (optional) - average pooling on tokens (works on sequence dimension)
class TokenReducer(nn.Module):
    def __init__(self, reduction_factor: int):
        super().__init__()
        assert reduction_factor >= 1
        self.r = reduction_factor

    def forward(self, x):
        # x: B, N, C
        if self.r == 1:
            return x
        B, N, C = x.shape
        # if N not divisible by r, pad
        pad = (self.r - (N % self.r)) % self.r
        if pad:
            x = torch.cat([x, x[:, :pad, :]], dim=1)
            N = x.shape[1]
        x = x.view(B, N // self.r, self.r, C).mean(dim=2)  # pooled tokens
        return x

# Linear attention (kernel feature map: elu + 1)
class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.out = nn.Linear(dim, dim)
        # small epsilon for numerical stability
        self.eps = 1e-6

    def feature_map(self, x):
        # x: (..., dim)
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, x):
        # x: B, N, C
        B, N, C = x.shape
        qkv = self.qkv(x)  # B, N, 3C
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: B, heads, N, head_dim

        # apply feature map
        q_phi = self.feature_map(q)  # positive
        k_phi = self.feature_map(k)

        # compute k_phi^T * v  -> shape: B, heads, head_dim, head_dim? careful:
        # We'll compute per-head context: (B, heads, head_dim, value_dim) but v's last dim=head_dim
        # compute denominator and numerator using einsum
        # numerator: (k_phi.transpose(-2,-1) @ v) summed over tokens -> (B, heads, head_dim, head_dim)? No,
        # better compute context = sum_over_tokens(k_phi * v)
        # context: B, heads, head_dim, head_dim? Actually v has shape (B,heads,N,D), k_phi same.
        # We'll compute context = einsum('bhnd,bhne->bhde') -> works if v last dim != head_dim; but here v last dim = head_dim
        # So context: B, heads, head_dim, head_dim ; to multiply q_phi (B,heads,N,D) with context -> heavy.
        # Simpler and efficient formulation:
        # out_i = (sum_t (q_i(t) * (k_phi(t) @ v(t)))) / (sum_t q_i(t) @ k_phi(t))
        # We'll compute:
        # k_phi_v = einsum('bhnd,bhne->bhde') where n is sequence, but v's shape is b h n d, so bhne not correct.
        # We implement per-head using broadcasting:
        # compute S = einsum('b h n d, b h n e -> b h d e') with v and k_phi? That yields d x d matrix; then multiply q
        # But simpler standard linear attention practice: compute
        # KV = torch.einsum('b h n d, b h n e -> b h d e') where d=head_dim, e=head_dim -> forms matrix; then out = einsum('b h n d, b h d e -> b h n e')
        # We'll follow that.

        # Compute KV: (B, heads, D, D)
        KV = torch.einsum('bhnd,bhne->bhde', k_phi, v)

        # compute normalization factor Z = q_phi @ (k_phi.sum(dim=2).unsqueeze(-1))
        K_sum = k_phi.sum(dim=2)  # B, heads, D
        # denom: B, heads, N, 1
        denom = torch.einsum('bhnd,bhd->bhn', q_phi, K_sum).unsqueeze(-1)  # B,h,n,1
        denom = denom + self.eps

        # compute attention output: out = q_phi @ KV  -> q_phi: B,h,n,d ; KV: B,h,d,e -> out: B,h,n,e
        out = torch.einsum('bhnd,bhde->bhne', q_phi, KV)
        out = out / denom  # normalization
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out(out)
        return out

# Spectral mixing block using FFT (cheap global mixing)
class SpectralMixing(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(4, dim // 4)
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        # we will create frequency-domain complex weights per frequency at runtime (size depends on sequence length)
        # implement as two small linear layers that map real and imag parts separately
        self.freq_real_mapper = nn.Linear(dim, dim)
        self.freq_imag_mapper = nn.Linear(dim, dim)

    def forward(self, x):
        # x: B, N, C
        B, N, C = x.shape
        # rFFT along sequence dim -> shape (B, F, C), complex
        Xf = torch.fft.rfft(x, dim=1)  # complex64/128
        # split real & imag
        real = Xf.real
        imag = Xf.imag
        # map via small linear layers applied per-frequency across channel dim
        real_m = self.freq_real_mapper(real) - 0.5 * self.freq_imag_mapper(imag)
        imag_m = self.freq_imag_mapper(real) + 0.5 * self.freq_real_mapper(imag)
        Xf_mod = torch.complex(real_m, imag_m)
        # inverse
        x_time = torch.fft.irfft(Xf_mod, n=N, dim=1)
        # pass through pointwise mlp
        out = x + self.mlp(x_time)
        return out

# Transformer-like block using linear attention + FFN; includes LayerScale
class EfficientBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, use_linear_attn=True, layerscale=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.use_linear_attn = use_linear_attn
        if use_linear_attn:
            self.attn = LinearAttention(dim, num_heads=num_heads)
        else:
            # fall back to standard efficient attention (scaled dot-product) - keep for ablation
            self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )
        if layerscale:
            self.gamma_1 = nn.Parameter(1e-2 * torch.ones(dim))
            self.gamma_2 = nn.Parameter(1e-2 * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x):
        # Attention
        if self.use_linear_attn:
            attn_out = self.attn(self.norm1(x))
        else:
            # MultiheadAttention expects (B, N, C)
            attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
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

# Upgraded model
class UpgradedFastViT(nn.Module):
    def __init__(self,
                 image_size=5,
                 patch_size=1,
                 in_channels=103,
                 num_classes=9,
                 embed_dim=768,
                 depth=6,
                 num_heads=12,
                 mlp_ratio=4.0,
                 token_reduction_factor=1,
                 use_spectral=True,
                 use_linear_attn=True,
                 layerscale=True):
        super().__init__()

        embed_dim = make_divisible(embed_dim, 8)
        self.patch_embed = DepthwiseSeparablePatchEmbed(in_channels, embed_dim, patch_size=patch_size)
        # positional embedding uses reduced token count based on reduction factor
        seq_len = (image_size // patch_size) ** 2
        reduced_seq = (seq_len + token_reduction_factor - 1) // token_reduction_factor
        self.pos_embed = nn.Parameter(torch.zeros(1, reduced_seq, embed_dim))
        self.token_reducer = TokenReducer(token_reduction_factor)
        self.blocks = nn.ModuleList([
            EfficientBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           use_linear_attn=use_linear_attn, layerscale=layerscale)
            for _ in range(depth)
        ])
        self.use_spectral = use_spectral
        if use_spectral:
            self.spectral = SpectralMixing(embed_dim, hidden_dim=max(4, embed_dim // 4))
        else:
            self.spectral = nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # use standard initialization for head
        nn.init.normal_(self.head.weight, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x, labels=None):
        # x: B, C, H, W
        B = x.shape[0]
        x = self.patch_embed(x)  # B, N, C
        x = self.token_reducer(x)  # reduce tokens if requested
        # ensure pos_embed matches reduced length: if shape mismatch adapt by interpolation
        if x.shape[1] != self.pos_embed.shape[1]:
            # simple interpolation on pos embeddings (1D)
            pos = nn.functional.interpolate(self.pos_embed.permute(0, 2, 1), size=x.shape[1], mode='linear', align_corners=False)
            pos = pos.permute(0, 2, 1)
        else:
            pos = self.pos_embed
        x = x + pos

        for block in self.blocks:
            x = block(x)

        x = self.spectral(x)  # global spectral mixing

        x = self.norm(x)
        # global pooling
        feat = x.mean(dim=1)
        logits = self.head(feat)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
