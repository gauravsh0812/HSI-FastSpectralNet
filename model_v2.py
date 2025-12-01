# optimized_proto_hyperformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# -----------------------------
# Utility helpers
# -----------------------------
def make_divisible(x, d=8):
    return int((x + d - 1) // d * d)


def freeze_bn_eval(m):  # convenience: no-op but could freeze any BN if present
    for p in m.parameters():
        p.requires_grad = p.requires_grad


# -----------------------------
# Fast patch embed: depthwise sep conv + optional norm
# -----------------------------
class FastPatchEmbed(nn.Module):
    def __init__(self, in_bands: int, out_dim: int, patch_size: int):
        super().__init__()
        # depthwise conv reduces memory traffic for large band counts
        self.dw = nn.Conv2d(in_bands, in_bands, kernel_size=patch_size, stride=patch_size,
                            groups=in_bands, bias=False)
        self.pw = nn.Conv2d(in_bands, out_dim, kernel_size=1, bias=False)
        # tiny bias-free layernorm on channel dim (applied after flatten)
        self.norm = nn.LayerNorm(out_dim)

        # initialize pointwise smaller scale
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.pw.weight, nonlinearity='linear')

    def forward(self, x):
        # x: B, C, H, W
        x = self.dw(x)          # B, C, H', W'
        x = self.pw(x)          # B, out_dim, H', W'
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1).contiguous()  # B, N, C  (avoid many transposes)
        x = self.norm(x)
        return x


# -----------------------------
# Efficient topk prototype gather (fast indexing)
# -----------------------------
def gather_topk_prototypes(prototypes: torch.Tensor, indices: torch.LongTensor, weights: torch.Tensor):
    # prototypes: (K, D)
    # indices: (B, N, topk)
    # weights: (B, N, topk, 1)
    # returns: (B, N, D)
    B, N, topk = indices.shape
    K, D = prototypes.shape
    # Expand prototypes to batch gather via indexing trick: prototypes[indices] works directly
    # indices: (B, N, topk) -> prototypes[indices] -> (B, N, topk, D)
    picked = prototypes[indices]  # uses advanced indexing; yields contiguous tensor on same device
    # weighted sum
    mixed = (picked * weights).sum(dim=2)  # (B, N, D)
    return mixed


# -----------------------------
# Tiny, fast linear attention using low-rank keys (optimized matmuls)
# - We implement the "attn = softmax(q, dim=-1) @ (K^T @ v)" pattern but avoid big transposes
# - We keep all tensors contiguous and use bmm where appropriate.
# -----------------------------
class FastLinearAttnBlock(nn.Module):
    def __init__(self, dim: int, ff_hidden: int, layerscale: float = 1e-2):
        super().__init__()
        # We'll implement q_proj and v_proj as smaller linear layers to reduce flops
        self.norm = nn.LayerNorm(dim)
        # bias=False to slightly reduce ops (safe when followed by residual)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        # small FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_hidden, bias=True),
            nn.GELU(),
            nn.Linear(ff_hidden, dim, bias=True)
        )
        # LayerScale: per-channel residual scaling
        self.gamma1 = nn.Parameter(layerscale * torch.ones((dim,)), requires_grad=True)
        self.gamma2 = nn.Parameter(layerscale * torch.ones((dim,)), requires_grad=True)

    def forward(self, x, keys):
        # x: (B, N, C), keys: (B, N, C) precomputed
        B, N, C = x.shape
        x_norm = self.norm(x)
        q = self.q_proj(x_norm)      # B,N,C
        v = self.v_proj(x_norm)      # B,N,C

        # ... HOT PATH: compute per-sample small matmuls with bmm
        # softmax on last dim across feature dim (NOT typical; we follow Grok formula softmax across channels)
        # But computing softmax across channels may be heavy if C large; we keep C moderate.
        q_s = F.softmax(q, dim=-1)   # B,N,C

        # keys.T @ v per token? Grok used keys.transpose(1,2) @ v -> (B, C, C) then q @ that -> (B,N,C)
        # compute KV = keys.transpose(1,2) @ v  -> shapes: (B, C, N) @ (B, N, C) -> (B, C, C)
        KV = torch.bmm(keys.transpose(1, 2), v)    # (B, C, C)

        # now out = q_s @ KV  -> per token: (B, N, C) @ (B, C, C) -> (B, N, C)
        # do as bmm by reshaping: q_s.view(B*N,1,C) @ KV.repeat(N,1,1)? that's costly.
        # Use efficient batched matmul: we can reshape KV to (B,1,C,C) and use torch.matmul broadcasting
        out = torch.matmul(q_s, KV.unsqueeze(1))  # B, N, 1, C
        out = out.squeeze(2)                      # B, N, C

        # Apply LayerScale residuals (element-wise multiply across channel dim)
        x = x + out * self.gamma1
        x = x + self.ffn(self.norm(x)) * self.gamma2
        return x


# -----------------------------
# Optimized ProtoHyperFormer
# -----------------------------
class OptimizedProtoHyperFormer(nn.Module):
    def __init__(
        self,
        in_bands: int = 103,
        num_classes: int = 9,
        patch_size: int = 16,
        dim: int = 192,
        depth: int = 3,
        k_spectral: int = 24,
        k_spatial: int = 24,
        topk: int = 2,
        reduced_bands: int = 8,
        temperature: float = 0.07,
        use_amp: bool = True
    ):
        """
        Optimized version focused on latency:
        - aggressive band reduction (default 8)
        - depthwise separable patch embed
        - minimal tokens via large patch_size
        - light FFNs and layer-scale
        - fast gather/topk with contiguous tensors
        - use_amp: inference/training with autocast (AMP) for speed
        """
        super().__init__()

        self.patch_size = patch_size
        self.dim = make_divisible(dim, 8)
        self.topk = topk
        self.temperature = temperature
        self.use_amp = use_amp

        # 1) Band reduction: 1x1 conv -> strongly regularized; small out channels
        # Use bias=True here for numeric stability (tiny cost)
        self.band_reduce = nn.Conv2d(in_bands, reduced_bands, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.band_reduce.weight, nonlinearity='linear')

        # 2) Fast patch embed (depthwise + pointwise)
        self.patch_embed = FastPatchEmbed(reduced_bands, self.dim, patch_size=patch_size)

        # compute token count per image (we keep pos_embed minimal)
        # pos embedding kept as small param if needed; we'll use simple learned token bias per token count
        # but for generality we create pos as length 1 and broadcast (works when tokens tiny)
        self.register_buffer('pos_bias', torch.zeros(1, 1, self.dim), persistent=False)

        # Prototype banks (small)
        self.spectral_prototypes = nn.Parameter(torch.randn(k_spectral, reduced_bands) * 0.02)
        self.spatial_prototypes = nn.Parameter(torch.randn(k_spatial, self.dim) * 0.02)

        # Router: tiny MLP (shared)
        self.router = nn.Sequential(
            nn.Linear(self.dim, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, k_spectral + k_spatial, bias=True)
        )

        # key projection: combine spectral and spatial mixed protos -> shorter projection
        self.key_proj = nn.Linear(reduced_bands + self.dim, self.dim, bias=False)

        # Blocks: extremely light
        ff_hidden = max(64, int(self.dim * 2))  # smaller than original 4x
        self.blocks = nn.ModuleList([FastLinearAttnBlock(self.dim, ff_hidden) for _ in range(depth)])

        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, num_classes, bias=True)

        # initialization: small
        nn.init.normal_(self.head.weight, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        x: (B, in_bands, H, W)
        """
        # Use autocast for faster fp16 matmuls on GPU when enabled
        if self.use_amp and x.is_cuda:
            return self._forward_amp(x, labels)
        else:
            return self._forward_nomix(x, labels)

    def _forward_amp(self, x, labels=None):
        # Keep operations inside torch.cuda.amp.autocast for faster inference/training
        with torch.cuda.amp.autocast():
            return self._forward_nomix(x, labels)

    def _forward_nomix(self, x, labels=None):
        B = x.shape[0]
        # 1) Band reduction
        x_reduced = self.band_reduce(x)        # (B, reduced_bands, H, W)

        # 2) Patch embed -> B,N,C
        x_tok = self.patch_embed(x_reduced)    # (B, N, dim)

        # 3) Router predicts per-token logits to pick prototypes
        router_logits = self.router(x_tok)     # (B, N, k_spectral + k_spatial)
        k_s = self.spectral_prototypes.shape[0]

        spect_logits = router_logits[..., :k_s]      # (B,N,k_s)
        spat_logits = router_logits[..., k_s:]       # (B,N,k_p)

        # 4) Top-k selection (small topk)
        # topk values and indices for spectral & spatial banks
        s_vals, s_idx = torch.topk(spect_logits, k=self.topk, dim=-1, largest=True, sorted=False)  # (B,N,topk)
        p_vals, p_idx = torch.topk(spat_logits,  k=self.topk, dim=-1, largest=True, sorted=False)

        # Softmax weights per-token/topk (temperature controls sharpness)
        w_s = F.softmax(s_vals / (self.temperature + 1e-8), dim=-1).unsqueeze(-1)  # (B,N,topk,1)
        w_p = F.softmax(p_vals / (self.temperature + 1e-8), dim=-1).unsqueeze(-1)

        # 5) Gather prototypes -> (B,N,D)
        # prototypes are small; gather_topk_prototypes uses advanced indexing (fast)
        spect_mixed = gather_topk_prototypes(self.spectral_prototypes, s_idx, w_s)  # (B,N,reduced_bands)
        spat_mixed = gather_topk_prototypes(self.spatial_prototypes, p_idx, w_p)     # (B,N,dim)

        # 6) Combine & project to keys
        # cat along feature dim and project: keep contiguous for bmm later
        combined = torch.cat([spect_mixed, spat_mixed], dim=-1)  # (B,N, reduced_bands + dim)
        keys = self.key_proj(combined)                           # (B,N,dim)

        # 7) Optionally add a learned per-token bias (small)
        # pos_bias broadcast (1,1,C) -> (B,N,C) via expand (no allocation when N=1)
        if keys.shape[1] == 1:
            keys = keys + self.pos_bias  # broadcast safe
        else:
            keys = keys + self.pos_bias.expand(1, keys.shape[1], -1)

        # 8) Attention blocks: use fast bmm-based matmuls
        x = x_tok
        for blk in self.blocks:
            x = blk(x, keys)

        # 9) Pooling + head
        feat = self.norm(x.mean(dim=1))  # (B, C)
        logits = self.head(feat)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    # -------------------------
    # Helpers
    # -------------------------
    @torch.no_grad()
    def get_latency(self, input_shape=(1, 103, 145, 145), device='cuda', repeats: int = 200, warmup: int = 30,
                    use_compile: bool = True):
        """
        Measure median latency (ms) using AMP & optional torch.compile.
        Run with device='cuda' for GPU timings. This warms up and then measures `repeats` runs.
        """
        self.eval()
        self.to(device)
        if use_compile and hasattr(torch, 'compile'):
            try:
                # compile with a simple wrapper for forward signature
                self = torch.compile(self)
                print("[info] torch.compile applied.")
            except Exception as e:
                print("[warn] torch.compile failed:", e)

        x = torch.randn(input_shape, device=device)
        # Warmup
        for _ in range(warmup):
            _ = self(x)

        # timing
        torch.cuda.synchronize()
        import time
        t0 = time.time()
        for _ in range(repeats):
            _ = self(x)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / repeats * 1000.0
        print(f"Avg latency over {repeats} runs: {elapsed:.3f} ms")
        return elapsed
