import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ---------------------------
# Helper modules / functions
# ---------------------------

class SpectralEncoder(nn.Module):
    """
    Encodes spectral bands per-pixel with small 1D conv stack.
    Input: (B, bands, H, W) -> Output: (B, spectral_dim, H, W)
    """
    def __init__(self, in_bands, spectral_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_bands, in_bands, kernel_size=3, padding=1, groups=1),  # mixing
            nn.GELU(),
            nn.Conv1d(in_bands, spectral_dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):
        # x: B, bands, H, W -> reshape to (B, bands, H*W) to apply 1D conv across bands
        B, bands, H, W = x.shape
        x_flat = x.view(B, bands, H * W)
        out = self.net(x_flat)  # B, spectral_dim, H*W
        out = out.view(B, out.shape[1], H, W)
        return out


def topk_mask_from_logits(logits, k):
    """
    logits: (..., M)
    return: masked probs where only top-k positions kept (others zero), renormalized
    """
    if k <= 0:
        return F.softmax(logits, dim=-1)
    vals, idx = torch.topk(logits, k, dim=-1)
    mask = torch.zeros_like(logits)
    mask.scatter_(-1, idx, 1.0)
    probs = F.softmax(logits, dim=-1) * mask
    # renormalize
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return probs / denom


# ---------------------------
# KeyBank Module
# ---------------------------

class DualKeyBank(nn.Module):
    """
    Maintains spectral and spatial prototype banks.
    Prototypes shape:
      K_spec: [heads, M_spec, head_dim]
      K_spat: [heads, M_spat, head_dim]
    Update via EMA or learnable (if ema_momentum is None, params are optimized)
    """
    def __init__(self, heads, head_dim, M_spec=16, M_spat=16, ema_momentum=0.99, device=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.M_spec = M_spec
        self.M_spat = M_spat
        self.ema_momentum = ema_momentum

        # prototypes: initialize small gaussian
        K_spec_init = 0.02 * torch.randn(heads, M_spec, head_dim)
        K_spat_init = 0.02 * torch.randn(heads, M_spat, head_dim)

        if ema_momentum is not None:
            # store as buffers (no gradients), updated via EMA in forward
            self.register_buffer('K_spec_buf', K_spec_init)
            self.register_buffer('K_spat_buf', K_spat_init)
            self.learnable = False
        else:
            # learnable prototypes (parameters)
            self.K_spec = nn.Parameter(K_spec_init)
            self.K_spat = nn.Parameter(K_spat_init)
            self.learnable = True

    def get_prototypes(self):
        if self.learnable:
            return self.K_spec, self.K_spat
        else:
            return self.K_spec_buf, self.K_spat_buf

    @torch.no_grad()
    def ema_update(self, agg_spec, agg_spat, counts_spec, counts_spat):
        """
        agg_spec: tensor [heads, M_spec, head_dim] (aggregated token features assigned to each prototype)
        counts_spec: [heads, M_spec] counts used for normalization (float)
        Similar for spat.
        We compute mean by agg / counts (taking care of zeros).
        """
        if self.ema_momentum is None:
            return  # nothing to do

        K_spec_old = self.K_spec_buf
        K_spat_old = self.K_spat_buf

        # normalize aggregates
        counts_spec = counts_spec.clamp_min(1e-6).unsqueeze(-1)  # [h, M, 1]
        counts_spat = counts_spat.clamp_min(1e-6).unsqueeze(-1)

        mean_spec = agg_spec / counts_spec
        mean_spat = agg_spat / counts_spat

        # Some prototypes might have zero counts; keep old value for those (counts_small handled by clamp)
        momentum = self.ema_momentum
        K_spec_new = (1.0 - momentum) * mean_spec + momentum * K_spec_old
        K_spat_new = (1.0 - momentum) * mean_spat + momentum * K_spat_old

        # write back
        self.K_spec_buf.copy_(K_spec_new)
        self.K_spat_buf.copy_(K_spat_new)


# ---------------------------
# HSIKeyBankAttention
# ---------------------------

class HSIKeyBankAttention(nn.Module):
    """
    Attention that uses dual key banks (spectral + spatial).
    Q, V are computed from tokens as usual; K is constructed from prototypes routed per-token.
    Returns: output tokens [B, N, C] and auxiliary losses if enabled.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 M_spec=16, M_spat=16, topk_spec=3, topk_spat=3,
                 ema_momentum=0.99, use_contrastive=True, contrastive_tau=0.1,
                 diversity_reg=1e-2):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # routing projections (token -> logits for prototypes)
        # We'll compute logits for each head in parallel by projecting dim-> heads*M
        self.route_spec = nn.Linear(dim, num_heads * M_spec)
        self.route_spat = nn.Linear(dim, num_heads * M_spat)

        # KeyBank
        self.keybank = DualKeyBank(num_heads, self.head_dim, M_spec=M_spec, M_spat=M_spat, ema_momentum=ema_momentum)

        # topk sparsity
        self.topk_spec = topk_spec
        self.topk_spat = topk_spat

        # contrastive + diversity
        self.use_contrastive = use_contrastive
        self.contrastive_tau = contrastive_tau
        self.diversity_reg = diversity_reg

        # small projection head for contrastive (token->proj and prototype->proj)
        if use_contrastive:
            proj_dim = self.head_dim
            self.token_proj = nn.Linear(self.head_dim, proj_dim)
            self.proto_proj = nn.Linear(self.head_dim, proj_dim)

    def forward(self, x, spectral_tokens=None):
        """
        x: [B, N, C] token embeddings (spatial tokens after patch_embed)
        spectral_tokens: [B, N, head_dim * num_heads?] or None
          - If you want to route using spectral-only features, pass them. Otherwise route from x.
        Returns dict: {"out": out_tokens, "aux_losses": {...}} optionally updates prototypes via EMA.
        """

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, N, head_dim]

        # flatten token features for routing: use x (B,N,C) then project to head-specific logits
        # route_spec_logits: [B, heads, N, M_spec]
        r_spec_logits = self.route_spec(x).view(B, N, self.num_heads, -1).permute(0,2,1,3)  # [B, heads, N, M_spec]
        r_spat_logits = self.route_spat(x).view(B, N, self.num_heads, -1).permute(0,2,1,3)  # [B, heads, N, M_spat]

        # apply topk masking + softmax
        r_spec_masked = topk_mask_from_logits(r_spec_logits, self.topk_spec)
        r_spat_masked = topk_mask_from_logits(r_spat_logits, self.topk_spat)

        # prototypes
        K_spec, K_spat = self.keybank.get_prototypes()  # shapes: [heads, M_spec, head_dim], [heads, M_spat, head_dim]

        # build effective keys per token by weighted sum over prototypes
        # r_spec_masked: [B, heads, N, M_spec], K_spec: [heads, M_spec, head_dim]
        # desired K_eff_spec: [B, heads, N, head_dim] = r_spec @ K_spec
        K_spec_exp = K_spec.unsqueeze(0).expand(B, -1, -1, -1)  # [B, heads, M_spec, head_dim]
        K_spat_exp = K_spat.unsqueeze(0).expand(B, -1, -1, -1)

        K_eff_spec = torch.einsum('bhnm,bhmk->bhnk', r_spec_masked, K_spec_exp)  # [B, heads, N, head_dim]
        K_eff_spat = torch.einsum('bhnm,bhmk->bhnk', r_spat_masked, K_spat_exp)

        # combine spectral + spatial keys (simple sum; you can make this learnable)
        K_eff = K_eff_spec + K_eff_spat

        # Linearized attention similar to your original but with K_eff
        # softmax across token dimension in keys (we follow same pattern as EfficientAttention)
        K_norm = K_eff.softmax(dim=-2)  # softmax over N tokens dimension (dim -2 because K_eff is [B, h, N, d])
        # context: bhde = sum_n k * v
        context = torch.einsum('bhnd,bhne->bhde', K_norm, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)

        aux = {}

        # -----------------------------
        # Contrastive & diversity losses
        # -----------------------------
        if self.use_contrastive:
            # project token features per-head: take v (or q) as token features; use v here
            # tokens_proj: [B, heads, N, proj_dim]
            tokens_proj = self.token_proj(v)  # v: [B,heads,N,D]
            # prototypes proj: [heads, M, proj_dim] -> expand B
            prot_spec_proj = self.proto_proj(K_spec)  # [heads, M_spec, D]
            prot_spat_proj = self.proto_proj(K_spat)

            # compute positive similarities between token and its assigned prototypes (use r_masked)
            # First, L2 normalize
            t_norm = F.normalize(tokens_proj, dim=-1)
            p_spec_norm = F.normalize(prot_spec_proj, dim=-1)
            p_spat_norm = F.normalize(prot_spat_proj, dim=-1)

            # compute sim token-proto: [B,heads,N,M]
            sim_spec = torch.einsum('bhnd,hmd->bhnm', t_norm, p_spec_norm)
            sim_spat = torch.einsum('bhnd,hmd->bhnm', t_norm, p_spat_norm)

            # positive score is average similarity to prototypes weighted by routing probabilities
            pos_spec = (sim_spec * r_spec_masked).sum(dim=-1)  # [B,heads,N]
            pos_spat = (sim_spat * r_spat_masked).sum(dim=-1)

            # negatives: in-batch other prototypes - we will compute logits over prot dims
            # For stability, compute NT-Xent like loss: -log( exp(pos / tau) / sum_exp_over_all_prototypes )
            all_sim_spec = sim_spec  # b,h,n,m
            denom_spec = torch.logsumexp(all_sim_spec / self.contrastive_tau, dim=-1)  # [B,heads,N]
            loss_spec = - (pos_spec / self.contrastive_tau - denom_spec).mean()

            all_sim_spat = sim_spat
            denom_spat = torch.logsumexp(all_sim_spat / self.contrastive_tau, dim=-1)
            loss_spat = - (pos_spat / self.contrastive_tau - denom_spat).mean()

            aux['contrastive_loss'] = 0.5 * (loss_spec + loss_spat)

            # diversity reg: prototypes within each head should be diverse (low cosine similarity)
            # pairwise cosine among prototypes
            def diversity_loss(protos):
                # protos: [h, M, d]
                h, M, d = protos.shape
                p = protos / (protos.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                # compute pairwise similarity excluding diagonal
                sim = torch.einsum('hmd,hnd->hmn', p, p)  # [h,M,M]
                mask = 1.0 - torch.eye(M, device=sim.device).unsqueeze(0)
                loss = (sim * mask).sum() / (h * M * (M - 1) + 1e-6)
                return loss
            aux['diversity_loss'] = self.diversity_reg * (diversity_loss(K_spec) + diversity_loss(K_spat))

        # -----------------------------
        # EMA prototype update (no grad)
        # -----------------------------
        if not self.keybank.learnable:
            # We need to compute aggregated token features assigned to each prototype to update prototypes.
            # Aggregate token features in the head-dim space (use v as representative feature)
            with torch.no_grad():
                # v: [B,heads,N,D]
                # r_spec_masked: [B,heads,N,M_spec]
                # compute agg_spec: sum over B,N of r * v -> [heads, M_spec, D]
                B_, H, N_, D = v.shape
                agg_spec = torch.einsum('bhnm,bhnd->hmd', r_spec_masked.sum(dim=0), v.sum(dim=0))
                # The above is an approximate batch-aggregated update. A more precise approach is:
                # agg_spec = torch.einsum('bhnm,bhnd->bhmd' ...) then sum over b to get [h,m,d], but to keep memory good do:
                # compute per-batch aggregated and sum across batch
                # We'll do a careful aggregation below:
                agg_spec_precise = torch.zeros(self.num_heads, self.keybank.M_spec, self.head_dim, device=v.device)
                counts_spec = torch.zeros(self.num_heads, self.keybank.M_spec, device=v.device)
                for b in range(B):
                    # r_spec_masked[b]: [h, N, M_spec] ; v[b]: [h, N, D]
                    r_b = r_spec_masked[b]  # [h, N, M]
                    v_b = v[b]  # [h, N, D]
                    # compute contribution: (r_b.transpose N,M) @ v_b => [h, M, D]
                    contrib = torch.einsum('hnm,hnd->hmd', r_b, v_b)
                    agg_spec_precise += contrib
                    counts_spec += r_b.sum(dim=1)  # sum over tokens -> [h, M]
                # same for spatial
                agg_spat_precise = torch.zeros(self.num_heads, self.keybank.M_spat, self.head_dim, device=v.device)
                counts_spat = torch.zeros(self.num_heads, self.keybank.M_spat, device=v.device)
                for b in range(B):
                    r_b = r_spat_masked[b]
                    v_b = v[b]
                    contrib = torch.einsum('hnm,hnd->hmd', r_b, v_b)
                    agg_spat_precise += contrib
                    counts_spat += r_b.sum(dim=1)

                # Perform EMA update on keybank
                self.keybank.ema_update(agg_spec_precise, agg_spat_precise, counts_spec, counts_spat)

        return {"out": out, "aux": aux}


# ---------------------------
# Updated ViT block & model using HSIKeyBankAttention
# ---------------------------

class newViTBlockHSI(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, **attn_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HSIKeyBankAttention(dim, num_heads, qkv_bias=qkv_bias, **attn_kwargs)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        attn_res = self.attn(self.norm1(x))
        x = x + attn_res["out"]
        aux = attn_res.get("aux", {})
        x = x + self.mlp(self.norm2(x))
        return x, aux


class newFastViTHSI(nn.Module):
    def __init__(self, image_size=5, patch_size=1, num_channels=103, num_classes=9,
                 embed_dim=768, depth=6, num_heads=12, mlp_ratio=4.,
                 spectral_dim=64, keybank_kwargs=None):
        super().__init__()
        self.spectral_encoder = SpectralEncoder(num_channels, spectral_dim=spectral_dim)
        # After spectral encoder, we may concatenate spectral features to patch embeddings
        # We'll implement patch_embed to take (spectral features stacked as channels)
        self.patch_embed = nn.Conv2d(spectral_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim))

        if keybank_kwargs is None:
            keybank_kwargs = {
                "M_spec": 16, "M_spat": 16,
                "topk_spec": 3, "topk_spat": 3,
                "ema_momentum": 0.995,
                "use_contrastive": True,
                "contrastive_tau": 0.1,
                "diversity_reg": 1e-2
            }
        self.blocks = nn.ModuleList([newViTBlockHSI(embed_dim, num_heads, mlp_ratio, qkv_bias=False, **keybank_kwargs) for _ in range(depth)])
        self.spectral_attention = nn.Sequential(  # small head on pooled features (optional)
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, labels=None):
        """
        x: [B, bands, H, W]
        """
        # 1) spectral encoding per pixel
        sp = self.spectral_encoder(x)  # [B, spectral_dim, H, W]

        # 2) patch embed
        tokens = self.patch_embed(sp)  # [B, embed_dim, H', W']
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, N, C]
        tokens = tokens + self.pos_embed

        aux_losses = {}
        for block in self.blocks:
            tokens, aux = block(tokens)
            # accumulate aux losses
            for k, v in aux.items():
                aux_losses.setdefault(k, 0.0)
                aux_losses[k] = aux_losses[k] + v

        pooled = tokens.mean(dim=1)
        pooled = self.spectral_attention(pooled)
        logits = self.head(self.norm(pooled))

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            ce_loss = loss_fn(logits, labels)
            total_loss = ce_loss
            # add aux losses with weights (you can tweak these)
            if aux_losses:
                total_loss = total_loss + 1.0 * aux_losses.get("contrastive_loss", 0.0) + aux_losses.get("diversity_loss", 0.0)
            return {"loss": total_loss, "logits": logits, "aux": aux_losses}
        return {"logits": logits, "aux": aux_losses}

# ---------------------------
# Quick sanity test snippet (run after saving file)
# ---------------------------

if __name__ == "__main__":
    # tiny random forward/backward test
    B = 2
    bands = 103
    H = W = 5
    model = newFastViTHSI(image_size=5, patch_size=1, num_channels=bands, num_classes=9,
                          embed_dim=64, depth=2, num_heads=4, spectral_dim=32)
    x = torch.randn(B, bands, H, W)
    labels = torch.randint(0, 9, (B,))
    out = model(x, labels)
    print("loss:", out["loss"].item())
    out["loss"].backward()
    print("backward OK")
