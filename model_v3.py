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
