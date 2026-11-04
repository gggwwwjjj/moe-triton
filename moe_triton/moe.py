"""MoE FFN layer (top-k routing, Switch-style load-balancing aux loss, capacity
factor with token dropping) and a small GPT-style language model that uses it.

Dispatch is sync-free: tokens are sorted by expert, each expert gets a fixed
buffer of `capacity` slots, experts run as one batched GEMM over [E, cap, d].
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .router import torch_router, triton_router
from .dispatch import Dispatch, Combine

ROUTERS = {"torch": torch_router, "triton": triton_router}


class MoE(nn.Module):
    def __init__(self, d, d_ff, n_experts=8, k=2, capacity_factor=1.25, router="torch", dispatch="torch"):
        super().__init__()
        self.E, self.k, self.cf = n_experts, k, capacity_factor
        self.router, self.dispatch = router, dispatch
        self.gate = nn.Linear(d, n_experts, bias=False)
        self.w1 = nn.Parameter(torch.randn(n_experts, d, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_experts, d_ff, d) * 0.02 / math.sqrt(2))
        self.stats = {}

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        T, d = x.shape
        E, k = self.E, self.k
        logits = self.gate(x).float()  # router math in fp32
        topv, topi, psum, counts = ROUTERS[self.router](logits, k)
        w = topv / topv.sum(-1, keepdim=True)  # renormalise over the chosen k

        # Switch Transformer aux loss: E * sum_e f_e * P_e
        f = counts.float() / (T * k)
        P = psum / T
        self.aux_loss = E * (f.detach() * P).sum()

        cap = max(1, int(math.ceil(self.cf * T * k / E)))
        flat_e = topi.reshape(-1)
        order = torch.argsort(flat_e, stable=True)  # earlier tokens win when over capacity
        sorted_e = flat_e[order]
        starts = torch.cumsum(counts, 0) - counts
        pos = torch.arange(T * k, device=x.device) - starts[sorted_e]
        keep = pos < cap
        slot = torch.where(keep, sorted_e * cap + pos, E * cap)  # dropped -> dummy slot
        tok = order // k

        if self.dispatch == "triton":
            slot_tk = torch.empty_like(order).scatter_(0, order, torch.where(keep, slot, -1))
            src_tk = torch.full((E * cap + 1,), -1, device=x.device, dtype=order.dtype)
            src_tk = src_tk.scatter_(0, slot, order)[:-1]  # dropped ones land in the dummy slot
            buf = Dispatch.apply(x, src_tk, slot_tk, k)
            h = torch.bmm(buf.view(E, cap, d), self.w1)
            h = torch.bmm(F.gelu(h), self.w2).view(E * cap, d)
            out = Combine.apply(h, w, src_tk, slot_tk, k).to(x.dtype)
        else:
            buf = x.new_zeros(E * cap + 1, d)
            buf[slot] = x[tok]
            h = torch.bmm(buf[:-1].view(E, cap, d), self.w1)
            h = torch.bmm(F.gelu(h), self.w2).view(E * cap, d)
            h = torch.cat([h, h.new_zeros(1, d)])
            y = h[slot] * (w.reshape(-1)[order] * keep).unsqueeze(-1).to(h.dtype)
            out = x.new_zeros(T, d).index_add_(0, tok, y.to(x.dtype))

        with torch.no_grad():
            self.stats = {"drop_frac": 1 - keep.float().mean(), "max_load": counts.max().float() / (T * k / E)}
        return out.view(shape)


class Block(nn.Module):
    def __init__(self, d, n_head, mlp):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.n_head = n_head
        self.mlp = mlp

    def forward(self, x):
        B, S, d = x.shape
        q, kk, v = self.qkv(self.ln1(x)).view(B, S, 3, self.n_head, d // self.n_head).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, kk, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, S, d))
        return x + self.mlp(self.ln2(x))


def dense_mlp(d, d_ff):
    return nn.Sequential(nn.Linear(d, d_ff, bias=False), nn.GELU(), nn.Linear(d_ff, d, bias=False))


class LM(nn.Module):
    """Byte-level causal LM. moe=None -> dense FFN in every block."""

    def __init__(self, vocab=256, d=384, n_layer=6, n_head=6, ctx=256, d_ff=1536, moe=None):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        mk = (lambda: MoE(d, **moe)) if moe else (lambda: dense_mlp(d, d_ff))
        self.blocks = nn.ModuleList(Block(d, n_head, mk()) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def moe_layers(self):
        return [b.mlp for b in self.blocks if isinstance(b.mlp, MoE)]
