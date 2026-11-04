"""Triton token dispatch (tokens -> expert capacity slots) and combine
(expert outputs -> tokens, weighted by router weights), forward and backward.

Everything is written as row gathers, so no atomics and deterministic results:
  slot_tk [T*k]   capacity slot of assignment (t, j), -1 if dropped
  src_tk  [E*cap] assignment t*k+j that fills each slot, -1 if empty
dispatch fwd : buf[s]  = x[src_tk[s] // k]
dispatch bwd : dx[t]   = sum_j dbuf[slot_tk[t, j]]
combine  fwd : out[t]  = sum_j w[t, j] * h[slot_tk[t, j]]
combine  bwd : dh[s]   = w[src_tk[s]] * dout[src_tk[s] // k]
               dw[t,j] = <dout[t], h[slot_tk[t, j]]>
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_rows(src_ptr, idx_ptr, scale_ptr, out_ptr, D, HAS_SCALE: tl.constexpr, BLOCK_D: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    i = tl.load(idx_ptr + r).to(tl.int64)
    if HAS_SCALE:
        s = tl.load(scale_ptr + r).to(tl.float32)
    else:
        s = 1.0
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        m = offs < D
        v = tl.load(src_ptr + i * D + offs, mask=m & (i >= 0), other=0.0).to(tl.float32)
        tl.store(out_ptr + r * D + offs, (v * s).to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _gather_sum_k(src_ptr, idx_ptr, scale_ptr, out_ptr, D, K: tl.constexpr, HAS_SCALE: tl.constexpr,
                  BLOCK_D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        m = offs < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for j in tl.static_range(K):
            i = tl.load(idx_ptr + t * K + j).to(tl.int64)
            v = tl.load(src_ptr + i * D + offs, mask=m & (i >= 0), other=0.0).to(tl.float32)
            if HAS_SCALE:
                v = v * tl.load(scale_ptr + t * K + j).to(tl.float32)
            acc += v
        tl.store(out_ptr + t * D + offs, acc.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _rowdot_k(a_ptr, b_ptr, idx_ptr, out_ptr, D, K: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    for j in tl.static_range(K):
        i = tl.load(idx_ptr + t * K + j).to(tl.int64)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            m = offs < D
            a = tl.load(a_ptr + t * D + offs, mask=m, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + i * D + offs, mask=m & (i >= 0), other=0.0).to(tl.float32)
            acc += a * b
        tl.store(out_ptr + t * K + j, tl.sum(acc, axis=0))


def _bd(D):
    return min(1024, triton.next_power_of_2(D))


def gather_rows(src, idx, scale=None, out_dtype=None):
    out = torch.empty(idx.numel(), src.shape[1], device=src.device, dtype=out_dtype or src.dtype)
    _gather_rows[(idx.numel(),)](src, idx, scale if scale is not None else idx, out, src.shape[1],
                                 HAS_SCALE=scale is not None, BLOCK_D=_bd(src.shape[1]), num_warps=4)
    return out


def gather_sum_k(src, idx, k, scale=None, out_dtype=None):
    T = idx.numel() // k
    out = torch.empty(T, src.shape[1], device=src.device, dtype=out_dtype or src.dtype)
    _gather_sum_k[(T,)](src, idx, scale if scale is not None else idx, out, src.shape[1],
                        K=k, HAS_SCALE=scale is not None, BLOCK_D=_bd(src.shape[1]), num_warps=4)
    return out


class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, src_tk, slot_tk, k):
        ctx.save_for_backward(src_tk, slot_tk)
        ctx.k = k
        src_tok = torch.where(src_tk >= 0, src_tk // k, -1)
        return gather_rows(x.contiguous(), src_tok)

    @staticmethod
    def backward(ctx, gbuf):
        src_tk, slot_tk = ctx.saved_tensors
        return gather_sum_k(gbuf.contiguous(), slot_tk, ctx.k), None, None, None


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, src_tk, slot_tk, k):
        h, w = h.contiguous(), w.contiguous()
        ctx.save_for_backward(h, w, src_tk, slot_tk)
        ctx.k = k
        return gather_sum_k(h, slot_tk, k, scale=w)

    @staticmethod
    def backward(ctx, gout):
        h, w, src_tk, slot_tk = ctx.saved_tensors
        k = ctx.k
        gout = gout.contiguous()
        valid = src_tk >= 0
        src_tok = torch.where(valid, src_tk // k, -1)
        w_slot = torch.where(valid, w.reshape(-1)[src_tk.clamp(min=0)], 0.0)
        gh = gather_rows(gout, src_tok, scale=w_slot, out_dtype=h.dtype)
        gw = torch.empty_like(w, dtype=torch.float32)
        _rowdot_k[(w.shape[0],)](gout, h, slot_tk, gw, h.shape[1], K=k, BLOCK_D=_bd(h.shape[1]), num_warps=4)
        return gh, gw.to(w.dtype), None, None, None
