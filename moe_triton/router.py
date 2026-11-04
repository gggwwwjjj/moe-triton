"""MoE router: softmax over experts + top-k, plus the two statistics the
load-balancing aux loss needs (per-expert prob sum and per-expert token count).

Two implementations with the same interface:
  torch_router(logits, k)  -> PyTorch reference (softmax, topk, sum, bincount)
  triton_router(logits, k) -> one fused Triton kernel forward, one fused backward

Both return (topv [T,k] fp32, topi [T,k] int64, psum [E] fp32, counts [E] int32).
Softmax is always computed in fp32, whatever the logits dtype.
"""
import torch
import triton
import triton.language as tl


def torch_router(logits, k):
    p = logits.float().softmax(dim=-1)
    topv, topi = p.topk(k, dim=-1)
    psum = p.sum(dim=0)
    counts = torch.bincount(topi.flatten(), minlength=logits.shape[1]).to(torch.int32)
    return topv, topi, psum, counts


def _block_t(E):
    # ~4K fp32 elements per program; atomics make autotune (repeated launches) unsafe in fwd
    return max(16, min(256, 4096 // triton.next_power_of_2(E)))


@triton.jit
def _router_fwd(x_ptr, topv_ptr, topi_ptr, psum_ptr, cnt_ptr, T, E, stride_t,
                K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, BLOCK_E)
    rmask = rows < T
    cmask = cols < E
    mask = rmask[:, None] & cmask[None, :]
    x = tl.load(x_ptr + rows[:, None] * stride_t + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    x = tl.where(cmask[None, :], x, -float("inf"))
    m = tl.max(x, axis=1)
    e = tl.exp(x - m[:, None])
    p = e / tl.sum(e, axis=1)[:, None]
    p = tl.where(mask, p, 0.0)
    tl.atomic_add(psum_ptr + cols, tl.sum(p, axis=0), mask=cmask)
    work = tl.where(mask, p, -1.0)
    for j in tl.static_range(K):
        v = tl.max(work, axis=1)
        i = tl.argmax(work, axis=1)
        tl.store(topv_ptr + rows * K + j, v, mask=rmask)
        tl.store(topi_ptr + rows * K + j, i.to(tl.int64), mask=rmask)
        hit = (cols[None, :] == i[:, None]) & rmask[:, None]
        tl.atomic_add(cnt_ptr + cols, tl.sum(hit.to(tl.int32), axis=0), mask=cmask)
        work = tl.where(hit, -1.0, work)


@triton.jit
def _router_bwd(x_ptr, topi_ptr, gv_ptr, gps_ptr, gx_ptr, T, E, stride_t, stride_gx,
                K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, BLOCK_E)
    rmask = rows < T
    cmask = cols < E
    mask = rmask[:, None] & cmask[None, :]
    # recompute softmax instead of saving probs
    x = tl.load(x_ptr + rows[:, None] * stride_t + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    x = tl.where(cmask[None, :], x, -float("inf"))
    m = tl.max(x, axis=1)
    e = tl.exp(x - m[:, None])
    p = e / tl.sum(e, axis=1)[:, None]
    p = tl.where(mask, p, 0.0)
    # dL/dp = grad from psum (same for every row) + scattered grad from top-k values
    gps = tl.load(gps_ptr + cols, mask=cmask, other=0.0)
    dp = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32) + gps[None, :]
    for j in tl.static_range(K):
        i = tl.load(topi_ptr + rows * K + j, mask=rmask, other=0)
        g = tl.load(gv_ptr + rows * K + j, mask=rmask, other=0.0)
        dp += tl.where(cols[None, :] == i[:, None], g[:, None], 0.0)
    dot = tl.sum(p * dp, axis=1)
    gx = p * (dp - dot[:, None])
    tl.store(gx_ptr + rows[:, None] * stride_gx + cols[None, :], gx.to(gx_ptr.dtype.element_ty), mask=mask)


class _TritonRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, k):
        assert logits.dim() == 2 and logits.stride(1) == 1
        T, E = logits.shape
        dev = logits.device
        topv = torch.empty(T, k, device=dev, dtype=torch.float32)
        topi = torch.empty(T, k, device=dev, dtype=torch.int64)
        psum = torch.zeros(E, device=dev, dtype=torch.float32)
        counts = torch.zeros(E, device=dev, dtype=torch.int32)
        BLOCK_E = triton.next_power_of_2(E)
        BT = _block_t(E)
        _router_fwd[(triton.cdiv(T, BT),)](logits, topv, topi, psum, counts, T, E, logits.stride(0),
                                            K=k, BLOCK_T=BT, BLOCK_E=BLOCK_E, num_warps=4)
        ctx.save_for_backward(logits, topi)
        ctx.k = k
        ctx.mark_non_differentiable(topi, counts)
        return topv, topi, psum, counts

    @staticmethod
    def backward(ctx, gv, _gi, gps, _gc):
        logits, topi = ctx.saved_tensors
        T, E = logits.shape
        gv = torch.zeros_like(topi, dtype=torch.float32) if gv is None else gv.float().contiguous()
        gps = torch.zeros(E, device=logits.device) if gps is None else gps.float().contiguous()
        gx = torch.empty_like(logits)
        BLOCK_E = triton.next_power_of_2(E)
        BT = _block_t(E)
        _router_bwd[(triton.cdiv(T, BT),)](logits, topi, gv, gps, gx, T, E, logits.stride(0), gx.stride(0),
                                            K=ctx.k, BLOCK_T=BT, BLOCK_E=BLOCK_E, num_warps=4)
        return gx, None


def triton_router(logits, k):
    return _TritonRouter.apply(logits.contiguous(), k)
