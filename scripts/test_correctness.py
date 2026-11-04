"""Triton router vs PyTorch reference: forward outputs and logits gradients.
Also checks a full MoE layer (output + all parameter grads) with each router."""
import json, sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_triton.router import torch_router, triton_router
from moe_triton.moe import MoE

torch.manual_seed(0)
dev = "cuda"
TOL = {torch.float32: dict(atol=1e-6, rtol=1e-5), torch.bfloat16: dict(atol=1e-2, rtol=1e-2)}
rows = []


def maxerr(a, b):
    return (a.float() - b.float()).abs().max().item()


def check_router(T, E, k, dtype):
    x = torch.randn(T, E, device=dev, dtype=dtype) * 2
    xr = x.clone().requires_grad_()
    xt = x.clone().requires_grad_()
    rv, ri, rp, rc = torch_router(xr, k)
    tv, ti, tp, tc = triton_router(xt, k)
    gv = torch.randn_like(rv)
    gp = torch.randn_like(rp)
    # Ties (frequent with bf16 logits) may be ordered differently by torch.topk and
    # the kernel. So: the kernel's choice is valid iff its values equal torch's
    # top-k values; the reference gradient is taken w.r.t. the kernel's indices.
    p_ref = xr.float().softmax(-1)
    torch.autograd.backward([p_ref.gather(1, ti), p_ref.sum(0)], [gv, gp])
    torch.autograd.backward([tv, tp], [gv, gp])
    idx_diff_rows = (ri != ti).any(-1).sum().item()
    valid = torch.equal(p_ref.detach().gather(1, ti), rv.detach()) and bool((ti.sort(-1).values.diff(dim=-1) != 0).all())
    fwd_tol = TOL[torch.float32]  # softmax is fp32 in both paths, for both input dtypes
    ok = (valid and torch.equal(rc, tc)
          and torch.allclose(rv, tv, **fwd_tol)
          and torch.allclose(rp, tp, atol=1e-6 * T, rtol=1e-5)  # sum over T rows, atomic order differs
          and torch.allclose(xr.grad.float(), xt.grad.float(), **TOL[dtype]))
    r = dict(test="router", T=T, E=E, k=k, dtype=str(dtype).split(".")[-1],
             topv_maxerr=maxerr(rv, tv), psum_maxerr=maxerr(rp, tp),
             tie_order_rows=idx_diff_rows, topk_valid=valid, counts_equal=bool(torch.equal(rc, tc)),
             grad_maxerr=maxerr(xr.grad, xt.grad), ok=bool(ok))
    rows.append(r)
    print(r, flush=True)


def check_layer(T, d, d_ff, E, k, dtype, dispatch="torch", cf=1.25):
    torch.manual_seed(1)
    a = MoE(d, d_ff, E, k, cf, router="torch").to(dev, dtype)
    b = MoE(d, d_ff, E, k, cf, router="triton", dispatch=dispatch).to(dev, dtype)
    b.load_state_dict(a.state_dict())
    x = torch.randn(T, d, device=dev, dtype=dtype)
    g = torch.randn(T, d, device=dev, dtype=dtype)
    ya, yb = a(x), b(x)
    (ya * g).sum().add(a.aux_loss).backward()
    (yb * g).sum().add(b.aux_loss).backward()
    gerr = max(maxerr(pa.grad, pb.grad) / (pa.grad.float().abs().max().item() + 1e-12)
               for pa, pb in zip(a.parameters(), b.parameters()))
    tol = TOL[dtype]
    ok = torch.allclose(ya.float(), yb.float(), **tol) and gerr < (1e-4 if dtype == torch.float32 else 2e-2)
    r = dict(test="moe_layer", dispatch=dispatch, cf=cf, T=T, d=d, E=E, k=k, dtype=str(dtype).split(".")[-1],
             out_maxerr=maxerr(ya, yb), aux_diff=abs(a.aux_loss.item() - b.aux_loss.item()),
             param_grad_max_rel_err=gerr, drop_frac=a.stats["drop_frac"].item(), ok=bool(ok))
    rows.append(r)
    print(r, flush=True)


if __name__ == "__main__":
    for dtype in (torch.float32, torch.bfloat16):
        for T in (1, 7, 4096, 65536):
            for E in (8, 16, 64, 128, 6):
                for k in (1, 2, 4):
                    check_router(T, E, k, dtype)
    for dtype in (torch.float32, torch.bfloat16):
        for disp in ("torch", "triton"):
            check_layer(4096, 256, 512, 8, 2, dtype, disp)
            check_layer(4096, 256, 512, 64, 2, dtype, disp)
            check_layer(4096, 256, 512, 64, 4, dtype, disp)
            check_layer(1000, 200, 300, 16, 2, dtype, disp)  # odd sizes, D not a power of 2
            check_layer(4096, 256, 512, 8, 2, dtype, disp, cf=0.5)  # heavy token dropping
    out = sys.argv[1] if len(sys.argv) > 1 else "results/correctness.json"
    json.dump(rows, open(out, "w"), indent=1)
    n_bad = sum(not r["ok"] for r in rows)
    print(f"{len(rows) - n_bad}/{len(rows)} passed")
