"""Timing on one GPU. Each case: 25 warmup iters, then 100 timed iters (CUDA
events around each iter), report the median.

Part 1: router alone, forward and forward+backward, for
        PyTorch eager / torch.compile(reference) / Triton kernel.
Part 2: whole MoE layer forward+backward (bf16) with torch vs triton router,
        time + peak memory, and a torch.profiler table for one config.
"""
import json, sys, os, statistics
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_triton.router import torch_router, triton_router
from moe_triton.moe import MoE

dev = "cuda"
WARM, ITERS = 25, 100
torch._dynamo.config.cache_size_limit = 512  # one graph per shape; avoid silent eager fallback
compiled_router = torch.compile(torch_router, dynamic=False)


def timeit(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def bench_router(rows):
    impls = {"eager": torch_router, "compile": compiled_router, "triton": triton_router}
    for dtype in (torch.bfloat16, torch.float32):
        for E in (8, 16, 64, 128):
            for T in (4096, 16384, 65536, 262144):
                k = 2
                x = torch.randn(T, E, device=dev, dtype=dtype, requires_grad=True)
                gv = torch.randn(T, k, device=dev)
                gp = torch.randn(E, device=dev)
                r = dict(T=T, E=E, k=k, dtype=str(dtype).split(".")[-1])
                for name, f in impls.items():
                    def fwd():
                        with torch.no_grad():
                            f(x, k)

                    def fwdbwd():
                        v, _, p, _ = f(x, k)
                        torch.autograd.backward([v, p], [gv, gp])
                        x.grad = None
                    try:
                        r[f"{name}_fwd_ms"] = timeit(fwd)
                        r[f"{name}_fwdbwd_ms"] = timeit(fwdbwd)
                    except Exception as e:  # report, don't hide
                        r[f"{name}_error"] = repr(e)[:200]
                rows.append(r)
                print(r, flush=True)


def bench_layer(rows, prof_out):
    for E, d, d_ff, T in ((8, 1024, 2048, 16384), (64, 1024, 512, 16384), (128, 512, 256, 32768)):
        for router, disp in (("torch", "torch"), ("triton", "torch"), ("triton", "triton")):
            torch.manual_seed(0)
            m = MoE(d, d_ff, E, 2, 1.25, router=router, dispatch=disp).to(dev, torch.bfloat16)
            x = torch.randn(T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
            g = torch.randn_like(x)

            def step():
                y = m(x)
                (y * g).sum().add(m.aux_loss).backward()
                x.grad = None
                for p in m.parameters():
                    p.grad = None
            ms = timeit(step)
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            step(); torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20
            r = dict(E=E, d=d, d_ff=d_ff, T=T, k=2, router=router, dispatch=disp, layer_fwdbwd_ms=ms, peak_extra_MiB=peak)
            rows.append(r)
            print(r, flush=True)
            if E == 64:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(10):
                        step()
                    torch.cuda.synchronize()
                with open(prof_out, "a") as fh:
                    fh.write(f"\n===== MoE layer E={E} d={d} d_ff={d_ff} T={T} router={router} dispatch={disp} (10 fwd+bwd steps) =====\n")
                    fh.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results"
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    rr, lr = [], []
    bench_router([])  # full warm-up pass (first cases of a fresh job were ~0.7 ms slower); discarded
    bench_router(rr)
    json.dump(rr, open(f"{out}/bench_router.json", "w"), indent=1)
    open(f"{out}/profile_layer.txt", "w").close()
    bench_layer(lr, f"{out}/profile_layer.txt")
    json.dump(lr, open(f"{out}/bench_layer.json", "w"), indent=1)
