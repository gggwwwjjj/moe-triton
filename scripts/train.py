"""Train a small byte-level LM on a TinyStories slice (dense or MoE FFN).
usage: train.py RUN_NAME --data FILE [--moe] [--router torch|triton] [--aux 0.01] [--steps N]
"""
import argparse, json, math, os, sys, time, statistics
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moe_triton.moe import LM

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--data", required=True)
ap.add_argument("--moe", action="store_true")
ap.add_argument("--router", default="torch")
ap.add_argument("--dispatch", default="torch")
ap.add_argument("--experts", type=int, default=8)
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--cf", type=float, default=1.25)
ap.add_argument("--aux", type=float, default=0.01)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/train")
a = ap.parse_args()

torch.manual_seed(a.seed)
dev = "cuda"
data = np.fromfile(a.data, dtype=np.uint8)
n_val = len(data) // 50
train, val = torch.from_numpy(data[:-n_val].copy()), torch.from_numpy(data[-n_val:].copy())


def batch(src, gen):
    i = torch.randint(0, len(src) - a.ctx - 1, (a.bs,), generator=gen)
    x = torch.stack([src[j:j + a.ctx] for j in i]).long()
    y = torch.stack([src[j + 1:j + 1 + a.ctx] for j in i]).long()
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)


moe = dict(n_experts=a.experts, k=a.k, capacity_factor=a.cf, router=a.router, dispatch=a.dispatch, d_ff=768) if a.moe else None
model = LM(ctx=a.ctx, d_ff=1536, moe=moe).to(dev)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
sched = lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
gtrain = torch.Generator().manual_seed(a.seed)


@torch.no_grad()
def evaluate():
    model.eval()
    g = torch.Generator().manual_seed(123)
    ls = []
    for _ in range(20):
        x, y = batch(val, g)
        with torch.autocast("cuda", torch.bfloat16):
            ls.append(F.cross_entropy(model(x).float().flatten(0, 1), y.flatten()).item())
    model.train()
    return sum(ls) / len(ls)


log = dict(args=vars(a), n_params=n_params, gpu=torch.cuda.get_device_name(), steps=[])
os.makedirs(a.out, exist_ok=True)
step_ms = []
torch.cuda.reset_peak_memory_stats()
for s in range(a.steps + 1):
    if s % 500 == 0 or s == a.steps:
        vl = evaluate()
        log["steps"].append(dict(step=s, val_loss=vl))
        print(f"step {s} val {vl:.4f}", flush=True)
    if s == a.steps:
        break
    for gr in opt.param_groups:
        gr["lr"] = a.lr * sched(s)
    x, y = batch(train, gtrain)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.autocast("cuda", torch.bfloat16):
        loss = F.cross_entropy(model(x).float().flatten(0, 1), y.flatten())
        aux = sum(m.aux_loss for m in model.moe_layers()) if a.moe else torch.zeros((), device=dev)
    (loss + a.aux * aux).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); step_ms.append((time.perf_counter() - t0) * 1e3)
    if s % 50 == 0:
        rec = dict(step=s, loss=loss.item(), aux=aux.item())
        if a.moe:
            ms = model.moe_layers()
            rec["max_load"] = [round(m.stats["max_load"].item(), 3) for m in ms]
            rec["drop_frac"] = [round(m.stats["drop_frac"].item(), 4) for m in ms]
        log["steps"].append(rec)
        if s % 500 == 0:
            print(rec, flush=True)

log["median_step_ms"] = statistics.median(step_ms[20:])
log["peak_mem_MiB"] = torch.cuda.max_memory_allocated() / 2**20
log["final_val_loss"] = log["steps"][-1]["val_loss"]
json.dump(log, open(f"{a.out}/{a.name}.json", "w"), indent=1)
print(a.name, "params", n_params, "final val", log["final_val_loss"], "median step ms", log["median_step_ms"],
      "peak MiB", log["peak_mem_MiB"], flush=True)
