# MoE layer in PyTorch + Triton kernels for routing and token dispatch

Single-GPU project (NVIDIA A40 on a PBS cluster; torch 2.10.0+cu128, triton 3.6.0), 2026-11-02.
All numbers below come from the files in `results/` and `logs/` (`logs/bench.log`: tests + timing job, `logs/train.log`, `logs/train_full.log`: training jobs).

## What is here

| file | content |
|---|---|
| `moe_triton/moe.py` | MoE FFN: linear router → softmax → top-k (renormalised over k), Switch-Transformer load-balancing aux loss `E·Σ f_e·P_e`, capacity factor with token dropping (earlier tokens win), experts as one batched GEMM over `[E, capacity, d]`; small byte-level GPT that uses it |
| `moe_triton/router.py` | `torch_router` (reference: softmax, topk, sum, bincount) and `triton_router`: **one fused Triton kernel** for softmax + top-k + per-expert prob sum + per-expert token count, and **one fused backward kernel** (recomputes softmax instead of storing probs) |
| `moe_triton/dispatch.py` | **Triton dispatch / combine**, forward and backward, written as row gathers (no atomics, deterministic): dispatch `buf[s] = x[tok(s)]`, combine `out[t] = Σ_j w[t,j]·h[slot(t,j)]`, plus `dw[t,j] = <dout[t], h[slot(t,j)]>` |
| `scripts/test_correctness.py` | Triton vs PyTorch, outputs and gradients |
| `scripts/bench.py` | timing (25 warm-up + 100 timed iters, CUDA events, median; a full discarded warm-up pass first) + peak memory + `torch.profiler` table |
| `scripts/train.py` | training on the first 200 MB of TinyStories (bytes, vocab 256; last 2 % held out for val) |
| `scripts/run.pbs` | PBS job: `qsub -v MODE=test|bench|train|train_full scripts/run.pbs` |

## 1. Correctness — 140/140 checks pass (`results/correctness.json`)

* **Router**, T ∈ {1, 7, 4096, 65536} × E ∈ {6, 8, 16, 64, 128} × k ∈ {1, 2, 4} × logits fp32 / bf16:
  top-k values atol 1e-6 / rtol 1e-5 (softmax is fp32 in both paths), token counts bit-exact,
  prob sums atol 1e-6·T (atomic summation order differs), logits gradient atol/rtol 1e-5 (fp32) and 1e-2 (bf16).
  With bf16 logits, equal probabilities (ties) are common and `torch.topk` orders them differently from the kernel;
  the test accepts the kernel's choice only if its values equal torch's top-k values, and takes the reference gradient w.r.t. the kernel's indices.
* **Whole MoE layer** (Triton router, PyTorch or Triton dispatch) vs the all-PyTorch layer, fp32 and bf16, incl. k=4, non-power-of-2 sizes (T=1000, d=200, E=16) and capacity factor 0.5 (heavy dropping):
  output and **every parameter gradient** match (fp32: max relative grad error ≤ 4e-7; bf16: ≤ 7.3e-3).

## 2. Router kernel alone (`results/bench_router.json`, k=2, ms, median)

Forward + backward (forward-only in the json):

| logits | E | T | PyTorch eager | torch.compile | Triton | speed-up vs eager / compile |
|---|---|---|---|---|---|---|
| bf16 | 8 | 65 536 | 0.653 | 0.831 | 0.195 | 3.4× / 4.3× |
| bf16 | 8 | 262 144 | 1.591 | 1.675 | 0.198 | 8.0× / 8.5× |
| bf16 | 64 | 65 536 | 1.077 | 1.002 | 0.201 | 5.4× / 5.0× |
| bf16 | 64 | 262 144 | 3.657 | 2.656 | 0.306 | 12.0× / 8.7× |
| bf16 | 128 | 262 144 | 6.645 | 4.439 | 0.632 | 10.5× / 7.0× |
| fp32 | 64 | 262 144 | 3.263 | 2.782 | 0.462 | 7.1× / 6.0× |

At T ≤ 16 384 everything is launch-bound (Triton ≈ 0.2 ms fwd+bwd, 1.2–3.6× faster).

## 3. Whole MoE layer, forward + backward, bf16 (`results/bench_layer.json`)

| E | d | d_ff | T | all PyTorch | + Triton router | + Triton router & dispatch | speed-up | peak extra memory |
|---|---|---|---|---|---|---|---|---|
| 8 | 1024 | 2048 | 16 384 | 17.33 ms | 17.28 ms | 13.76 ms | 1.26× | 706 → 706 MiB |
| 64 | 1024 | 512 | 16 384 | 9.67 ms | 9.61 ms | 5.90 ms | 1.64× | 453 → 390 MiB (−14 %) |
| 128 | 512 | 256 | 32 768 | 9.66 ms | 8.85 ms | 5.03 ms | 1.92× | 467 → 404 MiB (−14 %) |

**Why dispatch got a kernel** (`results/profile_layer.txt`, E=64, 10 steps): with the PyTorch layer the router kernels are ~1–2 % of GPU time, while the index-based dispatch/combine (`index_add_`, `indexing_backward_kernel`, gathers) is ~28 % (26 ms of 93 ms). The Triton router alone therefore gives ≤ 9 % at layer level; replacing dispatch/combine with gathers brings total GPU time from 93 ms to 54 ms.

## 4. Training (`results/train/*.json`, 6 layers, d=384, ctx 256, batch 64, 5000 steps, bf16 autocast, same A40 node)

MoE: every FFN is MoE, 8 experts × d_ff 768, top-2, capacity factor 1.25 — same active FFN width (2 × 768) as the dense d_ff 1536 baseline.

| run | params | val loss (nats/byte) @5000 | max expert load / uniform (last 500 steps) | tokens dropped | median step | peak memory |
|---|---|---|---|---|---|---|
| dense | 10.8 M | 0.616 | – | – | 33.8 ms | 1550 MiB |
| MoE, PyTorch router + dispatch | 32.1 M | 0.599 | 1.06 | 0 % | 64.4 ms | 2678 MiB |
| MoE, Triton router | 32.1 M | 0.600 | 1.08 | 0 % | 62.5 ms | 2677 MiB |
| **MoE, Triton router + dispatch** | 32.1 M | **0.600** | 1.06 | 0 % | **49.5 ms** (−23 %) | **2383 MiB** (−11 %) |
| MoE, no aux loss (Triton router) | 32.1 M | 0.624 | 3.65 | 25 % | 95.1 ms | 2677 MiB |

* Loss curves of the PyTorch and Triton runs agree to ≤ 0.01 at every eval point (bf16 noise), final 0.599 vs 0.600.
* Without the load-balancing loss the router collapses onto a few experts: max load 3.65× uniform, 25 % of token-assignments dropped at capacity 1.25, val loss 0.624 vs 0.599 (worse than the dense baseline).
* At init the load is already uneven (max load 2.3–3.2× uniform); the aux loss brings it to ~1.06×.

## Limitations

* One GPU type (A40), one run per config (no seeds); speed-ups are kernel/layer/step timings, not a tuned production MoE (no grouped GEMM — experts use one padded `bmm`; no expert parallelism).
* The router kernel keeps all E logits of a row in registers (fine up to E = 128 tested); block size is a fixed heuristic, not autotuned (autotune re-launches would double-count the atomics).
* The no-aux run is slower partly because the PyTorch dispatch writes all dropped tokens into one dummy slot (serialised duplicate-index updates).

## Reproduce

```
pip install -r requirements.txt
mkdir -p data && curl -L -r 0-209715199 -o data/tinystories_200MB.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
python scripts/test_correctness.py results/correctness.json
python scripts/bench.py results                          # timing + profile
python scripts/train.py moe_triton_full --data data/tinystories_200MB.txt --moe --router triton --dispatch triton --steps 5000
```
Without `--moe` the dense baseline is trained; `--router torch` / `--dispatch torch` select the PyTorch paths, `--aux 0` turns off the load-balancing loss. `scripts/run.pbs` runs the same steps as a PBS job (`MODE=test|bench|train|train_full`).
