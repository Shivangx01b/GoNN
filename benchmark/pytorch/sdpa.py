#!/usr/bin/env python
"""PyTorch scaled_dot_product_attention benchmark, matched shapes to GoNN's
fused flash-attention kernel.

float64: PyTorch has no flash kernel -> falls back to the math path (materializes
         the S*S score matrix). This is GoNN's target.
float32: PyTorch uses FlashAttention/mem-efficient kernels. Reported for honest
         reference (GoNN's fused kernel is fp64-only).

Shapes match benchmark/flashattn (BH, S, d), reshaped to (B=BH, H=1, S, d).
CUDA-event timed (matches GoNN). Writes benchmark/results/pytorch_sdpa.json.
"""
import json
import os

import torch
import torch.nn.functional as F

CFGS = [(64, 512, 64), (32, 1024, 64), (16, 2048, 64)]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def bench(BH, S, d, dtype, causal, iters=30, warmup=5):
    dev = "cuda"
    q = torch.randn(BH, 1, S, d, device=dev, dtype=dtype)
    k = torch.randn(BH, 1, S, d, device=dev, dtype=dtype)
    v = torch.randn(BH, 1, S, d, device=dev, dtype=dtype)
    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        starter.record()
        F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))
    times.sort()
    return times[len(times) // 2]


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    recs = []
    for dtype_name, dt in (("float64", torch.float64), ("float32", torch.float32)):
        for causal in (False, True):
            for BH, S, d in CFGS:
                ms = bench(BH, S, d, dt, causal)
                flop = 4.0 * BH * S * S * d
                if causal:
                    flop *= 0.5
                gf = flop / 1e9 / (ms / 1000.0)
                recs.append({
                    "framework": "pytorch-sdpa", "device": "cuda", "dtype": dtype_name,
                    "op": "attention", "bh": BH, "seq": S, "head_dim": d, "causal": causal,
                    "ms_per_iter": ms, "gflops": gf, "iters": 30, "transfer": "device-resident",
                })
                print(f"  {dtype_name} BH={BH} S={S} d={d} causal={causal}  {ms:.4f} ms  {gf:.1f} GFLOP/s")
    out = os.path.join(RESULTS_DIR, "pytorch_sdpa.json")
    with open(out, "w") as f:
        json.dump(recs, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
