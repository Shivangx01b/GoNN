#!/usr/bin/env python
"""PyTorch cross-framework benchmark for GoNN comparison.

Implements the ops/sizes/dtypes/methodology defined in benchmark/BENCH_SPEC.md
and emits one JSON file per device:

    benchmark/results/pytorch_<device>.json

Each record:
    framework, device, dtype, op, size, ms_per_iter, gflops, iters, transfer

Methodology (per spec):
  - Warmup: 5 iters (not timed).
  - Timed: median of `iters` runs (matmul: 20; elementwise/relu: 50).
  - CUDA: torch.cuda.synchronize() inside the timing boundaries.
  - CUDA tensors are device-resident (transfer="device-resident"); the op only
    is timed (idiomatic usage). CPU runs are transfer="host".
  - GFLOP/s reported for matmul only (2*N^3 / 1e9 / (ms_per_iter/1000)); null
    otherwise.
"""

import argparse
import json
import os
import statistics
import time

import torch

# ---------------------------------------------------------------------------
# Sizes / iter counts (full run)
# ---------------------------------------------------------------------------
MATMUL_SIZES = [256, 512, 1024, 2048]
ELEMENTWISE_SIZES = [1_000_000, 10_000_000]
MATMUL_ITERS = 20
ELEMENTWISE_ITERS = 50
WARMUP = 5

# Quick / smoke-test config: tiny sizes, few iters.
QUICK_MATMUL_SIZES = [64, 128]
QUICK_ELEMENTWISE_SIZES = [10_000, 100_000]
QUICK_MATMUL_ITERS = 3
QUICK_ELEMENTWISE_ITERS = 3
QUICK_WARMUP = 2

DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def _time_median(fn, device, iters, warmup):
    """Run `fn` warmup+iters times, syncing on CUDA inside the timed region.

    Returns the median per-iter time in milliseconds.
    """
    is_cuda = device == "cuda"

    for _ in range(warmup):
        fn()

    times_ms = []
    if is_cuda:
        # CUDA-event timing: measures pure kernel time without the Python
        # perf_counter loop overhead, matching GoNN's cudaEvent-timed
        # device-resident benchmark for an apples-to-apples comparison.
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            starter.record()
            fn()
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    return statistics.median(times_ms)


def bench_matmul(device, torch_dtype, N, iters, warmup):
    a = torch.randn((N, N), device=device, dtype=torch_dtype)
    b = torch.randn((N, N), device=device, dtype=torch_dtype)
    ms = _time_median(lambda: torch.matmul(a, b), device, iters, warmup)
    gflops = 2.0 * (N ** 3) / 1e9 / (ms / 1000.0)
    return ms, gflops


def bench_elementwise_add(device, torch_dtype, M, iters, warmup):
    a = torch.randn(M, device=device, dtype=torch_dtype)
    b = torch.randn(M, device=device, dtype=torch_dtype)
    ms = _time_median(lambda: torch.add(a, b), device, iters, warmup)
    return ms, None


def bench_relu(device, torch_dtype, M, iters, warmup):
    a = torch.randn(M, device=device, dtype=torch_dtype)
    ms = _time_median(lambda: torch.relu(a), device, iters, warmup)
    return ms, None


def run_device(device, quick):
    transfer = "device-resident" if device == "cuda" else "host"

    if quick:
        matmul_sizes = QUICK_MATMUL_SIZES
        elementwise_sizes = QUICK_ELEMENTWISE_SIZES
        matmul_iters = QUICK_MATMUL_ITERS
        elementwise_iters = QUICK_ELEMENTWISE_ITERS
        warmup = QUICK_WARMUP
    else:
        matmul_sizes = MATMUL_SIZES
        elementwise_sizes = ELEMENTWISE_SIZES
        matmul_iters = MATMUL_ITERS
        elementwise_iters = ELEMENTWISE_ITERS
        warmup = WARMUP

    records = []

    for dtype_name, torch_dtype in DTYPES.items():
        # matmul
        for N in matmul_sizes:
            ms, gflops = bench_matmul(device, torch_dtype, N, matmul_iters, warmup)
            records.append({
                "framework": "pytorch",
                "device": device,
                "dtype": dtype_name,
                "op": "matmul",
                "size": N,
                "ms_per_iter": ms,
                "gflops": gflops,
                "iters": matmul_iters,
                "transfer": transfer,
            })

        # elementwise_add + relu
        for M in elementwise_sizes:
            ms, _ = bench_elementwise_add(
                device, torch_dtype, M, elementwise_iters, warmup
            )
            records.append({
                "framework": "pytorch",
                "device": device,
                "dtype": dtype_name,
                "op": "elementwise_add",
                "size": M,
                "ms_per_iter": ms,
                "gflops": None,
                "iters": elementwise_iters,
                "transfer": transfer,
            })

            ms, _ = bench_relu(device, torch_dtype, M, elementwise_iters, warmup)
            records.append({
                "framework": "pytorch",
                "device": device,
                "dtype": dtype_name,
                "op": "relu",
                "size": M,
                "ms_per_iter": ms,
                "gflops": None,
                "iters": elementwise_iters,
                "transfer": transfer,
            })

    return records


def main():
    parser = argparse.ArgumentParser(description="PyTorch GoNN-comparison benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run tiny sizes / few iters for smoke testing.",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        records = run_device(device, args.quick)
        out_path = os.path.join(RESULTS_DIR, f"pytorch_{device}.json")
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"wrote {out_path} ({len(records)} records)")
        if args.quick:
            print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
