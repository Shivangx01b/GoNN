#!/usr/bin/env python
"""tinygrad cross-framework benchmark for GoNN comparison.

Implements the ops/sizes/dtypes/methodology defined in benchmark/BENCH_SPEC.md
and emits one JSON file per device:

    benchmark/results/tinygrad_<device>.json

Each record:
    framework, device, dtype, op, size, ms_per_iter, gflops, iters, transfer

Methodology (per spec):
  - Warmup: 5 iters (not timed).
  - Timed: median of `iters` runs (matmul: 20; elementwise/relu: 50).
  - tinygrad is LAZY: every timed op calls .realize() and, on a GPU backend,
    Device[dev].synchronize() *inside* the timing boundary so the kernel is
    actually compiled+launched+awaited. Inputs are realized once up front so we
    time compute, not graph construction.
  - GPU tensors are device-resident (transfer="device-resident"); the op only is
    timed (idiomatic usage). CPU runs are transfer="host".
  - GFLOP/s reported for matmul only (2*N^3 / 1e9 / (ms_per_iter/1000)); null
    otherwise.

Device mapping (this host / tinygrad 0.13, Windows):
  tinygrad exposes backends as device strings (CL, CUDA, CPU, CLANG, LLVM,
  PYTHON, ...). This script probes them at runtime and only benchmarks backends
  that actually compile+run. Discovered mapping is reported by the run.

  - The "CL" (OpenCL) backend runs on the NVIDIA GeForce RTX 3060 (an OpenCL GPU
    device, CL_DEVICE_TYPE=GPU). It is the GPU/CUDA-class backend here, so it is
    labeled device="cuda" with transfer="device-resident" per the spec.
  - The native CPU backends (CPU/CLANG/LLVM) and the CUDA backend fail to import
    on this Windows host (tinygrad 0.13's HCQ/ctypes library-path loader raises
    PermissionError on C:\\Windows\\system32\\...\\WindowsApps and the libc/cuda
    header autogen path is POSIX-oriented). Any backend that does not pass a live
    smoke check is SKIPPED (not faked) and noted.
"""

import argparse
import json
import os
import statistics
import time

from tinygrad import Device, Tensor, dtypes

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
    "float32": dtypes.float32,
    "float64": dtypes.float64,
}

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)

# Backend device-string -> (json device label, transfer label).
# We treat OpenCL-on-GPU (CL) as the cuda/device-resident GPU backend, and the
# native CPU backends as the host backend. Probed at runtime; only the ones that
# pass a live smoke test are used.
GPU_BACKENDS = ["CL", "CUDA", "GPU", "NV"]   # GPU/device-resident candidates
CPU_BACKENDS = ["CPU", "CLANG", "LLVM"]      # host candidates


def _sync(dev):
    """Synchronize a tinygrad device if it supports it (GPU backends do)."""
    try:
        Device[dev].synchronize()
    except Exception:
        pass


def _smoke(dev, dt):
    """Return True if `dev` can compile+run a tiny matmul in dtype `dt`."""
    try:
        a = Tensor.rand(8, 8, device=dev).cast(dt).realize()
        b = Tensor.rand(8, 8, device=dev).cast(dt).realize()
        r = (a @ b).realize()
        _sync(dev)
        v = r.numpy()
        return v.shape == (8, 8)
    except Exception:
        return False


def _device_name(dev):
    """Best-effort human name for the underlying compute device."""
    if dev == "CL":
        try:
            import ctypes
            import tinygrad.runtime.autogen.opencl as cl
            d = Device[dev]
            buf = ctypes.create_string_buffer(256)
            cl.clGetDeviceInfo(d.device_id, 0x102B, 256, buf, None)  # CL_DEVICE_NAME
            return buf.value.decode(errors="replace")
        except Exception:
            return dev
    return dev


def _time_median(make_op, dev, iters, warmup, is_gpu):
    """Run a realize()'d op warmup+iters times, syncing inside the timed region.

    `make_op()` must return an un-realized Tensor (the op result). We realize it
    (and sync on GPU) inside each timed iteration so the kernel actually runs.
    Returns the median per-iter time in milliseconds.
    """
    for _ in range(warmup):
        make_op().realize()
    if is_gpu:
        _sync(dev)

    times_ms = []
    for _ in range(iters):
        if is_gpu:
            _sync(dev)
            t0 = time.perf_counter()
            make_op().realize()
            _sync(dev)
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            make_op().realize()
            t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return statistics.median(times_ms)


def bench_matmul(dev, dt, N, iters, warmup, is_gpu):
    a = Tensor.rand(N, N, device=dev).cast(dt).realize()
    b = Tensor.rand(N, N, device=dev).cast(dt).realize()
    _sync(dev)
    ms = _time_median(lambda: a @ b, dev, iters, warmup, is_gpu)
    gflops = 2.0 * (N ** 3) / 1e9 / (ms / 1000.0)
    return ms, gflops


def bench_elementwise_add(dev, dt, M, iters, warmup, is_gpu):
    a = Tensor.rand(M, device=dev).cast(dt).realize()
    b = Tensor.rand(M, device=dev).cast(dt).realize()
    _sync(dev)
    ms = _time_median(lambda: a + b, dev, iters, warmup, is_gpu)
    return ms, None


def bench_relu(dev, dt, M, iters, warmup, is_gpu):
    a = Tensor.rand(M, device=dev).cast(dt).realize()
    _sync(dev)
    ms = _time_median(lambda: a.relu(), dev, iters, warmup, is_gpu)
    return ms, None


def run_device(backend, label, transfer, is_gpu, supported_dtypes, quick):
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

    for dtype_name in supported_dtypes:
        dt = DTYPES[dtype_name]

        # matmul
        for N in matmul_sizes:
            ms, gflops = bench_matmul(backend, dt, N, matmul_iters, warmup, is_gpu)
            records.append({
                "framework": "tinygrad",
                "device": label,
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
                backend, dt, M, elementwise_iters, warmup, is_gpu
            )
            records.append({
                "framework": "tinygrad",
                "device": label,
                "dtype": dtype_name,
                "op": "elementwise_add",
                "size": M,
                "ms_per_iter": ms,
                "gflops": None,
                "iters": elementwise_iters,
                "transfer": transfer,
            })

            ms, _ = bench_relu(
                backend, dt, M, elementwise_iters, warmup, is_gpu
            )
            records.append({
                "framework": "tinygrad",
                "device": label,
                "dtype": dtype_name,
                "op": "relu",
                "size": M,
                "ms_per_iter": ms,
                "gflops": None,
                "iters": elementwise_iters,
                "transfer": transfer,
            })

    return records


def discover_backends():
    """Probe tinygrad backends and return a list of usable ones.

    Each entry: dict(backend, label, transfer, is_gpu, dtypes). Only the first
    working GPU backend and the first working CPU backend are kept so we emit at
    most one tinygrad_cuda.json and one tinygrad_cpu.json.
    """
    chosen = []
    print(f"tinygrad Device.DEFAULT = {Device.DEFAULT}")

    # GPU / device-resident backend -> labeled "cuda"
    for b in GPU_BACKENDS:
        if _smoke(b, dtypes.float32):
            f64 = _smoke(b, dtypes.float64)
            dts = ["float32"] + (["float64"] if f64 else [])
            name = _device_name(b)
            print(f"  GPU backend '{b}' OK on '{name}' "
                  f"-> device='cuda' transfer='device-resident' "
                  f"float64={'yes' if f64 else 'NO (skipped)'}")
            chosen.append({
                "backend": b, "label": "cuda",
                "transfer": "device-resident", "is_gpu": True, "dtypes": dts,
            })
            break
        else:
            print(f"  GPU backend '{b}' unavailable (skipped)")

    # CPU / host backend -> labeled "cpu"
    for b in CPU_BACKENDS:
        if _smoke(b, dtypes.float32):
            f64 = _smoke(b, dtypes.float64)
            dts = ["float32"] + (["float64"] if f64 else [])
            print(f"  CPU backend '{b}' OK -> device='cpu' transfer='host' "
                  f"float64={'yes' if f64 else 'NO (skipped)'}")
            chosen.append({
                "backend": b, "label": "cpu",
                "transfer": "host", "is_gpu": False, "dtypes": dts,
            })
            break
        else:
            print(f"  CPU backend '{b}' unavailable (skipped)")

    if not chosen:
        print("  NO usable tinygrad backend found.")
    return chosen


def main():
    parser = argparse.ArgumentParser(description="tinygrad GoNN-comparison benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run tiny sizes / few iters for smoke testing.",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    backends = discover_backends()
    if not backends:
        raise SystemExit("no usable tinygrad backend; nothing benchmarked")

    for cfg in backends:
        records = run_device(
            cfg["backend"], cfg["label"], cfg["transfer"],
            cfg["is_gpu"], cfg["dtypes"], args.quick,
        )
        out_path = os.path.join(RESULTS_DIR, f"tinygrad_{cfg['label']}.json")
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"wrote {out_path} ({len(records)} records, "
              f"backend={cfg['backend']}, dtypes={cfg['dtypes']})")
        if args.quick:
            print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
