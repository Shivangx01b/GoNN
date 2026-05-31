#!/usr/bin/env python3
"""TensorFlow CPU benchmark for the GoNN cross-framework comparison.

Implements the ops/sizes/dtypes/JSON format from benchmark/BENCH_SPEC.md.

Notes on this host:
  - TensorFlow does NOT see a GPU here (tf.config.list_physical_devices('GPU')
    returns []), so only the CPU device is benchmarked. We force CPU placement
    with tf.device('/CPU:0').
  - Eager execution computes ops eagerly; we still force materialization of the
    result (read a scalar / call .numpy()) so we measure real compute, not lazy
    graph construction.
  - transfer="host" for all CPU runs (per spec).
"""

import os

# Suppress TF's verbose C++ logging before importing tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import statistics
import time

import tensorflow as tf


FRAMEWORK = "tensorflow"
DEVICE = "cpu"
TRANSFER = "host"

DTYPES = {
    "float32": tf.float32,
    "float64": tf.float64,
}

# Default (full) configuration per BENCH_SPEC.md.
MATMUL_SIZES = [256, 512, 1024, 2048]
ELEMENTWISE_SIZES = [1_000_000, 10_000_000]
WARMUP = 5
MATMUL_ITERS = 20
ELEMENTWISE_ITERS = 50

# Quick / smoke configuration.
QUICK_MATMUL_SIZES = [64, 128]
QUICK_ELEMENTWISE_SIZES = [10_000, 100_000]
QUICK_WARMUP = 2
QUICK_MATMUL_ITERS = 3
QUICK_ELEMENTWISE_ITERS = 3


def _materialize(t):
    """Force the eager tensor to be fully realized.

    Reading a single element via .numpy() blocks until the op has executed,
    so we measure real compute rather than lazy/async dispatch.
    """
    # Flatten and grab element 0; cheap, but requires the full result to exist.
    return float(tf.reshape(t, [-1])[0].numpy())


def _time_op(fn, warmup, iters):
    """Run `fn` warmup times (untimed) then `iters` timed times.

    `fn` must execute the op AND materialize its result. Returns median ms.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return statistics.median(samples)


def bench_matmul(sizes, dtype_name, tf_dtype, warmup, iters):
    records = []
    for n in sizes:
        with tf.device("/CPU:0"):
            # Build host input tensors once; reuse across iters.
            a = tf.constant(
                tf.random.uniform([n, n], dtype=tf.float32), dtype=tf.float32
            )
            a = tf.cast(a, tf_dtype)
            b = tf.cast(
                tf.constant(tf.random.uniform([n, n], dtype=tf.float32)), tf_dtype
            )

            def run():
                c = tf.linalg.matmul(a, b)
                return _materialize(c)

            ms = _time_op(run, warmup, iters)

        gflops = 2.0 * (n ** 3) / 1e9 / (ms / 1000.0)
        records.append(
            {
                "framework": FRAMEWORK,
                "device": DEVICE,
                "dtype": dtype_name,
                "op": "matmul",
                "size": n,
                "ms_per_iter": ms,
                "gflops": gflops,
                "iters": iters,
                "transfer": TRANSFER,
            }
        )
    return records


def bench_elementwise(sizes, dtype_name, tf_dtype, warmup, iters):
    records = []
    for m in sizes:
        with tf.device("/CPU:0"):
            a = tf.cast(
                tf.constant(tf.random.uniform([m], dtype=tf.float32)), tf_dtype
            )
            b = tf.cast(
                tf.constant(tf.random.uniform([m], dtype=tf.float32)), tf_dtype
            )
            # relu needs both positive and negative inputs to be meaningful.
            a_relu = tf.cast(
                tf.constant(
                    tf.random.uniform([m], minval=-1.0, maxval=1.0, dtype=tf.float32)
                ),
                tf_dtype,
            )

            def run_add():
                c = tf.add(a, b)
                return _materialize(c)

            def run_relu():
                c = tf.nn.relu(a_relu)
                return _materialize(c)

            ms_add = _time_op(run_add, warmup, iters)
            ms_relu = _time_op(run_relu, warmup, iters)

        records.append(
            {
                "framework": FRAMEWORK,
                "device": DEVICE,
                "dtype": dtype_name,
                "op": "elementwise_add",
                "size": m,
                "ms_per_iter": ms_add,
                "gflops": None,
                "iters": iters,
                "transfer": TRANSFER,
            }
        )
        records.append(
            {
                "framework": FRAMEWORK,
                "device": DEVICE,
                "dtype": dtype_name,
                "op": "relu",
                "size": m,
                "ms_per_iter": ms_relu,
                "gflops": None,
                "iters": iters,
                "transfer": TRANSFER,
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test with tiny sizes/iters.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: benchmark/results/tensorflow_cpu.json).",
    )
    args = parser.parse_args()

    if args.quick:
        matmul_sizes = QUICK_MATMUL_SIZES
        elementwise_sizes = QUICK_ELEMENTWISE_SIZES
        warmup = QUICK_WARMUP
        matmul_iters = QUICK_MATMUL_ITERS
        elementwise_iters = QUICK_ELEMENTWISE_ITERS
    else:
        matmul_sizes = MATMUL_SIZES
        elementwise_sizes = ELEMENTWISE_SIZES
        warmup = WARMUP
        matmul_iters = MATMUL_ITERS
        elementwise_iters = ELEMENTWISE_ITERS

    if args.out:
        out_path = args.out
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.normpath(
            os.path.join(here, "..", "results", "tensorflow_cpu.json")
        )

    # Confirm CPU-only on this host (informational on stderr-equivalent stdout).
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    print(f"tensorflow {tf.__version__} | CPUs={len(cpus)} GPUs={len(gpus)} "
          f"| threads(inter={tf.config.threading.get_inter_op_parallelism_threads()},"
          f"intra={tf.config.threading.get_intra_op_parallelism_threads()})")

    records = []
    for dtype_name, tf_dtype in DTYPES.items():
        records += bench_matmul(
            matmul_sizes, dtype_name, tf_dtype, warmup, matmul_iters
        )
        records += bench_elementwise(
            elementwise_sizes, dtype_name, tf_dtype, warmup, elementwise_iters
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
        f.write("\n")

    print(f"wrote {len(records)} records -> {out_path}")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
