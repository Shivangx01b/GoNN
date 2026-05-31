# GoNN cross-framework benchmark spec

Every framework benchmark MUST implement the same ops/sizes and emit the same
JSON so results are directly comparable. Be honest about methodology — this is
not a "GoNN wins" exercise; it is an apples-to-apples-where-possible measurement
with caveats stated.

## Ops and sizes

1. `matmul` — square `C = A @ B`, with `N x N @ N x N`, for `N in {256, 512, 1024, 2048}`.
   - FLOPs = `2 * N^3`. Report GFLOP/s = `2*N^3 / 1e9 / (ms_per_iter/1000)`.
2. `elementwise_add` — `C = A + B`, length `M in {1_000_000, 10_000_000}`.
3. `relu` — `C = relu(A)`, length `M in {1_000_000, 10_000_000}`.

## Dtypes / devices

- Run both `float32` and `float64` where supported.
  - GoNN is **float64 only** — emit only float64 records.
  - PyTorch / TF / tinygrad: run float32 AND float64.
- Devices: `cpu` always; `cuda` when available.

## Methodology (state it in the report, do not hide it)

- Warmup: 5 iters (not timed).
- Timed: take the **median** of `iters` runs (matmul: 20; elementwise: 50).
- GPU: call the framework's sync (`torch.cuda.synchronize()`, `Device.synchronize()`,
  GoNN `backend.Current().Synchronize()`) **inside** the timing loop boundaries so
  kernel launches are actually awaited.
- **What is timed differs by design and MUST be labeled:**
  - PyTorch / TF / tinygrad on CUDA: tensors already resident on device; time the
    op only (idiomatic usage). Label `transfer="device-resident"`.
  - GoNN CUDA backend: copies host->device, runs, device->host **every call**
    (no device buffer caching yet). Label `transfer="per-call-h2d-d2h"`.
  - All CPU runs: label `transfer="host"`.
- Single CPU thread is NOT enforced (report uses default threading); note the
  CPU core count in the report.

## Output format

Write `benchmark/results/<framework>_<device>.json` — a JSON array of records:

```json
[
  {"framework":"gonn","device":"cuda","dtype":"float64","op":"matmul",
   "size":1024,"ms_per_iter":1.83,"gflops":1173.0,"iters":20,
   "transfer":"per-call-h2d-d2h"}
]
```

Required keys: framework, device, dtype, op, size, ms_per_iter, iters, transfer.
`gflops` required for matmul (null otherwise). Numbers are medians.

frameworks: `gonn`, `pytorch`, `tensorflow`, `tinygrad`.
