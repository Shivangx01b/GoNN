// GoNN benchmark harness. Measures the active compute backend (CPU by default,
// CUDA when built with -tags cuda) on the ops defined in benchmark/BENCH_SPEC.md.
//
//	CPU:  go run ./benchmark/gonn
//	CUDA: go run -tags cuda ./benchmark/gonn   (inside the CUDA docker image)
//
// Emits benchmark/results/gonn_<device>.json. When the backend implements
// backend.Elementwiser it also prints a per-size CPU-vs-dispatch table used
// to tune tensor.DispatchPolicy.UnaryMinElems.
package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"gonn/backend"
	"gonn/backend/cuda"
)

type record struct {
	Framework string   `json:"framework"`
	Device    string   `json:"device"`
	Dtype     string   `json:"dtype"`
	Op        string   `json:"op"`
	Size      int      `json:"size"`
	MsPerIter float64  `json:"ms_per_iter"`
	GFLOPs    *float64 `json:"gflops"`
	Iters     int      `json:"iters"`
	Transfer  string   `json:"transfer"`
}

func medianMs(durs []time.Duration) float64 {
	sort.Slice(durs, func(i, j int) bool { return durs[i] < durs[j] })
	d := durs[len(durs)/2]
	return float64(d.Nanoseconds()) / 1e6
}

func randData(n int) []float64 {
	d := make([]float64, n)
	var s uint64 = 0x9e3779b97f4a7c15
	for i := range d {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		d[i] = float64(s%1000)/500.0 - 1.0
	}
	return d
}

// addHost / reluHost are the pure-Go reference paths, used when the backend
// has no Elementwiser capability (CPU) — matching what tensor ops run.
func addHost(a, b, out []float64) {
	for i := range a {
		out[i] = a[i] + b[i]
	}
}

func reluHost(a, out []float64) {
	for i, v := range a {
		out[i] = math.Max(v, 0)
	}
}

func main() {
	device := "cpu"
	transfer := "host"
	if b, err := cuda.Backend(); err == nil {
		backend.Use(b)
		device = "cuda"
		transfer = "per-call-h2d-d2h"
		fmt.Println("[gonn-bench] CUDA backend active")
	} else {
		fmt.Println("[gonn-bench] CPU backend active")
	}
	be := backend.Current()
	ew, hasEW := be.(backend.Elementwiser)

	var recs []record
	const warmup = 5

	// ---- matmul ----
	for _, n := range []int{256, 512, 1024, 2048} {
		A := randData(n * n)
		B := randData(n * n)
		iters := 20
		for i := 0; i < warmup; i++ {
			_ = be.Gemm(A, B, 1, n, n, n, false, false)
		}
		be.Synchronize()
		durs := make([]time.Duration, iters)
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			_ = be.Gemm(A, B, 1, n, n, n, false, false)
			be.Synchronize()
			durs[i] = time.Since(t0)
		}
		ms := medianMs(durs)
		gf := 2.0 * float64(n) * float64(n) * float64(n) / 1e9 / (ms / 1000.0)
		recs = append(recs, record{"gonn", device, "float64", "matmul", n, ms, &gf, iters, transfer})
		fmt.Printf("  matmul N=%-5d  %.3f ms  %.1f GFLOP/s\n", n, ms, gf)
	}

	// ---- elementwise add & relu (dispatched when available, else host) ----
	for _, m := range []int{1_000_000, 10_000_000} {
		A := randData(m)
		B := randData(m)
		out := make([]float64, m)
		iters := 50

		addOnce := func() {
			if !hasEW || !ew.Binary(backend.BinaryAdd, A, B, out) {
				addHost(A, B, out)
			}
		}
		reluOnce := func() {
			if !hasEW || !ew.Unary(backend.UnaryReLU, A, out) {
				reluHost(A, out)
			}
		}

		for i := 0; i < warmup; i++ {
			addOnce()
		}
		be.Synchronize()
		durs := make([]time.Duration, iters)
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			addOnce()
			be.Synchronize()
			durs[i] = time.Since(t0)
		}
		recs = append(recs, record{"gonn", device, "float64", "elementwise_add", m, medianMs(durs), nil, iters, transfer})
		fmt.Printf("  add    M=%-9d %.3f ms\n", m, medianMs(durs))

		for i := 0; i < warmup; i++ {
			reluOnce()
		}
		be.Synchronize()
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			reluOnce()
			be.Synchronize()
			durs[i] = time.Since(t0)
		}
		recs = append(recs, record{"gonn", device, "float64", "relu", m, medianMs(durs), nil, iters, transfer})
		fmt.Printf("  relu   M=%-9d %.3f ms\n", m, medianMs(durs))
	}

	// ---- unary dispatch break-even table (GPU backends only) ----
	// Times the pure-Go loop vs the dispatched kernel (incl. H2D/D2H) for a
	// transcendental op across sizes, to tune DispatchPolicy.UnaryMinElems.
	if hasEW {
		fmt.Println("[gonn-bench] unary dispatch break-even (tanh): size, host ms, dispatch ms")
		for _, m := range []int{1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22} {
			A := randData(m)
			out := make([]float64, m)
			const it = 20

			hostDurs := make([]time.Duration, it)
			for i := 0; i < it; i++ {
				t0 := time.Now()
				for j, v := range A {
					out[j] = math.Tanh(v)
				}
				hostDurs[i] = time.Since(t0)
			}
			devDurs := make([]time.Duration, it)
			for i := 0; i < it; i++ {
				t0 := time.Now()
				ew.Unary(backend.UnaryTanh, A, out)
				be.Synchronize()
				devDurs[i] = time.Since(t0)
			}
			fmt.Printf("  tanh M=%-9d host %.3f ms   dispatch %.3f ms\n",
				m, medianMs(hostDurs), medianMs(devDurs))
		}
	}

	if err := os.MkdirAll("benchmark/results", 0o755); err != nil {
		panic(err)
	}
	out := "benchmark/results/gonn_" + device + ".json"
	writeJSON(out, recs)
	fmt.Println("[gonn-bench] wrote", out)
}
