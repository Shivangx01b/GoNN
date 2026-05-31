// GoNN benchmark harness. Measures the active compute backend (CPU by default,
// CUDA when built with -tags cuda) on the ops defined in benchmark/BENCH_SPEC.md.
//
//	CPU:  go run ./benchmark/gonn
//	CUDA: go run -tags cuda ./benchmark/gonn   (inside the CUDA docker image)
//
// Emits benchmark/results/gonn_<device>.json.
package main

import (
	"fmt"
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

	var recs []record
	const warmup = 5

	// ---- matmul ----
	for _, n := range []int{256, 512, 1024, 2048} {
		A := randData(n * n)
		B := randData(n * n)
		iters := 20
		for i := 0; i < warmup; i++ {
			_ = be.MatMul(A, B, n, n, n)
		}
		be.Synchronize()
		durs := make([]time.Duration, iters)
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			_ = be.MatMul(A, B, n, n, n)
			be.Synchronize()
			durs[i] = time.Since(t0)
		}
		ms := medianMs(durs)
		gf := 2.0 * float64(n) * float64(n) * float64(n) / 1e9 / (ms / 1000.0)
		recs = append(recs, record{"gonn", device, "float64", "matmul", n, ms, &gf, iters, transfer})
		fmt.Printf("  matmul N=%-5d  %.3f ms  %.1f GFLOP/s\n", n, ms, gf)
	}

	// ---- elementwise add & relu ----
	for _, m := range []int{1_000_000, 10_000_000} {
		A := randData(m)
		B := randData(m)
		iters := 50

		for i := 0; i < warmup; i++ {
			_ = be.AddElem(A, B)
		}
		be.Synchronize()
		durs := make([]time.Duration, iters)
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			_ = be.AddElem(A, B)
			be.Synchronize()
			durs[i] = time.Since(t0)
		}
		recs = append(recs, record{"gonn", device, "float64", "elementwise_add", m, medianMs(durs), nil, iters, transfer})
		fmt.Printf("  add    M=%-9d %.3f ms\n", m, medianMs(durs))

		for i := 0; i < warmup; i++ {
			_ = be.ReLU(A)
		}
		be.Synchronize()
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			_ = be.ReLU(A)
			be.Synchronize()
			durs[i] = time.Since(t0)
		}
		recs = append(recs, record{"gonn", device, "float64", "relu", m, medianMs(durs), nil, iters, transfer})
		fmt.Printf("  relu   M=%-9d %.3f ms\n", m, medianMs(durs))
	}

	if err := os.MkdirAll("benchmark/results", 0o755); err != nil {
		panic(err)
	}
	out := "benchmark/results/gonn_" + device + ".json"
	writeJSON(out, recs)
	fmt.Println("[gonn-bench] wrote", out)
}
