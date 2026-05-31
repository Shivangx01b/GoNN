//go:build cuda
// +build cuda

// Device-resident GPU benchmark: GoNN's cuBLAS path with inputs allocated once
// on the device (no per-call H2D/D2H), CUDA-event timed. This is the
// apples-to-apples comparison against PyTorch/tinygrad. Run in the CUDA image:
//
//	go run -tags cuda ./benchmark/gpuresident
//
// Emits benchmark/results/gonn_cuda_resident.json.
package main

import (
	"encoding/json"
	"fmt"
	"os"

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

func main() {
	if _, err := cuda.Backend(); err != nil {
		fmt.Println("CUDA unavailable:", err)
		os.Exit(1)
	}
	var recs []record

	for _, f32 := range []bool{true, false} {
		dt := "float64"
		if f32 {
			dt = "float32"
		}
		for _, n := range []int{256, 512, 1024, 2048} {
			iters := 50
			ms := cuda.BenchMatMulDev(n, n, n, iters, f32)
			gf := 2.0 * float64(n) * float64(n) * float64(n) / 1e9 / (ms / 1000.0)
			recs = append(recs, record{"gonn-resident", "cuda", dt, "matmul", n, ms, &gf, iters, "device-resident"})
			fmt.Printf("  matmul %-7s N=%-5d %.4f ms  %.1f GFLOP/s\n", dt, n, ms, gf)
		}
		for _, m := range []int{1_000_000, 10_000_000} {
			iters := 100
			ms := cuda.BenchAddDev(m, iters, f32)
			recs = append(recs, record{"gonn-resident", "cuda", dt, "elementwise_add", m, ms, nil, iters, "device-resident"})
			fmt.Printf("  add    %-7s M=%-9d %.4f ms\n", dt, m, ms)
		}
	}

	if err := os.MkdirAll("benchmark/results", 0o755); err != nil {
		panic(err)
	}
	out := "benchmark/results/gonn_cuda_resident.json"
	f, _ := os.Create(out)
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(recs); err != nil {
		panic(err)
	}
	fmt.Println("[gpuresident] wrote", out)
}
