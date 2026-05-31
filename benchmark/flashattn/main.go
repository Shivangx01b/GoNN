//go:build cuda
// +build cuda

// Fused flash-attention (float64) correctness check + benchmark vs PyTorch SDPA.
// PyTorch's flash kernels are fp16/bf16/fp32 only; for fp64 SDPA falls back to
// the "math" path (materialized S*S scores, several kernel launches), which a
// fused online-softmax kernel can beat.
//
//	go run -tags cuda ./benchmark/flashattn
//
// Emits benchmark/results/gonn_flash.json.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"

	"gonn/backend/cuda"
)

type record struct {
	Framework string   `json:"framework"`
	Device    string   `json:"device"`
	Dtype     string   `json:"dtype"`
	Op        string   `json:"op"`
	BH        int      `json:"bh"`
	Seq       int      `json:"seq"`
	HeadDim   int      `json:"head_dim"`
	Causal    bool     `json:"causal"`
	MsPerIter float64  `json:"ms_per_iter"`
	GFLOPs    *float64 `json:"gflops"`
	Iters     int      `json:"iters"`
	Transfer  string   `json:"transfer"`
}

func refAttn(Q, K, V []float64, BH, S, d int, scale float64, causal bool) []float64 {
	O := make([]float64, BH*S*d)
	for bh := 0; bh < BH; bh++ {
		for i := 0; i < S; i++ {
			jmax := S
			if causal {
				jmax = i + 1
			}
			scores := make([]float64, jmax)
			mx := math.Inf(-1)
			for j := 0; j < jmax; j++ {
				s := 0.0
				for t := 0; t < d; t++ {
					s += Q[(bh*S+i)*d+t] * K[(bh*S+j)*d+t]
				}
				s *= scale
				scores[j] = s
				if s > mx {
					mx = s
				}
			}
			sum := 0.0
			for j := range scores {
				scores[j] = math.Exp(scores[j] - mx)
				sum += scores[j]
			}
			for t := 0; t < d; t++ {
				acc := 0.0
				for j := 0; j < jmax; j++ {
					acc += scores[j] * V[(bh*S+j)*d+t]
				}
				O[(bh*S+i)*d+t] = acc / sum
			}
		}
	}
	return O
}

func randv(n int) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = rand.NormFloat64()
	}
	return d
}

func maxAbsDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		if v := math.Abs(a[i] - b[i]); v > m {
			m = v
		}
	}
	return m
}

func main() {
	if _, err := cuda.Backend(); err != nil {
		fmt.Println("CUDA unavailable:", err)
		os.Exit(1)
	}
	rand.Seed(7)

	// ---- correctness vs CPU reference ----
	fmt.Println("=== flash-attn f64 correctness (GPU fused vs CPU ref) ===")
	ok := true
	for _, causal := range []bool{false, true} {
		BH, S, d := 3, 12, 8
		Q, K, V := randv(BH*S*d), randv(BH*S*d), randv(BH*S*d)
		O := make([]float64, BH*S*d)
		scale := 1.0 / math.Sqrt(float64(d))
		cuda.FlashAttnF64(Q, K, V, O, BH, S, d, scale, causal)
		ref := refAttn(Q, K, V, BH, S, d, scale, causal)
		diff := maxAbsDiff(O, ref)
		status := "OK"
		if diff > 1e-9 {
			status = "FAIL"
			ok = false
		}
		fmt.Printf("  causal=%-5v  maxAbsDiff=%.3e  [%s]\n", causal, diff, status)
	}
	if !ok {
		fmt.Println("FLASH-ATTN CORRECTNESS FAILED")
		os.Exit(1)
	}

	// ---- benchmark ----
	fmt.Println("=== flash-attn f64 benchmark (device-resident, CUDA-event timed) ===")
	type cfg struct{ BH, S, d int }
	cfgs := []cfg{
		{64, 512, 64},
		{32, 1024, 64},
		{16, 2048, 64},
	}
	var recs []record
	for _, causal := range []bool{false, true} {
		for _, c := range cfgs {
			iters := 30
			ms := cuda.BenchFlashAttnF64(c.BH, c.S, c.d, iters, causal)
			// FLOPs: QK^T (2*BH*S*S*d) + P*V (2*BH*S*S*d); causal ~halves it.
			flop := 4.0 * float64(c.BH) * float64(c.S) * float64(c.S) * float64(c.d)
			if causal {
				flop *= 0.5
			}
			gf := flop / 1e9 / (ms / 1000.0)
			recs = append(recs, record{
				Framework: "gonn-flash", Device: "cuda", Dtype: "float64", Op: "attention",
				BH: c.BH, Seq: c.S, HeadDim: c.d, Causal: causal,
				MsPerIter: ms, GFLOPs: &gf, Iters: iters, Transfer: "device-resident",
			})
			fmt.Printf("  BH=%-3d S=%-5d d=%d causal=%-5v  %.4f ms  %.1f GFLOP/s\n",
				c.BH, c.S, c.d, causal, ms, gf)
		}
	}

	if err := os.MkdirAll("benchmark/results", 0o755); err != nil {
		panic(err)
	}
	out := "benchmark/results/gonn_flash.json"
	f, _ := os.Create(out)
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(recs); err != nil {
		panic(err)
	}
	fmt.Println("[flashattn] wrote", out)
}
