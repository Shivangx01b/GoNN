// Correctness check for the active compute backend (intended for CUDA).
// Builds small inputs, runs ops through backend.Current(), and compares
// against an independent pure-Go reference. Covers the batched/transposed
// GEMM paths and every enum-dispatched elementwise kind. Exits non-zero on
// mismatch.
//
//	go run -tags cuda ./benchmark/verify
package main

import (
	"fmt"
	"math"
	"os"

	"gonn/backend"
	"gonn/backend/cuda"
)

// refGemm is a naive reference for row-major batched C = op(A) @ op(B).
func refGemm(a, b []float64, batch, m, k, n int, transA, transB bool) []float64 {
	out := make([]float64, batch*m*n)
	at := func(bi, i, j int) float64 {
		if transA {
			return a[bi*m*k+j*m+i]
		}
		return a[bi*m*k+i*k+j]
	}
	bt := func(bi, i, j int) float64 {
		if transB {
			return b[bi*k*n+j*k+i]
		}
		return b[bi*k*n+i*n+j]
	}
	for bi := 0; bi < batch; bi++ {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var s float64
				for p := 0; p < k; p++ {
					s += at(bi, i, p) * bt(bi, p, j)
				}
				out[bi*m*n+i*n+j] = s
			}
		}
	}
	return out
}

func maxAbsDiff(x, y []float64) float64 {
	d := 0.0
	for i := range x {
		if v := math.Abs(x[i] - y[i]); v > d {
			d = v
		}
	}
	return d
}

func check(name string, got, want []float64, tol float64) bool {
	d := maxAbsDiff(got, want)
	status := "OK"
	ok := d <= tol
	if !ok {
		status = "FAIL"
	}
	fmt.Printf("  %-24s maxAbsDiff=%.3e  tol=%.0e  [%s]\n", name, d, tol, status)
	return ok
}

func waveData(n int, freq float64) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = math.Sin(float64(i) * freq)
	}
	return d
}

func main() {
	b, err := cuda.Backend()
	if err != nil {
		fmt.Println("CUDA backend unavailable:", err)
		os.Exit(1)
	}
	backend.Use(b)
	be := backend.Current()
	fmt.Println("Backend:", be.Name())

	allOK := true

	// ---- GEMM: plain, transposed, batched, batched+transposed --------------
	m, k, n := 64, 48, 32
	A := waveData(m*k, 0.1)
	B := waveData(k*n, 0.07)
	allOK = check("gemm",
		be.Gemm(A, B, 1, m, k, n, false, false),
		refGemm(A, B, 1, m, k, n, false, false), 1e-12) && allOK
	// For the trans cases the stored dims flip; reuse buffers of the right size.
	At := waveData(k*m, 0.11) // stored (k,m) -> op(A) = (m,k)
	Bt := waveData(n*k, 0.05) // stored (n,k) -> op(B) = (k,n)
	allOK = check("gemm transA",
		be.Gemm(At, B, 1, m, k, n, true, false),
		refGemm(At, B, 1, m, k, n, true, false), 1e-12) && allOK
	allOK = check("gemm transB",
		be.Gemm(A, Bt, 1, m, k, n, false, true),
		refGemm(A, Bt, 1, m, k, n, false, true), 1e-12) && allOK
	allOK = check("gemm transAB",
		be.Gemm(At, Bt, 1, m, k, n, true, true),
		refGemm(At, Bt, 1, m, k, n, true, true), 1e-12) && allOK
	const batch = 4
	Ab := waveData(batch*m*k, 0.09)
	Bb := waveData(batch*k*n, 0.06)
	allOK = check("gemm batched",
		be.Gemm(Ab, Bb, batch, m, k, n, false, false),
		refGemm(Ab, Bb, batch, m, k, n, false, false), 1e-12) && allOK
	Abt := waveData(batch*k*m, 0.08)
	allOK = check("gemm batched transA",
		be.Gemm(Abt, Bb, batch, m, k, n, true, false),
		refGemm(Abt, Bb, batch, m, k, n, true, false), 1e-12) && allOK

	// ---- Elementwise capability --------------------------------------------
	ew, ok := be.(backend.Elementwiser)
	if !ok {
		fmt.Println("backend does not implement Elementwiser [FAIL]")
		os.Exit(1)
	}
	x := []float64{-2, -0.5, 0, 0.5, 1, 2, 3, -3}
	y := []float64{1, 2, 3, 4, 5, 6, 7, 8}

	unaryRef := map[backend.UnaryKind]struct {
		name string
		f    func(float64) float64
	}{
		backend.UnaryReLU:    {"relu", func(v float64) float64 { return math.Max(v, 0) }},
		backend.UnarySigmoid: {"sigmoid", func(v float64) float64 { return 1 / (1 + math.Exp(-v)) }},
		backend.UnaryTanh:    {"tanh", math.Tanh},
		backend.UnaryExp:     {"exp", math.Exp},
		backend.UnaryGELU: {"gelu", func(v float64) float64 {
			return 0.5 * v * (1 + math.Tanh(0.7978845608028654*(v+0.044715*v*v*v)))
		}},
		backend.UnarySiLU: {"silu", func(v float64) float64 { return v / (1 + math.Exp(-v)) }},
	}
	for kind, ref := range unaryRef {
		got := make([]float64, len(x))
		if !ew.Unary(kind, x, got) {
			fmt.Printf("  %-24s declined [FAIL]\n", ref.name)
			allOK = false
			continue
		}
		want := make([]float64, len(x))
		for i, v := range x {
			want[i] = ref.f(v)
		}
		allOK = check(ref.name, got, want, 1e-12) && allOK
	}
	// log on positive inputs only
	{
		pos := []float64{0.1, 0.5, 1, 2, 5, 10, 0.01, 3}
		got := make([]float64, len(pos))
		if !ew.Unary(backend.UnaryLog, pos, got) {
			fmt.Println("  log declined [FAIL]")
			allOK = false
		} else {
			want := make([]float64, len(pos))
			for i, v := range pos {
				want[i] = math.Log(v)
			}
			allOK = check("log", got, want, 1e-12) && allOK
		}
	}

	binaryRef := map[backend.BinaryKind]struct {
		name string
		f    func(a, b float64) float64
	}{
		backend.BinaryAdd: {"add", func(a, b float64) float64 { return a + b }},
		backend.BinarySub: {"sub", func(a, b float64) float64 { return a - b }},
		backend.BinaryMul: {"mul", func(a, b float64) float64 { return a * b }},
		backend.BinaryDiv: {"div", func(a, b float64) float64 { return a / b }},
	}
	for kind, ref := range binaryRef {
		got := make([]float64, len(x))
		if !ew.Binary(kind, x, y, got) {
			fmt.Printf("  %-24s declined [FAIL]\n", ref.name)
			allOK = false
			continue
		}
		want := make([]float64, len(x))
		for i := range x {
			want[i] = ref.f(x[i], y[i])
		}
		allOK = check(ref.name, got, want, 1e-12) && allOK
	}

	if !allOK {
		fmt.Println("VERIFY: FAILED")
		os.Exit(1)
	}
	fmt.Println("VERIFY: ALL OK on", be.Name())
}
