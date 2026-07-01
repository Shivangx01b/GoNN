package nn

import (
	"math"
	"testing"

	"gonn/tensor"
)

// lrnRef is a brute-force LocalResponseNorm reference: window
// [c - n/2, c + (n-1)/2] clamped to valid channels, divisor always n.
func lrnRef(x *tensor.Tensor, size int, alpha, beta, k float64) []float64 {
	N, C := x.Shape[0], x.Shape[1]
	rest := len(x.Data) / (N * C)
	out := make([]float64, len(x.Data))
	lo, hi := size/2, (size-1)/2
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for s := 0; s < rest; s++ {
				sum := 0.0
				for cp := c - lo; cp <= c+hi; cp++ {
					if cp >= 0 && cp < C {
						v := x.Data[(n*C+cp)*rest+s]
						sum += v * v
					}
				}
				idx := (n*C+c)*rest + s
				out[idx] = x.Data[idx] / math.Pow(k+alpha/float64(size)*sum, beta)
			}
		}
	}
	return out
}

func TestLocalResponseNormSingleChannel(t *testing.T) {
	// With C=1 and size=1 LRN reduces to b = a / (k + alpha*a^2)^beta.
	x := tensor.New([]float64{0.5, -1.0, 2.0, 3.0}, 1, 1, 4)
	l := NewLocalResponseNorm(1, WithLRNAlpha(0.1), WithLRNBeta(0.5), WithLRNK(2.0))
	y := l.Forward(x)
	for i, a := range x.Data {
		want := a / math.Pow(2.0+0.1*a*a, 0.5)
		if math.Abs(y.Data[i]-want) > 1e-12 {
			t.Errorf("LRN single channel [%d]: got %g, want %g", i, y.Data[i], want)
		}
	}
}

func TestLocalResponseNormBruteForce(t *testing.T) {
	x := seededRandn(110, 2, 5, 3)
	for _, size := range []int{1, 2, 3, 5, 7} {
		l := NewLocalResponseNorm(size, WithLRNAlpha(0.2), WithLRNBeta(0.6), WithLRNK(1.5))
		y := l.Forward(x)
		want := lrnRef(x, size, 0.2, 0.6, 1.5)
		if !dataClose(y.Data, want, 1e-12) {
			t.Errorf("LRN size=%d: output differs from brute-force reference", size)
		}
	}
}

func TestLocalResponseNormDefaultsAndShape(t *testing.T) {
	l := NewLocalResponseNorm(2)
	if l.Alpha != 1e-4 || l.Beta != 0.75 || l.K != 1.0 {
		t.Fatalf("LRN defaults: got alpha=%g beta=%g k=%g", l.Alpha, l.Beta, l.K)
	}
	// Works for any (N, C, spatial...) rank; 5D shape is preserved.
	x := seededRandn(112, 2, 4, 2, 3, 2)
	y := l.Forward(x)
	if !shapeEq(y.Shape, x.Shape) {
		t.Fatalf("LRN 5D shape: got %v, want %v", y.Shape, x.Shape)
	}
	if !dataClose(y.Data, lrnRef(x, 2, 1e-4, 0.75, 1.0), 1e-12) {
		t.Fatalf("LRN 5D output differs from brute-force reference")
	}
}

func TestGradCheckLocalResponseNorm(t *testing.T) {
	// Large alpha so the normalizer contributes meaningfully to the gradient.
	l := NewLocalResponseNorm(3, WithLRNAlpha(0.5), WithLRNK(1.5))
	x := seededRandn(111, 2, 4, 3).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return l.Forward(x).Square().Mean() }
	gradCheck(t, "LocalResponseNorm", loss, []*tensor.Tensor{x}, gcEps, gcTol, 0)
}
