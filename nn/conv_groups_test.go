package nn

import (
	"math"
	"math/rand"
	"testing"

	"gonn/tensor"
)

func expectPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic", name)
		}
	}()
	f()
}

// naiveConv2dGrouped is a direct-loop reference for grouped 2D convolution.
// x: (N, InC, H, W); w: (OutC, InC/g, KH, KW); b: (OutC,) or nil.
func naiveConv2dGrouped(x, w, b *tensor.Tensor, groups int, stride, pad, dil [2]int) *tensor.Tensor {
	N, inC, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outC, inCg, KH, KW := w.Shape[0], w.Shape[1], w.Shape[2], w.Shape[3]
	outCg := outC / groups
	OH := (H+2*pad[0]-dil[0]*(KH-1)-1)/stride[0] + 1
	OW := (W+2*pad[1]-dil[1]*(KW-1)-1)/stride[1] + 1
	out := tensor.Zeros(N, outC, OH, OW)
	for n := 0; n < N; n++ {
		for oc := 0; oc < outC; oc++ {
			grp := oc / outCg
			for oh := 0; oh < OH; oh++ {
				for ow := 0; ow < OW; ow++ {
					sum := 0.0
					if b != nil {
						sum = b.Data[oc]
					}
					for icl := 0; icl < inCg; icl++ {
						ic := grp*inCg + icl
						for kh := 0; kh < KH; kh++ {
							for kw := 0; kw < KW; kw++ {
								ih := oh*stride[0] + kh*dil[0] - pad[0]
								iw := ow*stride[1] + kw*dil[1] - pad[1]
								if ih < 0 || ih >= H || iw < 0 || iw >= W {
									continue
								}
								sum += w.Data[((oc*inCg+icl)*KH+kh)*KW+kw] *
									x.Data[((n*inC+ic)*H+ih)*W+iw]
							}
						}
					}
					out.Data[((n*outC+oc)*OH+oh)*OW+ow] = sum
				}
			}
		}
	}
	return out
}

// naiveConvTranspose2dGrouped is a direct-loop reference for grouped 2D
// transposed convolution. x: (N, InC, H, W); w: (InC, OutC/g, KH, KW).
func naiveConvTranspose2dGrouped(x, w, b *tensor.Tensor, groups int, stride, pad, dil, outPad [2]int) *tensor.Tensor {
	N, inC, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outCg, KH, KW := w.Shape[1], w.Shape[2], w.Shape[3]
	outC := outCg * groups
	inCg := inC / groups
	OH := (H-1)*stride[0] - 2*pad[0] + dil[0]*(KH-1) + outPad[0] + 1
	OW := (W-1)*stride[1] - 2*pad[1] + dil[1]*(KW-1) + outPad[1] + 1
	out := tensor.Zeros(N, outC, OH, OW)
	if b != nil {
		for n := 0; n < N; n++ {
			for oc := 0; oc < outC; oc++ {
				for i := 0; i < OH*OW; i++ {
					out.Data[(n*outC+oc)*OH*OW+i] = b.Data[oc]
				}
			}
		}
	}
	for n := 0; n < N; n++ {
		for ic := 0; ic < inC; ic++ {
			grp := ic / inCg
			for ih := 0; ih < H; ih++ {
				for iw := 0; iw < W; iw++ {
					v := x.Data[((n*inC+ic)*H+ih)*W+iw]
					for ocl := 0; ocl < outCg; ocl++ {
						oc := grp*outCg + ocl
						for kh := 0; kh < KH; kh++ {
							for kw := 0; kw < KW; kw++ {
								oh := ih*stride[0] + kh*dil[0] - pad[0]
								ow := iw*stride[1] + kw*dil[1] - pad[1]
								if oh < 0 || oh >= OH || ow < 0 || ow >= OW {
									continue
								}
								out.Data[((n*outC+oc)*OH+oh)*OW+ow] +=
									v * w.Data[((ic*outCg+ocl)*KH+kh)*KW+kw]
							}
						}
					}
				}
			}
		}
	}
	return out
}

func requireSameTensor(t *testing.T, name string, got, want *tensor.Tensor, tol float64) {
	t.Helper()
	if !intsEqual(got.Shape, want.Shape) {
		t.Fatalf("%s: shape %v != %v", name, got.Shape, want.Shape)
	}
	for i := range want.Data {
		if math.Abs(got.Data[i]-want.Data[i]) > tol {
			t.Fatalf("%s: data[%d] = %.12g, want %.12g", name, i, got.Data[i], want.Data[i])
		}
	}
}

// Depthwise conv (groups == inC == outC) hand-checked against the definition:
// out[n,c,i,j] = sum_{kh,kw} w[c,0,kh,kw] * x[n,c,i+kh,j+kw].
func TestConv2dDepthwiseHandCheck(t *testing.T) {
	c := NewConv2d(2, 2, 2, WithGroups(2), WithNoBias())
	if !intsEqual(c.Weight.Shape, []int{2, 1, 2, 2}) {
		t.Fatalf("depthwise weight shape %v, want [2 1 2 2]", c.Weight.Shape)
	}
	copy(c.Weight.Data, []float64{
		1, 2, 3, 4, // channel 0 kernel
		-1, 0.5, 2, -2, // channel 1 kernel
	})
	// x: (1, 2, 3, 3); channel 0 = 0..8, channel 1 = 1..9.
	xData := make([]float64, 18)
	for i := 0; i < 9; i++ {
		xData[i] = float64(i)
		xData[9+i] = float64(i + 1)
	}
	x := tensor.New(xData, 1, 2, 3, 3)
	got := c.Forward(x)

	want := tensor.Zeros(1, 2, 2, 2)
	w := [][]float64{{1, 2, 3, 4}, {-1, 0.5, 2, -2}}
	for ch := 0; ch < 2; ch++ {
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				sum := 0.0
				for kh := 0; kh < 2; kh++ {
					for kw := 0; kw < 2; kw++ {
						sum += w[ch][kh*2+kw] * xData[ch*9+(i+kh)*3+(j+kw)]
					}
				}
				want.Data[(ch*2+i)*2+j] = sum
			}
		}
	}
	requireSameTensor(t, "depthwise", got, want, 1e-12)
}

func TestConv2dGroupsMatchesNaive(t *testing.T) {
	c := NewConv2d(4, 6, 3, WithGroups(2), WithStride(2), WithPad(1), WithDilation(1))
	x := seededRandn(51, 2, 4, 6, 7)
	got := c.Forward(x)
	want := naiveConv2dGrouped(x, c.Weight, c.Bias, 2, [2]int{2, 2}, [2]int{1, 1}, [2]int{1, 1})
	requireSameTensor(t, "Conv2d groups=2", got, want, 1e-10)
}

func TestConvTranspose2dGroupsMatchesNaive(t *testing.T) {
	c := NewConvTranspose2d(4, 6, 3, WithGroups(2), WithStride(2), WithPad(1), WithOutputPadding(1))
	x := seededRandn(52, 2, 4, 4, 5)
	got := c.Forward(x)
	want := naiveConvTranspose2dGrouped(x, c.Weight, c.Bias, 2,
		[2]int{2, 2}, [2]int{1, 1}, [2]int{1, 1}, [2]int{1, 1})
	requireSameTensor(t, "ConvTranspose2d groups=2", got, want, 1e-10)
}

func TestGradCheckConv2dGroups(t *testing.T) {
	c := NewConv2d(4, 4, 3, WithGroups(2), WithStride(2), WithPad(1))
	x := seededRandn(53, 2, 4, 6, 7).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return c.Forward(x).Square().Mean() }
	gradCheck(t, "Conv2dGroups", loss, append(c.Parameters(), x), gcEps, gcTol, 40)
}

func TestGradCheckConvTranspose2dGroups(t *testing.T) {
	c := NewConvTranspose2d(4, 4, 3, WithGroups(2), WithStride(2), WithPad(1))
	x := seededRandn(54, 2, 4, 4, 4).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return c.Forward(x).Square().Mean() }
	gradCheck(t, "ConvTranspose2dGroups", loss, append(c.Parameters(), x), gcEps, gcTol, 40)
}

// WithGroups(1) must be bit-identical to omitting the option: same RNG draw
// count and order, same weight/bias values, same forward numerics.
func TestGroupsOneMatchesDefault(t *testing.T) {
	rand.Seed(7001)
	a := NewConv2d(3, 4, 3, WithStride(2), WithPad(1))
	after1 := rand.Float64()
	rand.Seed(7001)
	b := NewConv2d(3, 4, 3, WithStride(2), WithPad(1), WithGroups(1))
	after2 := rand.Float64()
	if after1 != after2 {
		t.Fatalf("WithGroups(1) consumed a different number of RNG draws")
	}
	if !intsEqual(a.Weight.Shape, b.Weight.Shape) {
		t.Fatalf("weight shape %v != %v", a.Weight.Shape, b.Weight.Shape)
	}
	for i := range a.Weight.Data {
		if a.Weight.Data[i] != b.Weight.Data[i] {
			t.Fatalf("weight[%d]: %v != %v", i, a.Weight.Data[i], b.Weight.Data[i])
		}
	}
	for i := range a.Bias.Data {
		if a.Bias.Data[i] != b.Bias.Data[i] {
			t.Fatalf("bias[%d]: %v != %v", i, a.Bias.Data[i], b.Bias.Data[i])
		}
	}
	x := seededRandn(55, 2, 3, 6, 7)
	requireSameTensor(t, "groups=1 forward", b.Forward(x), a.Forward(x), 0)

	rand.Seed(7002)
	ta := NewConvTranspose2d(3, 4, 3, WithStride(2))
	rand.Seed(7002)
	tb := NewConvTranspose2d(3, 4, 3, WithStride(2), WithGroups(1))
	for i := range ta.Weight.Data {
		if ta.Weight.Data[i] != tb.Weight.Data[i] {
			t.Fatalf("transposed weight[%d]: %v != %v", i, ta.Weight.Data[i], tb.Weight.Data[i])
		}
	}
}

func TestGroupsWeightShapeAndFanIn(t *testing.T) {
	c := NewConv2d(4, 6, 3, WithGroups(2), WithNoBias())
	if !intsEqual(c.Weight.Shape, []int{6, 2, 3, 3}) {
		t.Fatalf("conv weight shape %v, want [6 2 3 3]", c.Weight.Shape)
	}
	// fanIn = (4/2)*9 = 18 -> bound = sqrt(1/18); all draws must respect it.
	bound := math.Sqrt(1.0 / 18.0)
	for i, v := range c.Weight.Data {
		if v < -bound || v > bound {
			t.Fatalf("weight[%d] = %v outside init bound %v", i, v, bound)
		}
	}

	ct := NewConvTranspose2d(4, 6, 3, WithGroups(2), WithNoBias())
	if !intsEqual(ct.Weight.Shape, []int{4, 3, 3, 3}) {
		t.Fatalf("transposed weight shape %v, want [4 3 3 3]", ct.Weight.Shape)
	}
}

func TestGroupsValidation(t *testing.T) {
	expectPanic(t, "inC not divisible", func() { NewConv2d(3, 4, 3, WithGroups(2)) })
	expectPanic(t, "outC not divisible", func() { NewConv2d(4, 3, 3, WithGroups(2)) })
	expectPanic(t, "groups < 1", func() { NewConv2d(4, 4, 3, WithGroups(0)) })
	expectPanic(t, "transposed inC not divisible", func() { NewConvTranspose1d(3, 4, 2, WithGroups(2)) })
}

// ---- output_padding -------------------------------------------------------

func TestConvTransposeOutputPaddingShapes(t *testing.T) {
	// 1D: out = (L-1)*s - 2*p + d*(K-1) + op + 1.
	c1 := NewConvTranspose1d(2, 3, 3, WithStride(3), WithPad(1), WithOutputPadding(2))
	y1 := c1.Forward(seededRandn(56, 2, 2, 5))
	if want := (5-1)*3 - 2*1 + (3 - 1) + 2 + 1; !intsEqual(y1.Shape, []int{2, 3, want}) {
		t.Fatalf("1d shape %v, want [2 3 %d]", y1.Shape, want)
	}

	// 2D with per-dim output padding and dilation.
	c2 := NewConvTranspose2d(2, 3, 3, WithStride(3, 2), WithPad(1), WithDilation(2), WithOutputPadding(2, 1))
	y2 := c2.Forward(seededRandn(57, 2, 2, 4, 5))
	wantH := (4-1)*3 - 2*1 + 2*(3-1) + 2 + 1
	wantW := (5-1)*2 - 2*1 + 2*(3-1) + 1 + 1
	if !intsEqual(y2.Shape, []int{2, 3, wantH, wantW}) {
		t.Fatalf("2d shape %v, want [2 3 %d %d]", y2.Shape, wantH, wantW)
	}

	// 3D broadcast output padding.
	c3 := NewConvTranspose3d(2, 2, 2, WithStride(2), WithOutputPadding(1))
	y3 := c3.Forward(seededRandn(58, 1, 2, 3, 3, 3))
	if want := (3-1)*2 + (2 - 1) + 1 + 1; !intsEqual(y3.Shape, []int{1, 2, want, want, want}) {
		t.Fatalf("3d shape %v, want all-%d spatial", y3.Shape, want)
	}
}

// Hand check: 1D transposed conv, stride 2, kernel 2, output_padding 1.
// y = sum_i x[i] * w scattered at i*stride, plus one trailing zero position:
// [x0*a, x0*b, x1*a, x1*b, 0].
func TestConvTranspose1dOutputPaddingHandCheck(t *testing.T) {
	c := NewConvTranspose1d(1, 1, 2, WithStride(2), WithOutputPadding(1), WithNoBias())
	a, b := 2.0, -3.0
	copy(c.Weight.Data, []float64{a, b})
	x0, x1 := 5.0, 7.0
	y := c.Forward(tensor.New([]float64{x0, x1}, 1, 1, 2))
	want := tensor.New([]float64{x0 * a, x0 * b, x1 * a, x1 * b, 0}, 1, 1, 5)
	requireSameTensor(t, "outpad hand check", y, want, 1e-12)
}

func TestGradCheckConvTranspose2dOutputPadding(t *testing.T) {
	c := NewConvTranspose2d(2, 3, 2, WithStride(2), WithOutputPadding(1))
	x := seededRandn(59, 2, 2, 3, 3).SetRequiresGrad(true)
	loss := func() *tensor.Tensor { return c.Forward(x).Square().Mean() }
	gradCheck(t, "ConvTranspose2dOutPad", loss, append(c.Parameters(), x), gcEps, gcTol, 40)
}

func TestOutputPaddingValidation(t *testing.T) {
	expectPanic(t, "outPad >= stride", func() {
		NewConvTranspose2d(2, 2, 3, WithStride(2), WithOutputPadding(2))
	})
	expectPanic(t, "outPad on forward conv", func() {
		NewConv2d(2, 2, 3, WithOutputPadding(1))
	})
	expectPanic(t, "negative outPad", func() {
		NewConvTranspose2d(2, 2, 3, WithStride(2), WithOutputPadding(-1))
	})
}
