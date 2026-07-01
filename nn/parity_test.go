package nn

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gonn/tensor"
)

// This file locks in a pre-refactor baseline: for one instance of every
// module type it records (a) the parameter list (count, order, shapes) and
// (b) a forward-pass signature on a fixed input with the global RNG seeded,
// which pins both the weight-init draw order and the forward numerics.
//
// Regenerate with:  GONN_UPDATE_GOLDEN=1 go test ./nn -run TestModuleParity

const parityGolden = "testdata/parity.golden"

func paramSig(params []*tensor.Tensor) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "params=%d", len(params))
	for _, p := range params {
		fmt.Fprintf(&sb, " %v", p.Shape)
	}
	return sb.String()
}

func fwdSig(out *tensor.Tensor) string {
	sum := 0.0
	for _, v := range out.Data {
		sum += v
	}
	s := fmt.Sprintf("shape=%v sum=%.12g", out.Shape, sum)
	n := 3
	if len(out.Data) < n {
		n = len(out.Data)
	}
	for i := 0; i < n; i++ {
		s += fmt.Sprintf(" v%d=%.12g", i, out.Data[i])
	}
	return s
}

type paramsOwner interface {
	Parameters() []*tensor.Tensor
}

func sig(m paramsOwner, out *tensor.Tensor) string {
	return paramSig(m.Parameters()) + " | " + fwdSig(out)
}

func TestModuleParity(t *testing.T) {
	entries := []struct {
		name string
		run  func() string
	}{
		{"Linear(4,3,bias)", func() string {
			rand.Seed(101)
			m := NewLinear(4, 3, true)
			return sig(m, m.Forward(tensor.Randn(2, 4)))
		}},
		{"Linear(4,3,nobias)", func() string {
			rand.Seed(102)
			m := NewLinear(4, 3, false)
			return sig(m, m.Forward(tensor.Randn(2, 4)))
		}},
		{"Conv1d(2,3,k3,s2,p1)", func() string {
			rand.Seed(103)
			m := NewConv1d(2, 3, 3, WithStride(2), WithPad(1))
			return sig(m, m.Forward(tensor.Randn(2, 2, 7)))
		}},
		{"Conv2d(2,3,k3,s2,p1)", func() string {
			rand.Seed(104)
			m := NewConv2d(2, 3, 3, WithStride(2), WithPad(1))
			return sig(m, m.Forward(tensor.Randn(2, 2, 6, 7)))
		}},
		{"Conv2dHW(2,3,k32,s21,p10)", func() string {
			rand.Seed(105)
			m := NewConv2d(2, 3, 3, WithKernel(3, 2), WithStride(2, 1), WithPad(1, 0))
			return sig(m, m.Forward(tensor.Randn(2, 2, 6, 7)))
		}},
		{"Conv3d(2,2,k2,s1,p0)", func() string {
			rand.Seed(106)
			m := NewConv3d(2, 2, 2)
			return sig(m, m.Forward(tensor.Randn(2, 2, 3, 4, 4)))
		}},
		{"ConvTranspose1d(2,3,k3,s2,p1)", func() string {
			rand.Seed(107)
			m := NewConvTranspose1d(2, 3, 3, WithStride(2), WithPad(1))
			return sig(m, m.Forward(tensor.Randn(2, 2, 5)))
		}},
		{"ConvTranspose2d(2,3,k3,s2,p1)", func() string {
			rand.Seed(108)
			m := NewConvTranspose2d(2, 3, 3, WithStride(2), WithPad(1))
			return sig(m, m.Forward(tensor.Randn(2, 2, 4, 4)))
		}},
		{"ConvTranspose3d(2,2,k2,s2,p0)", func() string {
			rand.Seed(109)
			m := NewConvTranspose3d(2, 2, 2, WithStride(2))
			return sig(m, m.Forward(tensor.Randn(1, 2, 3, 3, 3)))
		}},
		{"MaxPool1d(2,2)", func() string {
			rand.Seed(110)
			m := NewMaxPool1d(2)
			return sig(m, m.Forward(tensor.Randn(2, 2, 6)))
		}},
		{"AvgPool1d(2,2)", func() string {
			rand.Seed(111)
			m := NewAvgPool1d(2)
			return sig(m, m.Forward(tensor.Randn(2, 2, 6)))
		}},
		{"MaxPool2d(2,2)", func() string {
			rand.Seed(112)
			m := NewMaxPool2d(2)
			return sig(m, m.Forward(tensor.Randn(2, 2, 4, 4)))
		}},
		{"AvgPool2d(2,2)", func() string {
			rand.Seed(113)
			m := NewAvgPool2d(2)
			return sig(m, m.Forward(tensor.Randn(2, 2, 4, 4)))
		}},
		{"MaxPool3d(2,2)", func() string {
			rand.Seed(114)
			m := NewMaxPool3d(2)
			return sig(m, m.Forward(tensor.Randn(1, 2, 4, 4, 4)))
		}},
		{"AvgPool3d(2,2)", func() string {
			rand.Seed(115)
			m := NewAvgPool3d(2)
			return sig(m, m.Forward(tensor.Randn(1, 2, 4, 4, 4)))
		}},
		{"AdaptiveAvgPool1d(3)", func() string {
			rand.Seed(116)
			m := NewAdaptiveAvgPool1d(3)
			return sig(m, m.Forward(tensor.Randn(2, 2, 7)))
		}},
		{"AdaptiveMaxPool2d(2,2)", func() string {
			rand.Seed(117)
			m := NewAdaptiveMaxPool2d(2, 2)
			return sig(m, m.Forward(tensor.Randn(2, 2, 5, 5)))
		}},
		{"AdaptiveAvgPool2d(2,2)", func() string {
			rand.Seed(118)
			m := NewAdaptiveAvgPool2d(2, 2)
			return sig(m, m.Forward(tensor.Randn(2, 2, 5, 5)))
		}},
		{"AdaptiveAvgPool3d(2,2,2)", func() string {
			rand.Seed(119)
			m := NewAdaptiveAvgPool3d(2, 2, 2)
			return sig(m, m.Forward(tensor.Randn(1, 2, 5, 5, 5)))
		}},
		{"LayerNorm(6)", func() string {
			rand.Seed(120)
			m := NewLayerNorm(6)
			return sig(m, m.Forward(tensor.Randn(3, 6)))
		}},
		{"RMSNorm(6)", func() string {
			rand.Seed(121)
			m := NewRMSNorm(6)
			return sig(m, m.Forward(tensor.Randn(3, 6)))
		}},
		{"BatchNorm1d(4)-train", func() string {
			rand.Seed(122)
			m := NewBatchNorm1d(4)
			return sig(m, m.Forward(tensor.Randn(5, 4)))
		}},
		{"BatchNorm1d(4)-eval", func() string {
			rand.Seed(123)
			m := NewBatchNorm1d(4)
			x := tensor.Randn(5, 4)
			m.Forward(x) // one training step to move running stats
			m.SetTraining(false)
			return sig(m, m.Forward(x))
		}},
		{"BatchNorm2d(3)-train", func() string {
			rand.Seed(124)
			m := NewBatchNorm2d(3)
			return sig(m, m.Forward(tensor.Randn(2, 3, 4, 4)))
		}},
		{"GroupNorm(2,4)", func() string {
			rand.Seed(125)
			m := NewGroupNorm(2, 4)
			return sig(m, m.Forward(tensor.Randn(2, 4, 5)))
		}},
		{"InstanceNorm1d(2,affine)", func() string {
			rand.Seed(126)
			m := NewInstanceNorm1d(2, WithAffine(true))
			return sig(m, m.Forward(tensor.Randn(2, 2, 6)))
		}},
		{"InstanceNorm2d(2,noaffine)", func() string {
			rand.Seed(127)
			m := NewInstanceNorm2d(2)
			return sig(m, m.Forward(tensor.Randn(2, 2, 4, 4)))
		}},
		{"Embedding(5,3)", func() string {
			rand.Seed(128)
			m := NewEmbedding(5, 3)
			return sig(m, m.Forward(tensor.New([]float64{0, 2, 4, 2}, 4)))
		}},
		{"PReLU(3)", func() string {
			rand.Seed(129)
			m := NewPReLU(3)
			return sig(m, m.Forward(tensor.Randn(2, 3, 4)))
		}},
		{"GLU(dim=-1)", func() string {
			rand.Seed(130)
			m := NewGLU(-1)
			return sig(m, m.Forward(tensor.Randn(3, 8)))
		}},
		{"Bilinear(3,4,2,bias)", func() string {
			rand.Seed(131)
			m := NewBilinear(3, 4, 2, true)
			return sig(m, m.Forward(tensor.Randn(2, 3), tensor.Randn(2, 4)))
		}},
		{"Flatten(1,-1)", func() string {
			rand.Seed(132)
			m := NewFlatten(1, -1)
			return sig(m, m.Forward(tensor.Randn(2, 3, 4)))
		}},
		{"Unflatten(1,[2,3])", func() string {
			rand.Seed(133)
			m := NewUnflatten(1, 2, 3)
			return sig(m, m.Forward(tensor.Randn(2, 6)))
		}},
		{"CosineSimilarity(1)", func() string {
			rand.Seed(134)
			m := NewCosineSimilarity(1, 1e-8)
			return sig(m, m.Forward(tensor.Randn(3, 5), tensor.Randn(3, 5)))
		}},
		{"PairwiseDistance(2)", func() string {
			rand.Seed(135)
			m := NewPairwiseDistance(2, 1e-8)
			return sig(m, m.Forward(tensor.Randn(3, 5), tensor.Randn(3, 5)))
		}},
		{"ZeroPad2d(1,1,1,1)", func() string {
			rand.Seed(136)
			m := NewZeroPad2d(1, 1, 1, 1)
			return sig(m, m.Forward(tensor.Randn(1, 2, 3, 3)))
		}},
		{"ConstantPad2d(1,0,2,1,v=2.5)", func() string {
			rand.Seed(137)
			m := NewConstantPad2d(1, 0, 2, 1, 2.5)
			return sig(m, m.Forward(tensor.Randn(1, 2, 3, 3)))
		}},
		{"ReflectionPad2d(1,1,1,1)", func() string {
			rand.Seed(138)
			m := NewReflectionPad2d(1, 1, 1, 1)
			return sig(m, m.Forward(tensor.Randn(1, 2, 3, 3)))
		}},
		{"ReplicationPad2d(2,1,1,2)", func() string {
			rand.Seed(139)
			m := NewReplicationPad2d(2, 1, 1, 2)
			return sig(m, m.Forward(tensor.Randn(1, 2, 3, 3)))
		}},
		{"Upsample(2,nearest)", func() string {
			rand.Seed(140)
			m := NewUpsample(2, "nearest")
			return sig(m, m.Forward(tensor.Randn(1, 2, 3, 3)))
		}},
		{"Upsample(2,bilinear)", func() string {
			rand.Seed(141)
			m := NewUpsample(2, "bilinear")
			return sig(m, m.Forward(tensor.Randn(1, 2, 3, 3)))
		}},
		{"PixelShuffle(2)", func() string {
			rand.Seed(142)
			m := NewPixelShuffle(2)
			return sig(m, m.Forward(tensor.Randn(1, 8, 3, 3)))
		}},
		{"PixelUnshuffle(2)", func() string {
			rand.Seed(143)
			m := NewPixelUnshuffle(2)
			return sig(m, m.Forward(tensor.Randn(1, 2, 6, 6)))
		}},
		{"RNN(3,4)", func() string {
			rand.Seed(144)
			m := NewRNN(3, 4)
			return sig(m, m.Forward(tensor.Randn(2, 3, 3)))
		}},
		{"LSTM(3,4)", func() string {
			rand.Seed(145)
			m := NewLSTM(3, 4)
			return sig(m, m.Forward(tensor.Randn(2, 3, 3)))
		}},
		{"GRU(3,4)", func() string {
			rand.Seed(146)
			m := NewGRU(3, 4)
			return sig(m, m.Forward(tensor.Randn(2, 3, 3)))
		}},
		{"MultiLayerRNN(3,4,2,bidir)", func() string {
			rand.Seed(147)
			m := NewRNN(3, 4, WithLayers(2), WithBidirectional())
			return sig(m, m.Forward(tensor.Randn(2, 3, 3)))
		}},
		{"MultiLayerLSTM(3,4,2)", func() string {
			rand.Seed(148)
			m := NewLSTM(3, 4, WithLayers(2))
			return sig(m, m.Forward(tensor.Randn(2, 3, 3)))
		}},
		{"MultiLayerGRU(3,4,2,bidir)", func() string {
			rand.Seed(149)
			m := NewGRU(3, 4, WithLayers(2), WithBidirectional())
			return sig(m, m.Forward(tensor.Randn(2, 3, 3)))
		}},
		{"MultiHeadAttention(8,2)-causal", func() string {
			rand.Seed(150)
			m := NewMultiHeadAttention(8, 2)
			x := tensor.Randn(2, 3, 8)
			return sig(m, m.Forward(x, x, x, true))
		}},
		{"TransformerEncoderLayer(8,2,16)", func() string {
			rand.Seed(151)
			m := NewTransformerEncoderLayer(8, 2, 16)
			return sig(m, m.Forward(tensor.Randn(2, 3, 8)))
		}},
		{"TransformerEncoder(2,8,2,16)", func() string {
			rand.Seed(152)
			m := NewTransformerEncoder(2, 8, 2, 16)
			return sig(m, m.Forward(tensor.Randn(2, 3, 8)))
		}},
		{"TransformerDecoderLayer(8,2,16)", func() string {
			rand.Seed(153)
			m := NewTransformerDecoderLayer(8, 2, 16)
			return sig(m, m.Forward(tensor.Randn(2, 3, 8), tensor.Randn(2, 4, 8)))
		}},
		{"Seq2Seq(5,6,4,8)", func() string {
			rand.Seed(154)
			m := NewSeq2Seq(5, 6, 4, 8)
			src := tensor.New([]float64{0, 1, 2, 3, 4, 0}, 2, 3)
			tgt := tensor.New([]float64{0, 1, 2, 3}, 2, 2)
			return sig(m, m.Forward(src, tgt))
		}},
		{"Sequential(Lin-ReLU-Lin)", func() string {
			rand.Seed(155)
			m := NewSequential(NewLinear(4, 8, true), ReLU(), NewLinear(8, 2, true))
			return sig(m, m.Forward(tensor.Randn(3, 4)))
		}},
		{"Softmax(axis1)", func() string {
			rand.Seed(156)
			m := NewSoftmax(1)
			return paramSig(nil) + " | " + fwdSig(m.Forward(tensor.Randn(3, 5)))
		}},
		{"GELU-module", func() string {
			rand.Seed(157)
			m := GELU()
			return paramSig(nil) + " | " + fwdSig(m.Forward(tensor.Randn(3, 5)))
		}},
		{"SiLU-module", func() string {
			rand.Seed(158)
			m := SiLU()
			return paramSig(nil) + " | " + fwdSig(m.Forward(tensor.Randn(3, 5)))
		}},
	}

	var sb strings.Builder
	for _, e := range entries {
		fmt.Fprintf(&sb, "%s\t%s\n", e.name, e.run())
	}
	got := sb.String()

	if os.Getenv("GONN_UPDATE_GOLDEN") != "" {
		if err := os.MkdirAll(filepath.Dir(parityGolden), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(parityGolden, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("updated %s (%d entries)", parityGolden, len(entries))
		return
	}

	wantBytes, err := os.ReadFile(parityGolden)
	if err != nil {
		t.Fatalf("read golden (run with GONN_UPDATE_GOLDEN=1 to create): %v", err)
	}
	// Normalize line endings: git autocrlf checkouts on Windows materialize
	// the golden file with CRLF while the generated signature uses LF.
	want := strings.ReplaceAll(string(wantBytes), "\r\n", "\n")
	if got == want {
		return
	}
	gotLines := strings.Split(strings.TrimRight(got, "\n"), "\n")
	wantLines := strings.Split(strings.TrimRight(want, "\n"), "\n")
	for i := 0; i < len(gotLines) || i < len(wantLines); i++ {
		var g, w string
		if i < len(gotLines) {
			g = gotLines[i]
		}
		if i < len(wantLines) {
			w = wantLines[i]
		}
		if g != w {
			t.Errorf("parity mismatch line %d:\n  got:  %s\n  want: %s", i+1, g, w)
		}
	}
}
