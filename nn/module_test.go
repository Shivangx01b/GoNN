package nn

import (
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"gonn/tensor"
)

var tensorPtrType = reflect.TypeOf((*tensor.Tensor)(nil))

// collectGradTensors walks v reflectively and returns every reachable
// *tensor.Tensor with RequiresGrad set — the ground truth that Parameters()
// must cover (catches forgotten reg() calls in constructors). Values reached
// through unexported fields cannot be Interface()'d, so tensors are detected
// by type and recovered via their pointer instead.
func collectGradTensors(v reflect.Value, seen map[uintptr]bool, out *[]*tensor.Tensor) {
	switch v.Kind() {
	case reflect.Ptr:
		if v.IsNil() {
			return
		}
		if v.Type() == tensorPtrType {
			t := (*tensor.Tensor)(unsafe.Pointer(v.Pointer()))
			if t.RequiresGrad && !seen[v.Pointer()] {
				seen[v.Pointer()] = true
				*out = append(*out, t)
			}
			return
		}
		if seen[v.Pointer()] {
			return
		}
		seen[v.Pointer()] = true
		collectGradTensors(v.Elem(), seen, out)
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			collectGradTensors(v.Field(i), seen, out)
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			collectGradTensors(v.Index(i), seen, out)
		}
	case reflect.Interface:
		if !v.IsNil() {
			collectGradTensors(v.Elem(), seen, out)
		}
	}
}

// TestRegistrationCompleteness constructs one instance of every parameterful
// module type and asserts that every grad-requiring tensor reachable from it
// appears in Parameters().
func TestRegistrationCompleteness(t *testing.T) {
	modules := map[string]Child{
		"Linear":          NewLinear(4, 3, true),
		"Bilinear":        NewBilinear(3, 4, 2, true),
		"Conv1d":          NewConv1d(2, 3, 3),
		"Conv2d":          NewConv2d(2, 3, 3),
		"Conv3d":          NewConv3d(2, 2, 2),
		"ConvTranspose1d": NewConvTranspose1d(2, 3, 3),
		"ConvTranspose2d": NewConvTranspose2d(2, 3, 3),
		"ConvTranspose3d": NewConvTranspose3d(2, 2, 2),
		"LayerNorm":       NewLayerNorm(6),
		"RMSNorm":         NewRMSNorm(6),
		"BatchNorm1d":     NewBatchNorm1d(4),
		"BatchNorm2d":     NewBatchNorm2d(3),
		"GroupNorm":       NewGroupNorm(2, 4),
		"InstanceNorm1d":  NewInstanceNorm1d(2, WithAffine(true)),
		"InstanceNorm2d":  NewInstanceNorm2d(2, WithAffine(true)),
		"Embedding":       NewEmbedding(5, 3),
		"PReLU":           NewPReLU(3),
		"RNNCell":         NewRNNCell(3, 4),
		"LSTMCell":        NewLSTMCell(3, 4),
		"GRUCell":         NewGRUCell(3, 4),
		"RNN":             NewRNN(3, 4, WithLayers(2), WithBidirectional()),
		"LSTM":            NewLSTM(3, 4, WithLayers(2)),
		"GRU":             NewGRU(3, 4, WithBidirectional()),
		"Seq2Seq":         NewSeq2Seq(5, 6, 4, 8),
		"MHA":             NewMultiHeadAttention(8, 2),
		"TransformerEnc":  NewTransformerEncoder(2, 8, 2, 16),
		"TransformerDecL": NewTransformerDecoderLayer(8, 2, 16),
		"Sequential":      NewSequential(NewLinear(4, 8, true), ReLU(), NewLinear(8, 2, true)),
	}
	for name, m := range modules {
		var reachable []*tensor.Tensor
		collectGradTensors(reflect.ValueOf(m), map[uintptr]bool{}, &reachable)
		registered := map[*tensor.Tensor]bool{}
		for _, p := range m.Parameters() {
			registered[p] = true
		}
		for _, rt := range reachable {
			if !registered[rt] {
				t.Errorf("%s: reachable grad tensor not in Parameters() (forgot reg()?)", name)
			}
		}
		if len(m.Parameters()) != len(reachable) {
			t.Errorf("%s: Parameters() has %d entries, %d grad tensors reachable",
				name, len(m.Parameters()), len(reachable))
		}
	}
}

// TestTrainEvalPropagates verifies SetTraining recurses through containers.
func TestTrainEvalPropagates(t *testing.T) {
	drop := NewDropout(0.5)
	bn := NewBatchNorm1d(4)
	model := NewSequential(NewLinear(4, 4, true), drop, bn)

	if !drop.Training() || !bn.Training() {
		t.Fatal("modules must start in training mode")
	}
	model.Eval()
	if drop.Training() || bn.Training() {
		t.Fatal("Eval() did not propagate to children")
	}
	model.Train()
	if !drop.Training() || !bn.Training() {
		t.Fatal("Train() did not propagate to children")
	}

	// Eval-mode Dropout must be the identity.
	model.Eval()
	x := tensor.Randn(3, 4)
	if y := drop.Forward(x); y != x {
		t.Fatal("eval-mode Dropout is not identity")
	}
}

// TestNamedParameters verifies hierarchical dotted names and ordering.
func TestNamedParameters(t *testing.T) {
	model := NewSequential(NewLinear(4, 8, true), ReLU(), NewLinear(8, 2, false))
	named := model.NamedParameters()
	wantNames := []string{"0.weight", "0.bias", "2.weight"}
	if len(named) != len(wantNames) {
		t.Fatalf("got %d named params, want %d", len(named), len(wantNames))
	}
	for i, w := range wantNames {
		if named[i].Name != w {
			t.Fatalf("named[%d] = %q, want %q", i, named[i].Name, w)
		}
	}
	// Order must match Parameters().
	ps := model.Parameters()
	for i := range ps {
		if named[i].T != ps[i] {
			t.Fatalf("NamedParameters()[%d] and Parameters()[%d] differ", i, i)
		}
	}

	// FilterParams selects by name.
	weights := FilterParams(model, func(name string) bool {
		return strings.HasSuffix(name, ".weight")
	})
	if len(weights) != 2 {
		t.Fatalf("FilterParams(.weight) got %d, want 2", len(weights))
	}
}

// TestBuffersRegistered verifies BatchNorm running stats appear as buffers,
// not parameters.
func TestBuffersRegistered(t *testing.T) {
	bn := NewBatchNorm2d(3)
	bufs := bn.Buffers()
	if len(bufs) != 2 {
		t.Fatalf("BatchNorm2d buffers = %d, want 2", len(bufs))
	}
	names := bufs[0].Name + "," + bufs[1].Name
	if !strings.Contains(names, "running_mean") || !strings.Contains(names, "running_var") {
		t.Fatalf("unexpected buffer names %q", names)
	}
	if len(bn.Parameters()) != 2 { // weight, bias only
		t.Fatalf("BatchNorm2d params = %d, want 2", len(bn.Parameters()))
	}
}

// TestConvDilation covers the new dilation capability against a hand
// computation: 1D conv, kernel 2, dilation 2 -> taps x[i] and x[i+2].
func TestConvDilation(t *testing.T) {
	c := NewConv1d(1, 1, 2, WithDilation(2), WithNoBias())
	c.Weight.Data[0], c.Weight.Data[1] = 1, 10
	x := tensor.New([]float64{1, 2, 3, 4, 5}, 1, 1, 5)
	y := c.Forward(x)
	// out length = (5 - 2*(2-1) - 1)/1 + 1 = 3; y[i] = x[i] + 10*x[i+2]
	want := []float64{31, 42, 53}
	if !intsEqual(y.Shape, []int{1, 1, 3}) {
		t.Fatalf("dilated conv shape %v, want [1 1 3]", y.Shape)
	}
	for i, w := range want {
		if y.Data[i] != w {
			t.Fatalf("dilated conv [%d] = %v, want %v", i, y.Data[i], w)
		}
	}
}
