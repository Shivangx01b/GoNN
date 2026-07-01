// This file implements quantization-aware training (QAT) in the PyTorch
// eager style: FakeQuant modules simulate int8 quantization in the float
// forward pass (with straight-through gradients so the underlying float
// parameters keep training), MovingAverageMinMaxObserver tracks
// activation/weight ranges with an EMA, and Convert freezes the trained
// model into an integer-kernel StaticLinear.

package quant

import (
	"fmt"
	"math"

	"gonn/nn"
	"gonn/tensor"
)

// DefaultAveragingConstant is PyTorch's default averaging_constant for
// MovingAverageMinMaxObserver (0.01).
const DefaultAveragingConstant = 0.01

// MovingAverageMinMaxObserver tracks an exponential moving average of the
// observed min/max, like torch.ao.quantization.MovingAverageMinMaxObserver.
// The first observation initializes Min/Max directly; every later observation
// folds the batch extremes in with
//
//	Min = (1-c)*Min + c*min(x)
//	Max = (1-c)*Max + c*max(x)
//
// where c is Momentum (PyTorch's averaging_constant). The zero value is
// usable: Momentum == 0 means DefaultAveragingConstant (0.01).
type MovingAverageMinMaxObserver struct {
	Momentum float64
	Min, Max float64
	seen     bool
}

// Observe folds t's value range into the running EMA min/max.
func (o *MovingAverageMinMaxObserver) Observe(t *tensor.Tensor) {
	if len(t.Data) == 0 {
		return
	}
	mn, mx := t.Data[0], t.Data[0]
	for _, v := range t.Data[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	if !o.seen {
		o.Min, o.Max = mn, mx
		o.seen = true
		return
	}
	c := o.Momentum
	if c == 0 {
		c = DefaultAveragingConstant
	}
	o.Min = (1-c)*o.Min + c*mn
	o.Max = (1-c)*o.Max + c*mx
}

// Seen reports whether any values have been observed.
func (o *MovingAverageMinMaxObserver) Seen() bool { return o.seen }

// ComputeQParams converts the EMA range into per-tensor affine qparams, like
// MinMaxObserver.ComputeQParams. Before any observation it returns the
// degenerate scale=1, zeroPoint=0.
func (o *MovingAverageMinMaxObserver) ComputeQParams() (scale float64, zeroPoint int) {
	if !o.seen {
		return 1, 0
	}
	return affineQParams(o.Min, o.Max)
}

// symmetricRangeScale converts an observed [mn, mx] range into a symmetric
// scale: max(|mn|, |mx|) / 127 with zero_point 0 (degenerate range -> 1).
func symmetricRangeScale(mn, mx float64) float64 {
	m := math.Max(math.Abs(mn), math.Abs(mx))
	if m == 0 {
		return 1
	}
	return m / float64(QMax)
}

// FakeQuant simulates int8 quantization in a float forward pass, the analog
// of torch.ao.quantization.FakeQuantize. It embeds nn.Base, so it composes in
// nn module trees (train/eval mode propagates from the parent).
//
// In training mode each Forward first updates the observer with the input,
// recomputes the qparams, and then emits quantize(dequantize(x)) with a
// clamped straight-through gradient: the incoming gradient passes through
// unchanged where the input lies inside the representable range
// [(QMin-zp)*scale, (QMax-zp)*scale] and is zeroed where the input saturated.
// In eval mode the observer is frozen: the last qparams computed during
// training are reused and the observer is not updated.
type FakeQuant struct {
	nn.Base
	Observer  *MovingAverageMinMaxObserver
	Symmetric bool // symmetric (zero_point=0, weight-style) vs affine (activation-style)

	// Scale/ZeroPoint are the qparams from the most recent training-mode
	// Forward (frozen and reused in eval mode). Before any training forward
	// they are the degenerate 1/0.
	Scale     float64
	ZeroPoint int
}

// NewFakeQuant builds a FakeQuant with a fresh default observer. symmetric
// selects the weight scheme (per-tensor symmetric, zero_point = 0); affine
// (asymmetric) is the activation scheme.
func NewFakeQuant(symmetric bool) *FakeQuant {
	return &FakeQuant{
		Observer:  &MovingAverageMinMaxObserver{},
		Symmetric: symmetric,
		Scale:     1,
	}
}

// QParams returns the current frozen qparams (scale, zero point).
func (f *FakeQuant) QParams() (scale float64, zeroPoint int) {
	return f.Scale, f.ZeroPoint
}

func (f *FakeQuant) computeQParams() (float64, int) {
	if !f.Observer.Seen() {
		return 1, 0
	}
	if f.Symmetric {
		return symmetricRangeScale(f.Observer.Min, f.Observer.Max), 0
	}
	return affineQParams(f.Observer.Min, f.Observer.Max)
}

// Forward fake-quantizes x: out = (quantize(x) - zp) * scale, element-wise.
// Training mode updates the observer and qparams first; eval mode uses the
// frozen qparams. The output participates in autograd with the clamped
// straight-through estimator described on the type.
func (f *FakeQuant) Forward(x *tensor.Tensor) *tensor.Tensor {
	if f.Training() {
		f.Observer.Observe(x)
		f.Scale, f.ZeroPoint = f.computeQParams()
	}
	scale, zp := f.Scale, f.ZeroPoint

	out := tensor.Zeros(x.Shape...)
	for i, v := range x.Data {
		q := quantizeValue(v, scale, zp)
		out.Data[i] = (float64(q) - float64(zp)) * scale
	}

	// Clamped STE: pass gradient only where the input was representable.
	qminReal := (float64(QMin) - float64(zp)) * scale
	qmaxReal := (float64(QMax) - float64(zp)) * scale
	tensor.MakeNode(out, "FakeQuant", []*tensor.Tensor{x}, func(grad *tensor.Tensor) []*tensor.Tensor {
		g := tensor.Zeros(x.Shape...)
		for i, v := range x.Data {
			if v >= qminReal && v <= qmaxReal {
				g.Data[i] = grad.Data[i]
			}
		}
		return []*tensor.Tensor{g}
	})
	return out
}

// QATLinear is a Linear prepared for quantization-aware training, the analog
// of torch.ao.nn.qat.Linear: the float weight and bias stay trainable, while
// a weight FakeQuant (per-tensor symmetric) and an input-activation FakeQuant
// (affine) simulate int8 inference during the forward pass:
//
//	y = fakequant(x) @ fakequant(W)^T + b
//
// The fake-quantized weight is recomputed from the live float weight on every
// forward, so gradients flow through the straight-through estimator into the
// real weight. After training, Convert freezes the model into a StaticLinear.
type QATLinear struct {
	nn.Base
	L        *nn.Linear
	WeightFQ *FakeQuant // per-tensor symmetric, observes the weight
	InputFQ  *FakeQuant // affine, observes the input activations
}

// NewQATLinearFrom wraps a float Linear for QAT. The Linear's parameters are
// exposed through the QATLinear's parameter tree, and train/eval mode
// propagates to both FakeQuants (Eval() freezes the observed qparams).
func NewQATLinearFrom(l *nn.Linear) *QATLinear {
	q := &QATLinear{
		L:        l,
		WeightFQ: NewFakeQuant(true),
		InputFQ:  NewFakeQuant(false),
	}
	q.RegisterChild("linear", l)
	q.RegisterChild("weight_fq", q.WeightFQ)
	q.RegisterChild("input_fq", q.InputFQ)
	return q
}

// Forward computes fakequant(x) @ fakequant(W)^T + b with full autograd.
func (q *QATLinear) Forward(x *tensor.Tensor) *tensor.Tensor {
	origShape := x.Shape
	feat := origShape[len(origShape)-1]
	if feat != q.L.InFeatures {
		panic(fmt.Sprintf("quant: QATLinear input last dim %d != InFeatures %d", feat, q.L.InFeatures))
	}
	batch := 1
	for i := 0; i < len(origShape)-1; i++ {
		batch *= origShape[i]
	}

	xq := q.InputFQ.Forward(x)
	wq := q.WeightFQ.Forward(q.L.Weight)

	y := xq.Reshape(batch, feat).MatMul(wq.Transpose())
	if q.L.Bias != nil {
		y = y.Add(q.L.Bias)
	}
	outShape := append([]int(nil), origShape[:len(origShape)-1]...)
	outShape = append(outShape, q.L.OutFeatures)
	return y.Reshape(outShape...)
}

// Convert freezes the QAT model into an integer-kernel StaticLinear:
//   - the trained float weight is quantized with the weight FakeQuant's frozen
//     qparams (per-tensor symmetric), or per output row when
//     WithPerChannelWeights() is passed (per-channel scales are derived from
//     the trained weight itself, like PyTorch's per_channel_symmetric);
//   - the input FakeQuant's frozen qparams become the static input qparams.
//
// With the default per-tensor weights the converted module reproduces the
// QATLinear's eval-mode outputs up to float64 rounding, because eval-mode
// fake quantization and the integer GEMM compute the same quantized values.
func (q *QATLinear) Convert(opts ...LinearOption) *StaticLinear {
	o := applyLinearOptions(opts)
	l := q.L
	in, out := l.InFeatures, l.OutFeatures

	wScales := make([]float64, out)
	if o.perChannel {
		qw := QuantizePerChannel(l.Weight, 0)
		copy(wScales, qw.Scales)
	} else {
		s, _ := q.WeightFQ.QParams()
		for i := range wScales {
			wScales[i] = s
		}
	}
	var bias []float64
	if l.Bias != nil {
		bias = l.Bias.Data
	}
	inScale, inZP := q.InputFQ.QParams()
	return &StaticLinear{
		qLinearCore: buildQLinearCore(in, out, l.Weight.Data, bias, wScales, o.perChannel),
		InScale:     inScale,
		InZeroPoint: inZP,
	}
}
