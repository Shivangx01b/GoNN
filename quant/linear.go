package quant

import (
	"fmt"
	"math"

	"gonn/nn"
	"gonn/tensor"
)

// LinearOption configures how a quantized Linear is built. Options are shared
// by NewDynamicLinearFrom, NewStaticLinearFrom, and QATLinear.Convert.
type LinearOption func(*linearOptions)

type linearOptions struct {
	perChannel bool
}

// WithPerChannelWeights quantizes the weight per output channel (axis 0, one
// symmetric scale per output row — PyTorch's per_channel_symmetric weight
// scheme) instead of the default per-tensor symmetric scale. Per-channel
// scales adapt to each row's magnitude, so weight matrices whose rows differ
// wildly in scale lose far less precision.
func WithPerChannelWeights() LinearOption {
	return func(o *linearOptions) { o.perChannel = true }
}

func applyLinearOptions(opts []LinearOption) linearOptions {
	var o linearOptions
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// qLinearCore holds the shared state of the quantized Linear variants:
// symmetric int8 weights (zero_point = 0; one scale per tensor or per output
// row), float64 bias, and the int32-accumulating GEMM. Inference-only:
// Forward returns a plain tensor with no autograd graph.
type qLinearCore struct {
	InFeatures  int
	OutFeatures int
	WQ          []int8    // (Out, In) row-major, symmetric int8
	WScale      float64   // per-tensor weight scale; 0 when PerChannel
	WScales     []float64 // per-output-row weight scales (len Out); in
	// per-tensor mode every entry equals WScale.
	PerChannel bool
	WRowSum    []int32 // per-row sums of WQ, for input zero-point correction
	Bias       []float64
}

// buildQLinearCore quantizes the (out, in) row-major weight w with the given
// per-row symmetric scales (zero_point = 0) and precomputes the row sums used
// for input zero-point correction.
func buildQLinearCore(in, out int, w, bias, wScales []float64, perChannel bool) qLinearCore {
	c := qLinearCore{
		InFeatures:  in,
		OutFeatures: out,
		WScales:     wScales,
		PerChannel:  perChannel,
	}
	if !perChannel {
		c.WScale = wScales[0]
	}
	c.WQ = make([]int8, out*in)
	c.WRowSum = make([]int32, out)
	for o := 0; o < out; o++ {
		s := wScales[o]
		var sum int32
		for i := 0; i < in; i++ {
			q := quantizeValue(w[o*in+i], s, 0)
			c.WQ[o*in+i] = q
			sum += int32(q)
		}
		c.WRowSum[o] = sum
	}
	if bias != nil {
		c.Bias = append([]float64(nil), bias...)
	}
	return c
}

func newQLinearCore(l *nn.Linear, o linearOptions) qLinearCore {
	in, out := l.InFeatures, l.OutFeatures
	wScales := make([]float64, out)
	if o.perChannel {
		q := QuantizePerChannel(l.Weight, 0)
		copy(wScales, q.Scales)
	} else {
		s := symmetricQParams(l.Weight.Data)
		for i := range wScales {
			wScales[i] = s
		}
	}
	var bias []float64
	if l.Bias != nil {
		bias = l.Bias.Data
	}
	return buildQLinearCore(in, out, l.Weight.Data, bias, wScales, o.perChannel)
}

// forward quantizes x with the given input qparams, runs the int8 x int8 ->
// int32 GEMM, and dequantizes with the combined scale:
//
//	y[b,o] = sx*sw[o] * (sum_k xq[b,k]*wq[o,k] - zx * sum_k wq[o,k]) + bias[o]
//
// which is exact in the quantized domain because the weight zero point is 0.
// sw[o] is the per-tensor weight scale, or the row's own scale in per-channel
// mode.
func (c *qLinearCore) forward(x *tensor.Tensor, inScale float64, inZeroPoint int) *tensor.Tensor {
	origShape := x.Shape
	feat := origShape[len(origShape)-1]
	if feat != c.InFeatures {
		panic(fmt.Sprintf("quant: linear input last dim %d != InFeatures %d", feat, c.InFeatures))
	}
	batch := 1
	for i := 0; i < len(origShape)-1; i++ {
		batch *= origShape[i]
	}

	xq := make([]int8, len(x.Data))
	for i, v := range x.Data {
		xq[i] = quantizeValue(v, inScale, inZeroPoint)
	}

	in, out := c.InFeatures, c.OutFeatures
	zx := int32(inZeroPoint)
	y := make([]float64, batch*out)
	for b := 0; b < batch; b++ {
		xrow := xq[b*in : (b+1)*in]
		for o := 0; o < out; o++ {
			wrow := c.WQ[o*in : (o+1)*in]
			var acc int32
			for k := 0; k < in; k++ {
				acc += int32(xrow[k]) * int32(wrow[k])
			}
			v := inScale * c.WScales[o] * float64(acc-zx*c.WRowSum[o])
			if c.Bias != nil {
				v += c.Bias[o]
			}
			y[b*out+o] = v
		}
	}

	outShape := append([]int(nil), origShape[:len(origShape)-1]...)
	outShape = append(outShape, out)
	return tensor.New(y, outShape...)
}

// DynamicLinear is a dynamically quantized Linear, the analog of
// torch.ao.nn.quantized.dynamic.Linear: weights are quantized once
// (symmetric int8, per-tensor by default or per-channel with
// WithPerChannelWeights), and on every Forward the input's affine qparams are
// computed from its own min/max before the integer GEMM.
// Inference-only; the returned tensor carries no autograd graph.
type DynamicLinear struct {
	qLinearCore
}

// NewDynamicLinearFrom converts a trained float Linear into a DynamicLinear.
// The original layer is not modified. Pass WithPerChannelWeights() to
// quantize the weight per output row.
func NewDynamicLinearFrom(l *nn.Linear, opts ...LinearOption) *DynamicLinear {
	return &DynamicLinear{qLinearCore: newQLinearCore(l, applyLinearOptions(opts))}
}

// Forward computes the quantized x @ W^T + b, deriving the input scale and
// zero point dynamically from x's min/max.
func (d *DynamicLinear) Forward(x *tensor.Tensor) *tensor.Tensor {
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, v := range x.Data {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	if len(x.Data) == 0 {
		mn, mx = 0, 0
	}
	scale, zp := affineQParams(mn, mx)
	return d.forward(x, scale, zp)
}

// StaticLinear is a statically quantized Linear: the input qparams are fixed
// at conversion time from a calibrated observer (the analog of eager-mode
// static quantization, where observers record activation ranges during a
// calibration pass and the converted module uses the frozen qparams).
type StaticLinear struct {
	qLinearCore
	InScale     float64
	InZeroPoint int
}

// NewStaticLinearFrom converts a trained float Linear using the activation
// range recorded by obs (calibrate first by calling obs.Observe on
// representative inputs). The original layer is not modified. Pass
// WithPerChannelWeights() to quantize the weight per output row.
func NewStaticLinearFrom(l *nn.Linear, obs *MinMaxObserver, opts ...LinearOption) *StaticLinear {
	scale, zp := obs.ComputeQParams()
	return &StaticLinear{
		qLinearCore: newQLinearCore(l, applyLinearOptions(opts)),
		InScale:     scale,
		InZeroPoint: zp,
	}
}

// Forward computes the quantized x @ W^T + b with the calibrated input
// qparams. Inputs outside the calibrated range saturate.
func (s *StaticLinear) Forward(x *tensor.Tensor) *tensor.Tensor {
	return s.forward(x, s.InScale, s.InZeroPoint)
}
