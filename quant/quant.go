// Package quant implements int8 per-tensor affine quantization in the style
// of PyTorch eager-mode dynamic/static quantization.
//
// Scope (documented, deliberate):
//   - CPU-only pure-Go integer kernels. PyTorch's eager quantization is also
//     CPU-only; GPU quantized kernels are out of scope.
//   - Inference-only: quantized layers return plain tensors with no autograd
//     graph. Quantization-aware training is out of scope.
//   - Per-tensor quantization only (no per-channel).
//
// Semantics follow PyTorch's per-tensor affine scheme: for activations the
// mapping is q = clamp(round(x/scale) + zero_point, -128, 127) with an
// asymmetric (affine) scale/zero-point derived from the observed min/max
// range nudged to include zero. Weights use symmetric quantization
// (zero_point = 0, scale = max|w|/127, values clamped to [-127, 127]).
//
// There is no module-tree rewriting helper (PyTorch's quantize_dynamic walks
// the model via hooks/module replacement, which this framework does not
// have). Convert layers explicitly instead:
//
//	ql := quant.NewDynamicLinearFrom(model.FC1) // once, after training
//	y := ql.Forward(x)                          // use in place of model.FC1
package quant

import (
	"fmt"
	"math"

	"gonn/tensor"
)

// Quantized integer range for int8 storage.
const (
	QMin = -128
	QMax = 127
)

// QTensor is an int8 per-tensor affine quantized tensor:
// real value = (q - ZeroPoint) * Scale.
type QTensor struct {
	Data      []int8
	Scale     float64
	ZeroPoint int
	Shape     []int
}

// Numel returns the number of elements.
func (q QTensor) Numel() int { return len(q.Data) }

// Quantize quantizes t with the given per-tensor affine parameters:
// q = clamp(round(x/scale) + zeroPoint, -128, 127).
func Quantize(t *tensor.Tensor, scale float64, zeroPoint int) QTensor {
	if scale <= 0 {
		panic(fmt.Sprintf("quant: scale must be positive, got %g", scale))
	}
	data := make([]int8, len(t.Data))
	for i, v := range t.Data {
		data[i] = quantizeValue(v, scale, zeroPoint)
	}
	return QTensor{
		Data:      data,
		Scale:     scale,
		ZeroPoint: zeroPoint,
		Shape:     append([]int(nil), t.Shape...),
	}
}

// Dequantize maps q back to float: x = (q - zeroPoint) * scale.
func Dequantize(q QTensor) *tensor.Tensor {
	d := make([]float64, len(q.Data))
	for i, v := range q.Data {
		d[i] = (float64(v) - float64(q.ZeroPoint)) * q.Scale
	}
	return tensor.New(d, q.Shape...)
}

func quantizeValue(v, scale float64, zeroPoint int) int8 {
	q := int(math.Round(v/scale)) + zeroPoint
	if q < QMin {
		q = QMin
	} else if q > QMax {
		q = QMax
	}
	return int8(q)
}

// affineQParams derives asymmetric (affine) scale/zero-point from a float
// range, PyTorch style: the range is nudged to include zero, scale spans the
// full int8 range, and the zero point is the integer that maps real 0.0
// exactly. A degenerate (empty) range yields scale=1, zeroPoint=0.
func affineQParams(minVal, maxVal float64) (scale float64, zeroPoint int) {
	mn := math.Min(0, minVal)
	mx := math.Max(0, maxVal)
	scale = (mx - mn) / float64(QMax-QMin)
	if scale == 0 {
		return 1, 0
	}
	zp := QMin - int(math.Round(mn/scale))
	if zp < QMin {
		zp = QMin
	} else if zp > QMax {
		zp = QMax
	}
	return scale, zp
}

// symmetricQParams derives a symmetric weight scale: zero_point = 0 and
// scale = max|w| / 127, so weights land in [-127, 127].
func symmetricQParams(data []float64) (scale float64) {
	maxAbs := 0.0
	for _, v := range data {
		if a := math.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 {
		return 1
	}
	return maxAbs / float64(QMax)
}

// MinMaxObserver records the running min/max of every observed tensor and
// converts the range into per-tensor affine int8 qparams (asymmetric,
// qmin=-128, qmax=127), like torch.ao.quantization.MinMaxObserver.
type MinMaxObserver struct {
	Min, Max float64
	seen     bool
}

// Observe folds t's value range into the running min/max.
func (o *MinMaxObserver) Observe(t *tensor.Tensor) {
	for _, v := range t.Data {
		if !o.seen {
			o.Min, o.Max = v, v
			o.seen = true
			continue
		}
		if v < o.Min {
			o.Min = v
		}
		if v > o.Max {
			o.Max = v
		}
	}
}

// Seen reports whether any values have been observed.
func (o *MinMaxObserver) Seen() bool { return o.seen }

// ComputeQParams converts the observed range into affine qparams. Before any
// observation it returns the degenerate scale=1, zeroPoint=0.
func (o *MinMaxObserver) ComputeQParams() (scale float64, zeroPoint int) {
	if !o.seen {
		return 1, 0
	}
	return affineQParams(o.Min, o.Max)
}
