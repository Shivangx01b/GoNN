package quant

import (
	"fmt"

	"gonn/tensor"
)

// QTensorPerChannel is an int8 per-channel quantized tensor: every slice along
// Axis has its own scale/zero-point, and
//
//	real value = (q - ZeroPoints[c]) * Scales[c]
//
// for elements whose index along Axis is c. This mirrors PyTorch's per-channel
// (per-axis) scheme; for Linear/Conv weights PyTorch uses ch_axis=0, i.e. one
// scale per output channel.
type QTensorPerChannel struct {
	Data       []int8
	Scales     []float64
	ZeroPoints []int
	Axis       int
	Shape      []int
}

// Numel returns the number of elements.
func (q QTensorPerChannel) Numel() int { return len(q.Data) }

// Channels returns the number of quantization channels (the size of Axis).
func (q QTensorPerChannel) Channels() int { return len(q.Scales) }

// QuantizePerChannel quantizes t symmetrically per channel along axis:
// for each slice c along axis, scale_c = max|slice_c| / 127 and zero_point = 0
// (PyTorch's per_channel_symmetric scheme used for weights). A slice that is
// all zeros gets the degenerate scale 1.
func QuantizePerChannel(t *tensor.Tensor, axis int) QTensorPerChannel {
	rank := len(t.Shape)
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		panic(fmt.Sprintf("quant: QuantizePerChannel axis %d out of range for shape %v", axis, t.Shape))
	}
	channels := t.Shape[axis]
	outer, inner := 1, 1
	for i := 0; i < axis; i++ {
		outer *= t.Shape[i]
	}
	for i := axis + 1; i < rank; i++ {
		inner *= t.Shape[i]
	}

	q := QTensorPerChannel{
		Data:       make([]int8, len(t.Data)),
		Scales:     make([]float64, channels),
		ZeroPoints: make([]int, channels),
		Axis:       axis,
		Shape:      append([]int(nil), t.Shape...),
	}
	for c := 0; c < channels; c++ {
		// Pass 1: symmetric scale from the channel's max magnitude.
		maxAbs := 0.0
		for o := 0; o < outer; o++ {
			base := (o*channels + c) * inner
			for i := 0; i < inner; i++ {
				v := t.Data[base+i]
				if v < 0 {
					v = -v
				}
				if v > maxAbs {
					maxAbs = v
				}
			}
		}
		scale := 1.0
		if maxAbs != 0 {
			scale = maxAbs / float64(QMax)
		}
		q.Scales[c] = scale
		// Pass 2: quantize the channel (zero_point = 0, symmetric).
		for o := 0; o < outer; o++ {
			base := (o*channels + c) * inner
			for i := 0; i < inner; i++ {
				q.Data[base+i] = quantizeValue(t.Data[base+i], scale, 0)
			}
		}
	}
	return q
}

// DequantizePerChannel maps q back to float:
// x = (q - ZeroPoints[c]) * Scales[c] with c the element's index along Axis.
func DequantizePerChannel(q QTensorPerChannel) *tensor.Tensor {
	rank := len(q.Shape)
	channels := q.Shape[q.Axis]
	outer, inner := 1, 1
	for i := 0; i < q.Axis; i++ {
		outer *= q.Shape[i]
	}
	for i := q.Axis + 1; i < rank; i++ {
		inner *= q.Shape[i]
	}
	d := make([]float64, len(q.Data))
	for c := 0; c < channels; c++ {
		scale := q.Scales[c]
		zp := float64(q.ZeroPoints[c])
		for o := 0; o < outer; o++ {
			base := (o*channels + c) * inner
			for i := 0; i < inner; i++ {
				d[base+i] = (float64(q.Data[base+i]) - zp) * scale
			}
		}
	}
	return tensor.New(d, q.Shape...)
}
