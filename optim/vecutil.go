package optim

// Parameter <-> flat-vector utilities, mirroring
// torch.nn.utils.parameters_to_vector / vector_to_parameters. These are
// plain data copies OUTSIDE the autograd graph (PyTorch's versions are
// differentiable views-then-cat; GoNN uses them for the same practical
// purposes — LBFGS-style flat updates, checkpoint diffs, EMA snapshots —
// where gradients through the packing are never wanted). Parameters are
// copied in slice order, each flattened in its underlying (contiguous
// row-major) data order.

import (
	"fmt"

	"gonn/tensor"
)

// ParametersToVector concatenates every parameter into one flat 1-D tensor
// (a copy — mutating the result does not touch the parameters). Panics on a
// nil parameter: unlike the grad-clipping helpers there is no sensible way
// to skip entries and keep VectorToParameters symmetric.
func ParametersToVector(params []*tensor.Tensor) *tensor.Tensor {
	total := 0
	for i, p := range params {
		if p == nil {
			panic(fmt.Sprintf("optim.ParametersToVector: params[%d] is nil", i))
		}
		total += p.Numel()
	}
	data := make([]float64, 0, total)
	for _, p := range params {
		data = append(data, p.Data...)
	}
	return tensor.New(data, total)
}

// VectorToParameters copies slices of the flat 1-D tensor vec back into the
// parameters, in the same order ParametersToVector packed them. Panics if
// vec's element count does not match the total parameter count, or on a nil
// parameter.
func VectorToParameters(vec *tensor.Tensor, params []*tensor.Tensor) {
	total := 0
	for i, p := range params {
		if p == nil {
			panic(fmt.Sprintf("optim.VectorToParameters: params[%d] is nil", i))
		}
		total += p.Numel()
	}
	if vec.Numel() != total {
		panic(fmt.Sprintf("optim.VectorToParameters: vector has %d elements, parameters need %d", vec.Numel(), total))
	}
	off := 0
	for _, p := range params {
		copy(p.Data, vec.Data[off:off+p.Numel()])
		off += p.Numel()
	}
}

// GradsToVector concatenates every parameter's gradient into one flat 1-D
// tensor (a copy), in the same layout as ParametersToVector. A parameter
// with a nil gradient contributes zeros (PyTorch has no direct equivalent;
// this keeps the result aligned with ParametersToVector for optimizer-style
// flat math). Panics on a nil parameter.
func GradsToVector(params []*tensor.Tensor) *tensor.Tensor {
	total := 0
	for i, p := range params {
		if p == nil {
			panic(fmt.Sprintf("optim.GradsToVector: params[%d] is nil", i))
		}
		total += p.Numel()
	}
	data := make([]float64, 0, total)
	for _, p := range params {
		if p.Grad == nil {
			data = append(data, make([]float64, p.Numel())...)
		} else {
			data = append(data, p.Grad.Data...)
		}
	}
	return tensor.New(data, total)
}
