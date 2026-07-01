package nn

import (
	"fmt"

	"gonn/tensor"
)

// N-dimensional padding machinery shared by every padding layer (1d/2d/3d).
// A padding is fully described by per-dim (before, after) amounts plus a
// per-dim index-mapping function that sends each output coordinate to the
// input coordinate it reads (or -1 for "fill"). The resulting flat index map
// is applied with indexMapGather (gather.go): one 0/1 gather row per output
// cell, so the whole operation is a single matmul and differentiable by
// construction (on -tags cuda builds the gather matmul dispatches to cuBLAS).

// padModeFunc maps a (possibly out-of-range) output-relative coordinate
// i = outCoord - before to a source coordinate in [0, size), or -1 to mean
// "fill value" (zero/constant padding outside the input).
type padModeFunc func(i, size int) int

// zeroPadIndex is the zero/constant mode: identity inside, -1 outside.
func zeroPadIndex(i, size int) int {
	if i < 0 || i >= size {
		return -1
	}
	return i
}

// circularIndex wraps the coordinate around: index modulo size, matching
// PyTorch's CircularPad ("values at the end pad the beginning and vice
// versa"). PyTorch allows wrapping around at most once (pad <= size), which
// callers enforce before building the map.
func circularIndex(i, size int) int {
	m := i % size
	if m < 0 {
		m += size
	}
	return m
}

// padND computes the output spatial dims and the flat output→input index map
// for padding spatial dims in with per-dim before/after amounts under mode.
// indexMap[r] is the input flat index read by output cell r, or -1 for fill.
func padND(in, before, after []int, mode padModeFunc) (out, indexMap []int) {
	nd := len(in)
	out = make([]int, nd)
	for d := 0; d < nd; d++ {
		out[d] = in[d] + before[d] + after[d]
		if out[d] <= 0 {
			panic(fmt.Sprintf("nn: padding produces non-positive output size %d for dim %d (in=%d before=%d after=%d)",
				out[d], d, in[d], before[d], after[d]))
		}
	}
	inStrides := rowMajorStrides(in)
	total := prodInts(out)
	indexMap = make([]int, total)
	idx := make([]int, nd)
	for r := 0; r < total; r++ {
		flat := 0
		for d := 0; d < nd; d++ {
			src := mode(idx[d]-before[d], in[d])
			if src < 0 {
				flat = -1
				break
			}
			flat += src * inStrides[d]
		}
		indexMap[r] = flat
		incMultiIndex(idx, out)
	}
	return out, indexMap
}

// applyPadND applies a flat index map to the spatial dims of x
// (N, C, spatial...) via indexMapGather, returning (N, C, out...).
func applyPadND(x *tensor.Tensor, out, indexMap []int) *tensor.Tensor {
	N, C := x.Shape[0], x.Shape[1]
	S := prodInts(x.Shape[2:])
	g := indexMapGather(indexMap, S)
	res := x.Reshape(N*C, S).MatMul(g.Transpose()) // (N*C, prod(out))
	return res.Reshape(append([]int{N, C}, out...)...)
}

// constantFill builds a (1, 1, out...) tensor holding value at every fill
// cell (indexMap == -1) and 0 elsewhere; added (broadcast over N, C) after
// the gather to realize ConstantPad.
func constantFill(out, indexMap []int, value float64) *tensor.Tensor {
	data := make([]float64, len(indexMap))
	for r, src := range indexMap {
		if src < 0 {
			data[r] = value
		}
	}
	return tensor.New(data, append([]int{1, 1}, out...)...)
}

// checkRank panics unless x has the expected rank.
func checkRank(name string, x *tensor.Tensor, rank int, want string) {
	if len(x.Shape) != rank {
		panic(fmt.Sprintf("%s: expected %dD input %s, got shape %v", name, rank, want, x.Shape))
	}
}

// checkReflectPad enforces PyTorch's ReflectionPad constraint: padding size
// must be strictly less than the corresponding input dimension.
func checkReflectPad(name string, in, before, after []int) {
	for d := range in {
		if before[d] >= in[d] || after[d] >= in[d] {
			panic(fmt.Sprintf("%s: padding size should be less than the corresponding input dimension, but got padding (%d, %d) for input size %d at spatial dim %d",
				name, before[d], after[d], in[d], d))
		}
	}
}

// checkCircularPad enforces PyTorch's CircularPad constraint: padding size
// must be at most the corresponding input dimension (wrap around at most
// once).
func checkCircularPad(name string, in, before, after []int) {
	for d := range in {
		if before[d] > in[d] || after[d] > in[d] {
			panic(fmt.Sprintf("%s: padding size should be at most the corresponding input dimension, but got padding (%d, %d) for input size %d at spatial dim %d",
				name, before[d], after[d], in[d], d))
		}
	}
}

// pad1d validates a (N, C, L) input and applies mode padding to L.
func pad1d(name string, x *tensor.Tensor, left, right int, mode padModeFunc) *tensor.Tensor {
	checkRank(name, x, 3, "(N, C, L)")
	out, im := padND([]int{x.Shape[2]}, []int{left}, []int{right}, mode)
	return applyPadND(x, out, im)
}

// pad3d validates a (N, C, D, H, W) input and applies mode padding. Pads
// follow PyTorch order: left/right on W, top/bottom on H, front/back on D.
func pad3d(name string, x *tensor.Tensor, left, right, top, bottom, front, back int, mode padModeFunc) *tensor.Tensor {
	checkRank(name, x, 5, "(N, C, D, H, W)")
	out, im := padND(x.Shape[2:], []int{front, top, left}, []int{back, bottom, right}, mode)
	return applyPadND(x, out, im)
}

// ---------------------------------------------------------------------------
// Zero padding
// ---------------------------------------------------------------------------

// ZeroPad1d pads (N, C, L) with zeros on the left and right,
// matching torch.nn.ZeroPad1d((left, right)).
type ZeroPad1d struct {
	Base
	Left, Right int
}

// NewZeroPad1d constructs a ZeroPad1d (PyTorch argument order: left, right).
func NewZeroPad1d(left, right int) *ZeroPad1d {
	return &ZeroPad1d{Left: left, Right: right}
}

// Forward applies zero padding to the last dim.
func (p *ZeroPad1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return pad1d("ZeroPad1d", x, p.Left, p.Right, zeroPadIndex)
}

// ZeroPad3d pads (N, C, D, H, W) with zeros on all six sides, matching
// torch.nn.ZeroPad3d((left, right, top, bottom, front, back)).
type ZeroPad3d struct {
	Base
	Left, Right, Top, Bottom, Front, Back int
}

// NewZeroPad3d constructs a ZeroPad3d (PyTorch argument order:
// left, right, top, bottom, front, back).
func NewZeroPad3d(left, right, top, bottom, front, back int) *ZeroPad3d {
	return &ZeroPad3d{Left: left, Right: right, Top: top, Bottom: bottom, Front: front, Back: back}
}

// Forward applies zero padding to the three spatial dims.
func (p *ZeroPad3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return pad3d("ZeroPad3d", x, p.Left, p.Right, p.Top, p.Bottom, p.Front, p.Back, zeroPadIndex)
}

// ---------------------------------------------------------------------------
// Constant padding
// ---------------------------------------------------------------------------

// ConstantPad1d pads (N, C, L) with a fixed scalar Value, matching
// torch.nn.ConstantPad1d((left, right), value).
type ConstantPad1d struct {
	Base
	Left, Right int
	Value       float64
}

// NewConstantPad1d constructs a ConstantPad1d (PyTorch argument order:
// left, right, value).
func NewConstantPad1d(left, right int, value float64) *ConstantPad1d {
	return &ConstantPad1d{Left: left, Right: right, Value: value}
}

// Forward applies constant padding to the last dim.
func (p *ConstantPad1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkRank("ConstantPad1d", x, 3, "(N, C, L)")
	out, im := padND([]int{x.Shape[2]}, []int{p.Left}, []int{p.Right}, zeroPadIndex)
	return applyPadND(x, out, im).Add(constantFill(out, im, p.Value))
}

// ConstantPad3d pads (N, C, D, H, W) with a fixed scalar Value, matching
// torch.nn.ConstantPad3d((left, right, top, bottom, front, back), value).
type ConstantPad3d struct {
	Base
	Left, Right, Top, Bottom, Front, Back int
	Value                                 float64
}

// NewConstantPad3d constructs a ConstantPad3d (PyTorch argument order:
// left, right, top, bottom, front, back, value).
func NewConstantPad3d(left, right, top, bottom, front, back int, value float64) *ConstantPad3d {
	return &ConstantPad3d{Left: left, Right: right, Top: top, Bottom: bottom, Front: front, Back: back, Value: value}
}

// Forward applies constant padding to the three spatial dims.
func (p *ConstantPad3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkRank("ConstantPad3d", x, 5, "(N, C, D, H, W)")
	out, im := padND(x.Shape[2:], []int{p.Front, p.Top, p.Left}, []int{p.Back, p.Bottom, p.Right}, zeroPadIndex)
	return applyPadND(x, out, im).Add(constantFill(out, im, p.Value))
}

// ---------------------------------------------------------------------------
// Reflection padding
// ---------------------------------------------------------------------------

// ReflectionPad1d reflects (N, C, L) across its boundaries without repeating
// the edge value, matching torch.nn.ReflectionPad1d((left, right)).
// Padding must be strictly less than L (PyTorch constraint).
type ReflectionPad1d struct {
	Base
	Left, Right int
}

// NewReflectionPad1d constructs a ReflectionPad1d (PyTorch argument order:
// left, right).
func NewReflectionPad1d(left, right int) *ReflectionPad1d {
	return &ReflectionPad1d{Left: left, Right: right}
}

// Forward applies reflection padding to the last dim.
func (p *ReflectionPad1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkRank("ReflectionPad1d", x, 3, "(N, C, L)")
	checkReflectPad("ReflectionPad1d", x.Shape[2:], []int{p.Left}, []int{p.Right})
	return pad1d("ReflectionPad1d", x, p.Left, p.Right, reflectIndex)
}

// ReflectionPad3d reflects (N, C, D, H, W) across its boundaries, matching
// torch.nn.ReflectionPad3d((left, right, top, bottom, front, back)).
// Each padding must be strictly less than its input dimension.
type ReflectionPad3d struct {
	Base
	Left, Right, Top, Bottom, Front, Back int
}

// NewReflectionPad3d constructs a ReflectionPad3d (PyTorch argument order:
// left, right, top, bottom, front, back).
func NewReflectionPad3d(left, right, top, bottom, front, back int) *ReflectionPad3d {
	return &ReflectionPad3d{Left: left, Right: right, Top: top, Bottom: bottom, Front: front, Back: back}
}

// Forward applies reflection padding to the three spatial dims.
func (p *ReflectionPad3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkRank("ReflectionPad3d", x, 5, "(N, C, D, H, W)")
	checkReflectPad("ReflectionPad3d", x.Shape[2:],
		[]int{p.Front, p.Top, p.Left}, []int{p.Back, p.Bottom, p.Right})
	return pad3d("ReflectionPad3d", x, p.Left, p.Right, p.Top, p.Bottom, p.Front, p.Back, reflectIndex)
}

// ---------------------------------------------------------------------------
// Replication padding
// ---------------------------------------------------------------------------

// ReplicationPad1d replicates the edge values of (N, C, L), matching
// torch.nn.ReplicationPad1d((left, right)).
type ReplicationPad1d struct {
	Base
	Left, Right int
}

// NewReplicationPad1d constructs a ReplicationPad1d (PyTorch argument order:
// left, right).
func NewReplicationPad1d(left, right int) *ReplicationPad1d {
	return &ReplicationPad1d{Left: left, Right: right}
}

// Forward applies replication (edge) padding to the last dim.
func (p *ReplicationPad1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return pad1d("ReplicationPad1d", x, p.Left, p.Right, replicateIndex)
}

// ReplicationPad3d replicates the edge values of (N, C, D, H, W), matching
// torch.nn.ReplicationPad3d((left, right, top, bottom, front, back)).
type ReplicationPad3d struct {
	Base
	Left, Right, Top, Bottom, Front, Back int
}

// NewReplicationPad3d constructs a ReplicationPad3d (PyTorch argument order:
// left, right, top, bottom, front, back).
func NewReplicationPad3d(left, right, top, bottom, front, back int) *ReplicationPad3d {
	return &ReplicationPad3d{Left: left, Right: right, Top: top, Bottom: bottom, Front: front, Back: back}
}

// Forward applies replication (edge) padding to the three spatial dims.
func (p *ReplicationPad3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return pad3d("ReplicationPad3d", x, p.Left, p.Right, p.Top, p.Bottom, p.Front, p.Back, replicateIndex)
}

// ---------------------------------------------------------------------------
// Circular padding
// ---------------------------------------------------------------------------

// CircularPad1d pads (N, C, L) with wrap-around values: the end of the dim
// pads the beginning and vice versa, matching
// torch.nn.CircularPad1d((left, right)). Padding must be at most L.
type CircularPad1d struct {
	Base
	Left, Right int
}

// NewCircularPad1d constructs a CircularPad1d (PyTorch argument order:
// left, right).
func NewCircularPad1d(left, right int) *CircularPad1d {
	return &CircularPad1d{Left: left, Right: right}
}

// Forward applies circular padding to the last dim.
func (p *CircularPad1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkRank("CircularPad1d", x, 3, "(N, C, L)")
	checkCircularPad("CircularPad1d", x.Shape[2:], []int{p.Left}, []int{p.Right})
	return pad1d("CircularPad1d", x, p.Left, p.Right, circularIndex)
}

// CircularPad2d pads (N, C, H, W) with wrap-around values, matching
// torch.nn.CircularPad2d((left, right, top, bottom)). Note the PyTorch
// argument order (left, right, top, bottom) — unlike the historical GoNN 2D
// pad layers, which take (top, bottom, left, right). Padding must be at most
// the corresponding input dimension.
type CircularPad2d struct {
	Base
	Left, Right, Top, Bottom int
}

// NewCircularPad2d constructs a CircularPad2d (PyTorch argument order:
// left, right, top, bottom).
func NewCircularPad2d(left, right, top, bottom int) *CircularPad2d {
	return &CircularPad2d{Left: left, Right: right, Top: top, Bottom: bottom}
}

// Forward applies circular padding to both spatial dims.
func (p *CircularPad2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkRank("CircularPad2d", x, 4, "(N, C, H, W)")
	checkCircularPad("CircularPad2d", x.Shape[2:],
		[]int{p.Top, p.Left}, []int{p.Bottom, p.Right})
	out, im := padND(x.Shape[2:], []int{p.Top, p.Left}, []int{p.Bottom, p.Right}, circularIndex)
	return applyPadND(x, out, im)
}

// CircularPad3d pads (N, C, D, H, W) with wrap-around values, matching
// torch.nn.CircularPad3d((left, right, top, bottom, front, back)).
// Padding must be at most the corresponding input dimension.
type CircularPad3d struct {
	Base
	Left, Right, Top, Bottom, Front, Back int
}

// NewCircularPad3d constructs a CircularPad3d (PyTorch argument order:
// left, right, top, bottom, front, back).
func NewCircularPad3d(left, right, top, bottom, front, back int) *CircularPad3d {
	return &CircularPad3d{Left: left, Right: right, Top: top, Bottom: bottom, Front: front, Back: back}
}

// Forward applies circular padding to the three spatial dims.
func (p *CircularPad3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkRank("CircularPad3d", x, 5, "(N, C, D, H, W)")
	checkCircularPad("CircularPad3d", x.Shape[2:],
		[]int{p.Front, p.Top, p.Left}, []int{p.Back, p.Bottom, p.Right})
	return pad3d("CircularPad3d", x, p.Left, p.Right, p.Top, p.Bottom, p.Front, p.Back, circularIndex)
}
