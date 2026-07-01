package nn

import "gonn/tensor"

// 2D padding layers, built on the shared N-d padding machinery in
// padding_nd.go: each output cell reads at most one input cell (or zero), so
// a single 0/1 gather matrix (indexMapGather) expresses zero/constant/
// reflection/replication padding as one matmul.
//
// Note: these historical 2D layers take (top, bottom, left, right); the
// newer 1d/3d/circular layers in padding_nd.go follow PyTorch tuple order
// (left, right, top, bottom, front, back).

// pad2d validates a (N, C, H, W) input and applies mode padding.
func pad2d(name string, x *tensor.Tensor, top, bottom, left, right int, mode padModeFunc) *tensor.Tensor {
	checkRank(name, x, 4, "(N, C, H, W)")
	out, im := padND(x.Shape[2:], []int{top, left}, []int{bottom, right}, mode)
	return applyPadND(x, out, im)
}

// ZeroPad2d pads (N, C, H, W) with zeros on the four sides.
type ZeroPad2d struct {
	Base
	Top, Bottom, Left, Right int
}

// NewZeroPad2d constructs a ZeroPad2d.
func NewZeroPad2d(top, bottom, left, right int) *ZeroPad2d {
	return &ZeroPad2d{Top: top, Bottom: bottom, Left: left, Right: right}
}

// Forward applies zero padding.
func (p *ZeroPad2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return pad2d("ZeroPad2d", x, p.Top, p.Bottom, p.Left, p.Right, zeroPadIndex)
}

// ConstantPad2d pads with a fixed scalar Value.
type ConstantPad2d struct {
	Base
	Top, Bottom, Left, Right int
	Value                    float64
}

// NewConstantPad2d constructs a ConstantPad2d.
func NewConstantPad2d(top, bottom, left, right int, value float64) *ConstantPad2d {
	return &ConstantPad2d{Top: top, Bottom: bottom, Left: left, Right: right, Value: value}
}

// Forward applies constant padding.
func (p *ConstantPad2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	checkRank("ConstantPad2d", x, 4, "(N, C, H, W)")
	out, im := padND(x.Shape[2:], []int{p.Top, p.Left}, []int{p.Bottom, p.Right}, zeroPadIndex)
	// Gather the interior, then add the border value tensor (broadcast over
	// N, C).
	return applyPadND(x, out, im).Add(constantFill(out, im, p.Value))
}

// reflectIndex maps a (possibly out-of-range) coordinate to the reflected
// index within [0, size). The classic "no edge repeat" reflection used by
// ReflectionPad: ...3 2 1 | 0 1 2 3 | 2 1 0...
func reflectIndex(i, size int) int {
	if size == 1 {
		return 0
	}
	period := 2*size - 2
	// Bring i into [0, period).
	i = i % period
	if i < 0 {
		i += period
	}
	if i >= size {
		i = period - i
	}
	return i
}

// replicateIndex clamps the index to [0, size-1].
func replicateIndex(i, size int) int {
	if i < 0 {
		return 0
	}
	if i >= size {
		return size - 1
	}
	return i
}

// ReflectionPad2d reflects the input across its borders.
type ReflectionPad2d struct {
	Base
	Top, Bottom, Left, Right int
}

// NewReflectionPad2d constructs a ReflectionPad2d.
func NewReflectionPad2d(top, bottom, left, right int) *ReflectionPad2d {
	return &ReflectionPad2d{Top: top, Bottom: bottom, Left: left, Right: right}
}

// Forward applies reflection padding.
func (p *ReflectionPad2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return pad2d("ReflectionPad2d", x, p.Top, p.Bottom, p.Left, p.Right, reflectIndex)
}

// ReplicationPad2d replicates the edge values.
type ReplicationPad2d struct {
	Base
	Top, Bottom, Left, Right int
}

// NewReplicationPad2d constructs a ReplicationPad2d.
func NewReplicationPad2d(top, bottom, left, right int) *ReplicationPad2d {
	return &ReplicationPad2d{Top: top, Bottom: bottom, Left: left, Right: right}
}

// Forward applies replication (edge) padding.
func (p *ReplicationPad2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return pad2d("ReplicationPad2d", x, p.Top, p.Bottom, p.Left, p.Right, replicateIndex)
}
