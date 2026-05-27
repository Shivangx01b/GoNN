package nn

import "gonn/tensor"

// buildPad2dGather returns a (H*W, outH*outW) matrix that places each input
// spatial cell into the output at position determined by indexMap[outIdx],
// where indexMap[k] >= 0 is the input flat index to read, or < 0 means zero.
//
// Returning a (outH*outW, H*W) gather lets us compute out = x @ G^T, similar
// to im2col. Caller provides indexMap of length outH*outW.
func buildPad2dGather(H, W, outH, outW int, indexMap []int) *tensor.Tensor {
	rows := outH * outW
	cols := H * W
	gData := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		src := indexMap[r]
		if src >= 0 && src < cols {
			gData[r*cols+src] = 1
		}
	}
	return tensor.New(gData, rows, cols)
}

// applyPad2d applies a 2D padding selector to an (N, C, H, W) input.
// indexMap describes where each output cell reads from in input flat indices,
// or -1 for "zero fill". An optional fillTensor (shape (N,C,outH,outW)) is
// added after the gather (used by Constant/Reflection/Replication to inject
// non-zero borders that depend on input via separate selector).
func applyPad2d(x *tensor.Tensor, outH, outW int, indexMap []int) *tensor.Tensor {
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	g := buildPad2dGather(H, W, outH, outW, indexMap)
	xFlat := x.Reshape(N*C, H*W)
	out := xFlat.MatMul(g.Transpose()) // (N*C, outH*outW)
	return out.Reshape(N, C, outH, outW)
}

// ZeroPad2d pads (N, C, H, W) with zeros on the four sides.
type ZeroPad2d struct{ Top, Bottom, Left, Right int }

// NewZeroPad2d constructs a ZeroPad2d.
func NewZeroPad2d(top, bottom, left, right int) *ZeroPad2d {
	return &ZeroPad2d{Top: top, Bottom: bottom, Left: left, Right: right}
}

// Forward applies zero padding.
func (p *ZeroPad2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("ZeroPad2d: expected 4D input")
	}
	H, W := x.Shape[2], x.Shape[3]
	outH := H + p.Top + p.Bottom
	outW := W + p.Left + p.Right
	indexMap := make([]int, outH*outW)
	for i := range indexMap {
		indexMap[i] = -1
	}
	for oh := 0; oh < outH; oh++ {
		ih := oh - p.Top
		if ih < 0 || ih >= H {
			continue
		}
		for ow := 0; ow < outW; ow++ {
			iw := ow - p.Left
			if iw < 0 || iw >= W {
				continue
			}
			indexMap[oh*outW+ow] = ih*W + iw
		}
	}
	return applyPad2d(x, outH, outW, indexMap)
}

// Parameters returns nothing.
func (p *ZeroPad2d) Parameters() []*tensor.Tensor { return nil }

// ConstantPad2d pads with a fixed scalar Value.
type ConstantPad2d struct {
	Top, Bottom, Left, Right int
	Value                    float64
}

// NewConstantPad2d constructs a ConstantPad2d.
func NewConstantPad2d(top, bottom, left, right int, value float64) *ConstantPad2d {
	return &ConstantPad2d{Top: top, Bottom: bottom, Left: left, Right: right, Value: value}
}

// Forward applies constant padding.
func (p *ConstantPad2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("ConstantPad2d: expected 4D input")
	}
	N, C, H, W := x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3]
	outH := H + p.Top + p.Bottom
	outW := W + p.Left + p.Right
	indexMap := make([]int, outH*outW)
	// fillMask[r] = 1 means this is a border cell to be set to Value.
	fillMask := make([]float64, outH*outW)
	for i := range indexMap {
		indexMap[i] = -1
	}
	for oh := 0; oh < outH; oh++ {
		ih := oh - p.Top
		for ow := 0; ow < outW; ow++ {
			iw := ow - p.Left
			if ih >= 0 && ih < H && iw >= 0 && iw < W {
				indexMap[oh*outW+ow] = ih*W + iw
			} else {
				fillMask[oh*outW+ow] = p.Value
			}
		}
	}
	gathered := applyPad2d(x, outH, outW, indexMap) // (N, C, outH, outW)
	// Add border value tensor (broadcast over N, C).
	fill := tensor.New(fillMask, 1, 1, outH, outW)
	_ = N
	_ = C
	return gathered.Add(fill)
}

// Parameters returns nothing.
func (p *ConstantPad2d) Parameters() []*tensor.Tensor { return nil }

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
type ReflectionPad2d struct{ Top, Bottom, Left, Right int }

// NewReflectionPad2d constructs a ReflectionPad2d.
func NewReflectionPad2d(top, bottom, left, right int) *ReflectionPad2d {
	return &ReflectionPad2d{Top: top, Bottom: bottom, Left: left, Right: right}
}

// Forward applies reflection padding.
func (p *ReflectionPad2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("ReflectionPad2d: expected 4D input")
	}
	H, W := x.Shape[2], x.Shape[3]
	outH := H + p.Top + p.Bottom
	outW := W + p.Left + p.Right
	indexMap := make([]int, outH*outW)
	for oh := 0; oh < outH; oh++ {
		ih := reflectIndex(oh-p.Top, H)
		for ow := 0; ow < outW; ow++ {
			iw := reflectIndex(ow-p.Left, W)
			indexMap[oh*outW+ow] = ih*W + iw
		}
	}
	return applyPad2d(x, outH, outW, indexMap)
}

// Parameters returns nothing.
func (p *ReflectionPad2d) Parameters() []*tensor.Tensor { return nil }

// ReplicationPad2d replicates the edge values.
type ReplicationPad2d struct{ Top, Bottom, Left, Right int }

// NewReplicationPad2d constructs a ReplicationPad2d.
func NewReplicationPad2d(top, bottom, left, right int) *ReplicationPad2d {
	return &ReplicationPad2d{Top: top, Bottom: bottom, Left: left, Right: right}
}

// Forward applies replication (edge) padding.
func (p *ReplicationPad2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 4 {
		panic("ReplicationPad2d: expected 4D input")
	}
	H, W := x.Shape[2], x.Shape[3]
	outH := H + p.Top + p.Bottom
	outW := W + p.Left + p.Right
	indexMap := make([]int, outH*outW)
	for oh := 0; oh < outH; oh++ {
		ih := replicateIndex(oh-p.Top, H)
		for ow := 0; ow < outW; ow++ {
			iw := replicateIndex(ow-p.Left, W)
			indexMap[oh*outW+ow] = ih*W + iw
		}
	}
	return applyPad2d(x, outH, outW, indexMap)
}

// Parameters returns nothing.
func (p *ReplicationPad2d) Parameters() []*tensor.Tensor { return nil }
