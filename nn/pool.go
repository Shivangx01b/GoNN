package nn

import (
	"fmt"

	"gonn/tensor"
)

// All fixed-window pooling layers share one core, poolNd, built on the same
// gatherMatrix as the conv layers. The default geometry (pad 0, dilation 1,
// floor output sizes) takes the historical path: gather every window, then
// MaxAxis/MeanAxis over the window axis. Padding, ceil_mode and dilation
// (opt-in, PyTorch semantics) route through a precomputed poolPlan that
// physically pads the input first — zeros for average pooling, the
// maxPoolPadSentinel for max pooling — and gathers windows from the padded
// tensor, so everything stays composed from differentiable tensor ops.

type poolMode int

const (
	maxPool poolMode = iota
	avgPool
)

// maxPoolPadSentinel fills padding (and ceil-mode overhang) cells for max
// pooling. PyTorch treats those cells as -inf, but the windows here are
// extracted with a 0/1 gather GEMM and 0 * (-Inf) would produce NaN and
// poison unrelated windows, so a huge negative FINITE value is used instead:
// 0 * (-1e300) = 0 stays exact, and any real input value > -1e300 wins
// against the sentinel. Inputs at or below -1e300 are outside the supported
// range. Because a sentinel cell never wins, gradients flow only through
// real input cells.
const maxPoolPadSentinel = -1e300

// PoolOpt configures a pooling layer.
type PoolOpt func(*poolOpts)

type poolOpts struct {
	kernel          []int
	stride          []int
	pad             []int
	dilation        []int
	ceil            bool
	countIncludePad bool
	ciSet           bool
}

// WithPoolKernel overrides the symmetric kernel with per-dimension sizes.
func WithPoolKernel(k ...int) PoolOpt { return func(o *poolOpts) { o.kernel = k } }

// WithPoolStride sets the stride (one value broadcasts, or one per dim).
// Default: the kernel size (non-overlapping windows).
func WithPoolStride(s ...int) PoolOpt { return func(o *poolOpts) { o.stride = s } }

// WithPoolPadding adds implicit padding on both sides of each spatial dim
// (PyTorch padding): zero cells for average pooling, -inf-like cells for max
// pooling (see maxPoolPadSentinel), so pads are never selected as maxima and
// gradients flow only through real cells. One value broadcasts, or one per
// dim. Each pad must be at most half the effective kernel size
// dilation*(kernel-1)+1, as in PyTorch. Default 0.
func WithPoolPadding(p ...int) PoolOpt { return func(o *poolOpts) { o.pad = p } }

// WithPoolCeilMode computes output sizes with ceil instead of floor:
// out = ceil((in + 2*pad - dilation*(kernel-1) - 1)/stride) + 1, with
// PyTorch's rule that the last window must start within the input or the
// left padding — out is reduced by 1 per dim where (out-1)*stride >= in+pad.
// Windows that extend past the padded end are clipped: max pooling ignores
// the missing cells (sentinel), average pooling counts only up to the padded
// extent in its divisor. Default floor.
func WithPoolCeilMode() PoolOpt { return func(o *poolOpts) { o.ceil = true } }

// WithPoolDilation sets the window dilation (one value broadcasts, or one
// per dim). Max pooling only — average pooling has no dilation in PyTorch
// and panics here too. Default 1.
func WithPoolDilation(d ...int) PoolOpt { return func(o *poolOpts) { o.dilation = d } }

// WithCountIncludePad(false) makes average pooling divide each window by the
// number of real (non-pad) cells it covers instead of the full window
// extent. Average pooling only. Default true (the PyTorch default), which
// divides by the window size (clipped to the padded extent in ceil mode —
// see poolDivisors).
func WithCountIncludePad(include bool) PoolOpt {
	return func(o *poolOpts) { o.countIncludePad = include; o.ciSet = true }
}

// poolNd is the shared pooling core.
type poolNd struct {
	Base
	Kernel, Stride, Pad, Dilation []int
	CeilMode                      bool
	CountIncludePad               bool // average pooling only
	mode                          poolMode

	cacheKey                    []int
	cachedG                     *tensor.Tensor
	cachedOut                   []int
	cachedNumWin, cachedWinSize int

	plan *poolPlan // cache for the padded/ceil-mode/dilated path
}

func (p *poolNd) initPool(rank, kernel int, mode poolMode, opts []PoolOpt) {
	o := poolOpts{countIncludePad: true}
	for _, fn := range opts {
		fn(&o)
	}
	p.Kernel = broadcastDims("kernel", o.kernel, rank, kernel)
	if len(o.stride) == 0 {
		p.Stride = append([]int(nil), p.Kernel...) // default stride = kernel
	} else {
		p.Stride = broadcastDims("stride", o.stride, rank, kernel)
	}
	p.Pad = broadcastDims("padding", o.pad, rank, 0)
	p.Dilation = broadcastDims("dilation", o.dilation, rank, 1)
	for d := 0; d < rank; d++ {
		if p.Kernel[d] < 1 || p.Stride[d] < 1 || p.Pad[d] < 0 || p.Dilation[d] < 1 {
			panic(fmt.Sprintf("nn: invalid pool geometry kernel=%v stride=%v padding=%v dilation=%v",
				p.Kernel, p.Stride, p.Pad, p.Dilation))
		}
		eff := p.Dilation[d]*(p.Kernel[d]-1) + 1
		if 2*p.Pad[d] > eff {
			panic(fmt.Sprintf("nn: pool padding %v should be at most half of effective kernel size %d (dim %d)",
				p.Pad, eff, d))
		}
	}
	if mode == avgPool {
		for d := 0; d < rank; d++ {
			if p.Dilation[d] != 1 {
				panic("nn: average pooling does not support dilation (PyTorch AvgPool has none)")
			}
		}
	} else if o.ciSet {
		panic("nn: WithCountIncludePad only applies to average pooling")
	}
	p.CeilMode = o.ceil
	p.CountIncludePad = o.countIncludePad
	p.mode = mode
}

// needsPlan reports whether the layer uses the padded/ceil-mode/dilated path.
// False means the historical default geometry (bit-identical numerics).
func (p *poolNd) needsPlan() bool {
	if p.CeilMode {
		return true
	}
	for d := range p.Kernel {
		if p.Pad[d] != 0 || p.Dilation[d] != 1 {
			return true
		}
	}
	return false
}

// gatherFor returns the (cached) default-geometry window gather. Only the
// default path (and LPPool, which shares this core) uses it; layers with
// padding/ceil_mode/dilation go through planFor instead.
func (p *poolNd) gatherFor(spatial []int) (*tensor.Tensor, []int, int, int) {
	if p.needsPlan() {
		panic("nn: this pooling layer does not support padding/ceil_mode/dilation options")
	}
	if p.cachedG != nil && intsEqual(p.cacheKey, spatial) {
		return p.cachedG, p.cachedOut, p.cachedNumWin, p.cachedWinSize
	}
	ones := make([]int, len(spatial))
	zeros := make([]int, len(spatial))
	for d := range ones {
		ones[d] = 1
	}
	spec := slidingSpec{
		In: append([]int(nil), spatial...), Kernel: p.Kernel,
		Stride: p.Stride, Pad: zeros, Dilation: ones,
	}
	g, out, numWin, winSize := gatherMatrix(spec)
	p.cacheKey = spec.In
	p.cachedG, p.cachedOut = g, out
	p.cachedNumWin, p.cachedWinSize = numWin, winSize
	return g, out, numWin, winSize
}

// poolPlan precomputes everything the padded/ceil-mode/dilated path needs
// for one input spatial size: the physical zero-pad gather, the sentinel
// border (max pooling), the window gather over the padded size, and the
// per-window divisors (average pooling). All members are constant leaves, so
// the plan is safe to reuse across forwards.
type poolPlan struct {
	key     []int          // unpadded input spatial size
	padOut  []int          // physically padded spatial size (pad + ceil overhang)
	padG    *tensor.Tensor // zero-pad gather (nil when padOut == key)
	border  *tensor.Tensor // (1,1,padOut...) sentinel constant; max pooling only
	g       *tensor.Tensor // window gather over the padded input
	out     []int          // output spatial size
	numWin  int
	winSize int
	div     *tensor.Tensor // (1, numWin) per-window divisors; average pooling only
}

// planFor builds (and caches) the poolPlan for the given input spatial size.
//
// Output sizes follow PyTorch's pooling_output_shape: floor by default, with
// ceil_mode out = ceil((in + 2*pad - dilation*(kernel-1) - 1)/stride) + 1
// reduced by 1 where the last window would start past in+pad (i.e. inside
// the right padding). The input is then physically padded to
// in + 2*pad + overhang per dim, where overhang covers any ceil-mode window
// that extends past the padded end, so every gather tap is in range: max
// pooling fills all pad/overhang cells with maxPoolPadSentinel, average
// pooling with zeros (they contribute nothing to window sums; the divisors
// handle the counting).
func (p *poolNd) planFor(in []int) *poolPlan {
	if p.plan != nil && intsEqual(p.plan.key, in) {
		return p.plan
	}
	rank := len(in)
	out := make([]int, rank)
	after := make([]int, rank)
	padOut := make([]int, rank)
	for d := 0; d < rank; d++ {
		eff := p.Dilation[d]*(p.Kernel[d]-1) + 1
		num := in[d] + 2*p.Pad[d] - eff
		if num < 0 {
			panic(fmt.Sprintf("nn: pool window (kernel=%v dilation=%v) larger than padded input %v (pad=%v) at dim %d",
				p.Kernel, p.Dilation, in, p.Pad, d))
		}
		if p.CeilMode {
			out[d] = (num+p.Stride[d]-1)/p.Stride[d] + 1
			// PyTorch: the last window must start inside the input or the
			// left padding, never inside the right padding.
			if (out[d]-1)*p.Stride[d] >= in[d]+p.Pad[d] {
				out[d]--
			}
		} else {
			out[d] = num/p.Stride[d] + 1
		}
		after[d] = p.Pad[d]
		if over := (out[d]-1)*p.Stride[d] + eff - (in[d] + 2*p.Pad[d]); over > 0 {
			after[d] += over // ceil-mode overhang past the padded end
		}
		padOut[d] = in[d] + p.Pad[d] + after[d]
	}
	plan := &poolPlan{key: append([]int(nil), in...), padOut: padOut, out: out}
	if !intsEqual(padOut, in) {
		_, im := padND(in, p.Pad, after, zeroPadIndex)
		plan.padG = indexMapGather(im, prodInts(in))
		if p.mode == maxPool {
			plan.border = constantFill(padOut, im, maxPoolPadSentinel)
		}
	}
	zeros := make([]int, rank)
	spec := slidingSpec{
		In: padOut, Kernel: p.Kernel,
		Stride: p.Stride, Pad: zeros, Dilation: p.Dilation,
	}
	g, gOut, numWin, winSize := gatherMatrix(spec)
	if !intsEqual(gOut, out) {
		panic(fmt.Sprintf("nn: internal: pooling plan output %v != gather output %v", out, gOut))
	}
	plan.g, plan.numWin, plan.winSize = g, numWin, winSize
	if p.mode == avgPool {
		plan.div = p.poolDivisors(in, out, numWin)
	}
	p.plan = plan
	return plan
}

// poolDivisors builds the (1, numWin) per-window divisor for average pooling
// with padding/ceil_mode, matching PyTorch's AvgPool kernels exactly:
//
//   - CountIncludePad (default): divisor = the window extent clipped to the
//     PADDED input, prod_d(min(start_d + kernel_d, in_d + 2*pad_d) - start_d)
//     with start_d = outIdx_d*stride_d in padded coordinates. Only ceil-mode
//     windows that overhang the padded end are clipped; otherwise this is the
//     full kernel size.
//   - !CountIncludePad: divisor = the number of REAL (non-pad) cells,
//     prod_d(min(start_d + kernel_d, in_d) - max(start_d, 0)) with start_d =
//     outIdx_d*stride_d - pad_d in unpadded coordinates.
//
// Both counts are always >= 1: the ceil-mode clip guarantees every window
// starts before in+pad, and pad <= kernel/2 guarantees it overlaps the input.
func (p *poolNd) poolDivisors(in, out []int, numWin int) *tensor.Tensor {
	rank := len(in)
	counts := make([][]float64, rank)
	for d := 0; d < rank; d++ {
		counts[d] = make([]float64, out[d])
		for o := 0; o < out[d]; o++ {
			var c int
			if p.CountIncludePad {
				start := o * p.Stride[d] // padded coordinates
				end := start + p.Kernel[d]
				if lim := in[d] + 2*p.Pad[d]; end > lim {
					end = lim
				}
				c = end - start
			} else {
				start := o*p.Stride[d] - p.Pad[d] // unpadded coordinates
				end := start + p.Kernel[d]
				if start < 0 {
					start = 0
				}
				if end > in[d] {
					end = in[d]
				}
				c = end - start
			}
			counts[d][o] = float64(c)
		}
	}
	div := make([]float64, numWin)
	idx := make([]int, rank)
	for w := 0; w < numWin; w++ {
		v := 1.0
		for d := 0; d < rank; d++ {
			v *= counts[d][idx[d]]
		}
		div[w] = v
		incMultiIndex(idx, out)
	}
	return tensor.New(div, 1, numWin)
}

// gatherWindows pads x per the plan (zero gather + sentinel border for max
// pooling) and gathers every window into (N*C, numWin, winSize). The border
// constant carries no gradient and the pad-gather rows for border cells are
// all-zero, so gradients flow only through real input cells.
func (p *poolNd) gatherWindows(x *tensor.Tensor, plan *poolPlan) *tensor.Tensor {
	N, C := x.Shape[0], x.Shape[1]
	padded := x
	if plan.padG != nil {
		padded = x.Reshape(N*C, prodInts(x.Shape[2:])).
			MatMul(plan.padG.Transpose()).
			Reshape(append([]int{N, C}, plan.padOut...)...)
		if plan.border != nil {
			padded = padded.Add(plan.border)
		}
	}
	return padded.Reshape(N*C, prodInts(plan.padOut)).
		MatMul(plan.g.Transpose()).
		Reshape(N*C, plan.numWin, plan.winSize)
}

// Forward gathers windows and reduces each with max or mean.
func (p *poolNd) Forward(x *tensor.Tensor) *tensor.Tensor {
	rank := len(p.Kernel)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: pool expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	N, C := x.Shape[0], x.Shape[1]
	if !p.needsPlan() {
		g, out, numWin, winSize := p.gatherFor(x.Shape[2:])

		win := x.Reshape(N*C, prodInts(x.Shape[2:])).
			MatMul(g.Transpose()).
			Reshape(N*C, numWin, winSize)
		var r *tensor.Tensor
		if p.mode == maxPool {
			r = win.MaxAxis(2, false)
		} else {
			r = win.MeanAxis(2, false)
		}
		return r.Reshape(append([]int{N, C}, out...)...)
	}

	plan := p.planFor(x.Shape[2:])
	win := p.gatherWindows(x, plan)
	var r *tensor.Tensor
	if p.mode == maxPool {
		r = win.MaxAxis(2, false)
	} else {
		r = win.SumAxis(2, false).Div(plan.div)
	}
	return r.Reshape(append([]int{N, C}, plan.out...)...)
}

// forwardWithIndices is Forward plus, for max pooling, an indices tensor of
// the same shape as the output holding the FLAT SPATIAL index of each selected
// maximum within its (n, c) plane — PyTorch's return_indices convention (e.g.
// for a (N, C, H, W) input the index is h*W + w). Indices always refer to the
// UNPADDED input: padding cells hold the sentinel and are never selected. The
// pooled output is computed by the exact same ops as Forward, so numerics are
// identical; ties resolve to the first (lowest-offset) maximum in both paths.
// Indices carry no gradient.
func (p *poolNd) forwardWithIndices(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	if p.mode != maxPool {
		panic("nn: forwardWithIndices is only defined for max pooling")
	}
	rank := len(p.Kernel)
	if len(x.Shape) != rank+2 {
		panic(fmt.Sprintf("nn: pool expected %dD input (N,C,spatial...), got shape %v", rank+2, x.Shape))
	}
	N, C := x.Shape[0], x.Shape[1]

	var win *tensor.Tensor
	var out []int
	var numWin, winSize int
	if !p.needsPlan() {
		var g *tensor.Tensor
		g, out, numWin, winSize = p.gatherFor(x.Shape[2:])
		win = x.Reshape(N*C, prodInts(x.Shape[2:])).
			MatMul(g.Transpose()).
			Reshape(N*C, numWin, winSize)
	} else {
		plan := p.planFor(x.Shape[2:])
		out, numWin, winSize = plan.out, plan.numWin, plan.winSize
		win = p.gatherWindows(x, plan)
	}
	r := win.MaxAxis(2, false)

	// Window-relative argmax, then translate each kernel offset back to the
	// UNPADDED input position it reads: pos[d] = winIdx[d]*stride[d] +
	// kIdx[d]*dilation[d] - pad[d]. Offsets that land in the padding map to
	// -1; they hold the sentinel and are never the maximum.
	arg := win.ArgMax(2) // (N*C, numWin)
	inStrides := rowMajorStrides(x.Shape[2:])
	lut := make([]int, numWin*winSize)
	winIdx := make([]int, rank)
	for w := 0; w < numWin; w++ {
		kIdx := make([]int, rank)
		for kk := 0; kk < winSize; kk++ {
			flat := 0
			for d := 0; d < rank; d++ {
				pos := winIdx[d]*p.Stride[d] + kIdx[d]*p.Dilation[d] - p.Pad[d]
				if pos < 0 || pos >= x.Shape[2+d] {
					flat = -1
					break
				}
				flat += pos * inStrides[d]
			}
			lut[w*winSize+kk] = flat
			incMultiIndex(kIdx, p.Kernel)
		}
		incMultiIndex(winIdx, out)
	}
	idx := tensor.Zeros(append([]int{N, C}, out...)...)
	for i := 0; i < N*C; i++ {
		for w := 0; w < numWin; w++ {
			idx.Data[i*numWin+w] = float64(lut[w*winSize+int(arg.Data[i*numWin+w])])
		}
	}
	return r.Reshape(append([]int{N, C}, out...)...), idx
}

// ---- Public pooling layers --------------------------------------------------

// MaxPool1d performs 1D max pooling on (N, C, L) inputs.
type MaxPool1d struct{ poolNd }

// NewMaxPool1d creates a MaxPool1d; stride defaults to the kernel size.
// Configure with WithPoolStride/WithPoolKernel/WithPoolPadding/
// WithPoolDilation/WithPoolCeilMode.
func NewMaxPool1d(kernel int, opts ...PoolOpt) *MaxPool1d {
	p := &MaxPool1d{}
	p.initPool(1, kernel, maxPool, opts)
	return p
}

// ForwardWithIndices returns the pooled output plus the flat spatial index
// (within each (n, c) plane of the unpadded input) of every selected maximum,
// for use with MaxUnpool1d. The output matches Forward exactly.
func (p *MaxPool1d) ForwardWithIndices(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	return p.forwardWithIndices(x)
}

// AvgPool1d performs 1D average pooling on (N, C, L) inputs.
type AvgPool1d struct{ poolNd }

// NewAvgPool1d creates an AvgPool1d; stride defaults to the kernel size.
// Configure with WithPoolStride/WithPoolKernel/WithPoolPadding/
// WithPoolCeilMode/WithCountIncludePad.
func NewAvgPool1d(kernel int, opts ...PoolOpt) *AvgPool1d {
	p := &AvgPool1d{}
	p.initPool(1, kernel, avgPool, opts)
	return p
}

// MaxPool2d performs 2D max pooling on (N, C, H, W) inputs.
type MaxPool2d struct{ poolNd }

// NewMaxPool2d creates a MaxPool2d; stride defaults to the kernel size.
// Configure with WithPoolStride/WithPoolKernel/WithPoolPadding/
// WithPoolDilation/WithPoolCeilMode.
func NewMaxPool2d(kernel int, opts ...PoolOpt) *MaxPool2d {
	p := &MaxPool2d{}
	p.initPool(2, kernel, maxPool, opts)
	return p
}

// ForwardWithIndices returns the pooled output plus the flat spatial index
// (h*W + w within each (n, c) plane of the unpadded input) of every selected
// maximum, for use with MaxUnpool2d. The output matches Forward exactly.
func (p *MaxPool2d) ForwardWithIndices(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	return p.forwardWithIndices(x)
}

// AvgPool2d performs 2D average pooling on (N, C, H, W) inputs.
type AvgPool2d struct{ poolNd }

// NewAvgPool2d creates an AvgPool2d; stride defaults to the kernel size.
// Configure with WithPoolStride/WithPoolKernel/WithPoolPadding/
// WithPoolCeilMode/WithCountIncludePad.
func NewAvgPool2d(kernel int, opts ...PoolOpt) *AvgPool2d {
	p := &AvgPool2d{}
	p.initPool(2, kernel, avgPool, opts)
	return p
}

// MaxPool3d performs 3D max pooling on (N, C, D, H, W) inputs.
type MaxPool3d struct{ poolNd }

// NewMaxPool3d creates a MaxPool3d; stride defaults to the kernel size.
// Configure with WithPoolStride/WithPoolKernel/WithPoolPadding/
// WithPoolDilation/WithPoolCeilMode.
func NewMaxPool3d(kernel int, opts ...PoolOpt) *MaxPool3d {
	p := &MaxPool3d{}
	p.initPool(3, kernel, maxPool, opts)
	return p
}

// ForwardWithIndices returns the pooled output plus the flat spatial index
// ((d*H + h)*W + w within each (n, c) plane of the unpadded input) of every
// selected maximum, for use with MaxUnpool3d. The output matches Forward
// exactly.
func (p *MaxPool3d) ForwardWithIndices(x *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	return p.forwardWithIndices(x)
}

// AvgPool3d performs 3D average pooling on (N, C, D, H, W) inputs.
type AvgPool3d struct{ poolNd }

// NewAvgPool3d creates an AvgPool3d; stride defaults to the kernel size.
// Configure with WithPoolStride/WithPoolKernel/WithPoolPadding/
// WithPoolCeilMode/WithCountIncludePad.
func NewAvgPool3d(kernel int, opts ...PoolOpt) *AvgPool3d {
	p := &AvgPool3d{}
	p.initPool(3, kernel, avgPool, opts)
	return p
}
