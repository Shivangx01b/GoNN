package nn

// Lazy modules — a Go adaptation of torch.nn.LazyModuleMixin (LazyLinear,
// LazyConvNd, LazyBatchNormNd, ...). A lazy module records only its
// output-side configuration at construction time; the FIRST Forward infers
// the input-dependent dimension (last-dim features for Linear, channel dim
// x.Shape[1] for conv/norm), constructs the real module — which draws its
// weights from the global RNG at that point, exactly as the eager
// constructor would — registers it as child "inner", and delegates every
// subsequent Forward to it.
//
// LOUD WARNING (same as PyTorch's): before the first Forward the module has
// NO parameters — Parameters()/NamedParameters() return empty. Run a dummy
// forward pass through the network BEFORE constructing the optimizer, or the
// optimizer will silently hold zero parameters:
//
//	model := nn.NewSequential(nn.NewLazyLinear(64, true), nn.ReLU(), nn.NewLazyLinear(10, true))
//	model.Forward(sample) // materializes every lazy module
//	opt := optim.NewSGD(model.Parameters(), 0.1)
//
// Deviations from PyTorch: initialization happens inside Forward (there is
// no initialize_parameters/load_state_dict machinery), the materialized
// module is an ordinary registered child named "inner" (so parameter names
// gain an "inner." segment, e.g. "0.inner.weight" inside a Sequential), and
// train/eval mode set before initialization is forwarded to the inner module
// when it is created.

import (
	"fmt"

	"gonn/tensor"
)

// lazyInitializer is satisfied by every lazy module.
type lazyInitializer interface {
	IsInitialized() bool
}

// IsLazy reports whether c is a lazy module (initialized or not) — the GoNN
// analogue of isinstance(m, LazyModuleMixin).
func IsLazy(c Child) bool {
	_, ok := c.(lazyInitializer)
	return ok
}

// lazyState is the shared core every lazy module embeds: the not-yet /
// already materialized inner module plus the Base that will own it.
type lazyState struct {
	Base
	inner Module
}

// IsInitialized reports whether the first Forward has materialized the
// inner module (PyTorch: has_uninitialized_params() == false).
func (l *lazyState) IsInitialized() bool { return l.inner != nil }

// InnerModule returns the materialized module, or nil before the first
// Forward.
func (l *lazyState) InnerModule() Module { return l.inner }

// materialize registers m as child "inner" and syncs train/eval mode (a lazy
// module put in eval mode before its first Forward materializes an inner
// module already in eval mode).
func (l *lazyState) materialize(m Module) Module {
	l.inner = m
	l.regChild("inner", m)
	m.SetTraining(l.Training())
	return m
}

// channelDim returns x.Shape[1], panicking with a helpful message for
// too-low-rank inputs.
func channelDim(what string, x *tensor.Tensor) int {
	if len(x.Shape) < 2 {
		panic(fmt.Sprintf("nn.%s: need at least 2D input (N, C, ...) to infer channels, got shape %v", what, x.Shape))
	}
	return x.Shape[1]
}

// ---- LazyLinear ---------------------------------------------------------

// LazyLinear is a Linear whose InFeatures is inferred from the last dim of
// the first Forward input. See the package comment in this file for the
// optimizer-ordering caveat.
type LazyLinear struct {
	lazyState
	OutFeatures int
	Bias        bool
}

// NewLazyLinear creates a LazyLinear producing out features.
func NewLazyLinear(out int, bias bool) *LazyLinear {
	return &LazyLinear{OutFeatures: out, Bias: bias}
}

// Forward materializes the inner Linear on first call (InFeatures = last dim
// of x) and delegates.
func (l *LazyLinear) Forward(x *tensor.Tensor) *tensor.Tensor {
	if l.inner == nil {
		if len(x.Shape) == 0 {
			panic("nn.LazyLinear: cannot infer InFeatures from a scalar input")
		}
		l.materialize(NewLinear(x.Shape[len(x.Shape)-1], l.OutFeatures, l.Bias))
	}
	return l.inner.Forward(x)
}

// Inner returns the materialized Linear, or nil before the first Forward.
func (l *LazyLinear) Inner() *Linear {
	if l.inner == nil {
		return nil
	}
	return l.inner.(*Linear)
}

// ---- Lazy convolutions --------------------------------------------------

// lazyConv is the shared config for the six lazy conv variants.
type lazyConv struct {
	lazyState
	OutChannels int
	Kernel      int
	opts        []ConvOpt
}

func (l *lazyConv) forward(x *tensor.Tensor, what string, build func(inC int) Module) *tensor.Tensor {
	if l.inner == nil {
		l.materialize(build(channelDim(what, x)))
	}
	return l.inner.Forward(x)
}

// LazyConv1d is a Conv1d whose input channel count is inferred from
// x.Shape[1] on the first Forward.
type LazyConv1d struct{ lazyConv }

// NewLazyConv1d creates a LazyConv1d; opts are forwarded to NewConv1d at
// materialization time.
func NewLazyConv1d(outC, kernel int, opts ...ConvOpt) *LazyConv1d {
	return &LazyConv1d{lazyConv{OutChannels: outC, Kernel: kernel, opts: opts}}
}

// Forward materializes the inner Conv1d on first call and delegates.
func (l *LazyConv1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyConv1d", func(inC int) Module { return NewConv1d(inC, l.OutChannels, l.Kernel, l.opts...) })
}

// LazyConv2d is a Conv2d whose input channel count is inferred from
// x.Shape[1] on the first Forward.
type LazyConv2d struct{ lazyConv }

// NewLazyConv2d creates a LazyConv2d; opts are forwarded to NewConv2d at
// materialization time.
func NewLazyConv2d(outC, kernel int, opts ...ConvOpt) *LazyConv2d {
	return &LazyConv2d{lazyConv{OutChannels: outC, Kernel: kernel, opts: opts}}
}

// Forward materializes the inner Conv2d on first call and delegates.
func (l *LazyConv2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyConv2d", func(inC int) Module { return NewConv2d(inC, l.OutChannels, l.Kernel, l.opts...) })
}

// LazyConv3d is a Conv3d whose input channel count is inferred from
// x.Shape[1] on the first Forward.
type LazyConv3d struct{ lazyConv }

// NewLazyConv3d creates a LazyConv3d; opts are forwarded to NewConv3d at
// materialization time.
func NewLazyConv3d(outC, kernel int, opts ...ConvOpt) *LazyConv3d {
	return &LazyConv3d{lazyConv{OutChannels: outC, Kernel: kernel, opts: opts}}
}

// Forward materializes the inner Conv3d on first call and delegates.
func (l *LazyConv3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyConv3d", func(inC int) Module { return NewConv3d(inC, l.OutChannels, l.Kernel, l.opts...) })
}

// LazyConvTranspose1d is a ConvTranspose1d whose input channel count is
// inferred from x.Shape[1] on the first Forward.
type LazyConvTranspose1d struct{ lazyConv }

// NewLazyConvTranspose1d creates a LazyConvTranspose1d; opts are forwarded
// to NewConvTranspose1d at materialization time.
func NewLazyConvTranspose1d(outC, kernel int, opts ...ConvOpt) *LazyConvTranspose1d {
	return &LazyConvTranspose1d{lazyConv{OutChannels: outC, Kernel: kernel, opts: opts}}
}

// Forward materializes the inner ConvTranspose1d on first call and delegates.
func (l *LazyConvTranspose1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyConvTranspose1d", func(inC int) Module {
		return NewConvTranspose1d(inC, l.OutChannels, l.Kernel, l.opts...)
	})
}

// LazyConvTranspose2d is a ConvTranspose2d whose input channel count is
// inferred from x.Shape[1] on the first Forward.
type LazyConvTranspose2d struct{ lazyConv }

// NewLazyConvTranspose2d creates a LazyConvTranspose2d; opts are forwarded
// to NewConvTranspose2d at materialization time.
func NewLazyConvTranspose2d(outC, kernel int, opts ...ConvOpt) *LazyConvTranspose2d {
	return &LazyConvTranspose2d{lazyConv{OutChannels: outC, Kernel: kernel, opts: opts}}
}

// Forward materializes the inner ConvTranspose2d on first call and delegates.
func (l *LazyConvTranspose2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyConvTranspose2d", func(inC int) Module {
		return NewConvTranspose2d(inC, l.OutChannels, l.Kernel, l.opts...)
	})
}

// LazyConvTranspose3d is a ConvTranspose3d whose input channel count is
// inferred from x.Shape[1] on the first Forward.
type LazyConvTranspose3d struct{ lazyConv }

// NewLazyConvTranspose3d creates a LazyConvTranspose3d; opts are forwarded
// to NewConvTranspose3d at materialization time.
func NewLazyConvTranspose3d(outC, kernel int, opts ...ConvOpt) *LazyConvTranspose3d {
	return &LazyConvTranspose3d{lazyConv{OutChannels: outC, Kernel: kernel, opts: opts}}
}

// Forward materializes the inner ConvTranspose3d on first call and delegates.
func (l *LazyConvTranspose3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyConvTranspose3d", func(inC int) Module {
		return NewConvTranspose3d(inC, l.OutChannels, l.Kernel, l.opts...)
	})
}

// ---- Lazy normalization -------------------------------------------------

// lazyNorm is the shared config for the lazy norm variants.
type lazyNorm struct {
	lazyState
	opts []NormOpt
}

func (l *lazyNorm) forward(x *tensor.Tensor, what string, build func(c int) Module) *tensor.Tensor {
	if l.inner == nil {
		l.materialize(build(channelDim(what, x)))
	}
	return l.inner.Forward(x)
}

// LazyBatchNorm1d is a BatchNorm1d whose feature count is inferred from
// x.Shape[1] on the first Forward ((N, C) and (N, C, L) inputs both put C at
// index 1).
type LazyBatchNorm1d struct{ lazyNorm }

// NewLazyBatchNorm1d creates a LazyBatchNorm1d; opts are forwarded to
// NewBatchNorm1d at materialization time.
func NewLazyBatchNorm1d(opts ...NormOpt) *LazyBatchNorm1d {
	return &LazyBatchNorm1d{lazyNorm{opts: opts}}
}

// Forward materializes the inner BatchNorm1d on first call and delegates.
func (l *LazyBatchNorm1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyBatchNorm1d", func(c int) Module { return NewBatchNorm1d(c, l.opts...) })
}

// LazyBatchNorm2d is a BatchNorm2d whose feature count is inferred from
// x.Shape[1] on the first Forward.
type LazyBatchNorm2d struct{ lazyNorm }

// NewLazyBatchNorm2d creates a LazyBatchNorm2d; opts are forwarded to
// NewBatchNorm2d at materialization time.
func NewLazyBatchNorm2d(opts ...NormOpt) *LazyBatchNorm2d {
	return &LazyBatchNorm2d{lazyNorm{opts: opts}}
}

// Forward materializes the inner BatchNorm2d on first call and delegates.
func (l *LazyBatchNorm2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyBatchNorm2d", func(c int) Module { return NewBatchNorm2d(c, l.opts...) })
}

// batchNorm3d is the 5D batch norm backing LazyBatchNorm3d. GoNN has no
// exported eager BatchNorm3d yet, so the lazy variant wraps the shared
// batchNormNd core directly (same statistics path BatchNorm1d/2d use for
// rank >= 3 inputs).
type batchNorm3d struct{ batchNormNd }

// Forward applies BN over (N, D, H, W) per channel.
func (b *batchNorm3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	if len(x.Shape) != 5 {
		panic(fmt.Sprintf("BatchNorm3d: expected 5D input, got shape %v", x.Shape))
	}
	return b.forwardChannelsFirst(x)
}

// LazyBatchNorm3d is a 5D batch norm ((N, C, D, H, W) inputs) whose feature
// count is inferred from x.Shape[1] on the first Forward.
type LazyBatchNorm3d struct{ lazyNorm }

// NewLazyBatchNorm3d creates a LazyBatchNorm3d; opts configure the inner
// norm at materialization time.
func NewLazyBatchNorm3d(opts ...NormOpt) *LazyBatchNorm3d {
	return &LazyBatchNorm3d{lazyNorm{opts: opts}}
}

// Forward materializes the inner 5D batch norm on first call and delegates.
func (l *LazyBatchNorm3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyBatchNorm3d", func(c int) Module {
		return &batchNorm3d{newBatchNorm(c, l.opts)}
	})
}

// LazyInstanceNorm1d is an InstanceNorm1d whose feature count is inferred
// from x.Shape[1] on the first Forward.
type LazyInstanceNorm1d struct{ lazyNorm }

// NewLazyInstanceNorm1d creates a LazyInstanceNorm1d; opts are forwarded to
// NewInstanceNorm1d at materialization time.
func NewLazyInstanceNorm1d(opts ...NormOpt) *LazyInstanceNorm1d {
	return &LazyInstanceNorm1d{lazyNorm{opts: opts}}
}

// Forward materializes the inner InstanceNorm1d on first call and delegates.
func (l *LazyInstanceNorm1d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyInstanceNorm1d", func(c int) Module { return NewInstanceNorm1d(c, l.opts...) })
}

// LazyInstanceNorm2d is an InstanceNorm2d whose feature count is inferred
// from x.Shape[1] on the first Forward.
type LazyInstanceNorm2d struct{ lazyNorm }

// NewLazyInstanceNorm2d creates a LazyInstanceNorm2d; opts are forwarded to
// NewInstanceNorm2d at materialization time.
func NewLazyInstanceNorm2d(opts ...NormOpt) *LazyInstanceNorm2d {
	return &LazyInstanceNorm2d{lazyNorm{opts: opts}}
}

// Forward materializes the inner InstanceNorm2d on first call and delegates.
func (l *LazyInstanceNorm2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyInstanceNorm2d", func(c int) Module { return NewInstanceNorm2d(c, l.opts...) })
}

// instanceNorm3d is the 5D instance norm backing LazyInstanceNorm3d. GoNN
// has no exported eager InstanceNorm3d yet, so the lazy variant wraps the
// shared instanceNormNd core with rank 5.
type instanceNorm3d struct{ instanceNormNd }

// LazyInstanceNorm3d is a 5D instance norm ((N, C, D, H, W) inputs) whose
// feature count is inferred from x.Shape[1] on the first Forward. Affine is
// off by default, as with the eager InstanceNorm layers.
type LazyInstanceNorm3d struct{ lazyNorm }

// NewLazyInstanceNorm3d creates a LazyInstanceNorm3d; opts configure the
// inner norm at materialization time.
func NewLazyInstanceNorm3d(opts ...NormOpt) *LazyInstanceNorm3d {
	return &LazyInstanceNorm3d{lazyNorm{opts: opts}}
}

// Forward materializes the inner 5D instance norm on first call and delegates.
func (l *LazyInstanceNorm3d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.forward(x, "LazyInstanceNorm3d", func(c int) Module {
		return &instanceNorm3d{newInstanceNorm(5, c, l.opts)}
	})
}
