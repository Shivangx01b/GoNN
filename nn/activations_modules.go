package nn

import "gonn/tensor"

// ReLU module.
type ReLU struct{}

func (ReLU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.ReLU() }
func (ReLU) Parameters() []*tensor.Tensor            { return nil }

// Sigmoid module.
type Sigmoid struct{}

func (Sigmoid) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Sigmoid() }
func (Sigmoid) Parameters() []*tensor.Tensor            { return nil }

// Tanh module.
type Tanh struct{}

func (Tanh) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Tanh() }
func (Tanh) Parameters() []*tensor.Tensor            { return nil }

// GELU module.
type GELU struct{}

func (GELU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.GELU() }
func (GELU) Parameters() []*tensor.Tensor            { return nil }

// SiLU module.
type SiLU struct{}

func (SiLU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.SiLU() }
func (SiLU) Parameters() []*tensor.Tensor            { return nil }

// Softmax module along Axis.
type Softmax struct{ Axis int }

func (s Softmax) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Softmax(s.Axis) }
func (Softmax) Parameters() []*tensor.Tensor              { return nil }

// LogSoftmax module along Axis.
type LogSoftmax struct{ Axis int }

func (s LogSoftmax) Forward(x *tensor.Tensor) *tensor.Tensor { return x.LogSoftmax(s.Axis) }
func (LogSoftmax) Parameters() []*tensor.Tensor              { return nil }

// LeakyReLU module with slope Alpha.
type LeakyReLU struct{ Alpha float64 }

func (l LeakyReLU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.LeakyReLU(l.Alpha) }
func (LeakyReLU) Parameters() []*tensor.Tensor              { return nil }

// ELU module.
type ELU struct{ Alpha float64 }

func (e ELU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.ELU(e.Alpha) }
func (ELU) Parameters() []*tensor.Tensor              { return nil }

// SELU module.
type SELU struct{}

func (SELU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.SELU() }
func (SELU) Parameters() []*tensor.Tensor            { return nil }

// ReLU6 module.
type ReLU6 struct{}

func (ReLU6) Forward(x *tensor.Tensor) *tensor.Tensor { return x.ReLU6() }
func (ReLU6) Parameters() []*tensor.Tensor            { return nil }

// HardTanh module.
type HardTanh struct{}

func (HardTanh) Forward(x *tensor.Tensor) *tensor.Tensor { return x.HardTanh() }
func (HardTanh) Parameters() []*tensor.Tensor            { return nil }

// HardSigmoid module.
type HardSigmoid struct{}

func (HardSigmoid) Forward(x *tensor.Tensor) *tensor.Tensor { return x.HardSigmoid() }
func (HardSigmoid) Parameters() []*tensor.Tensor            { return nil }

// HardSwish module.
type HardSwish struct{}

func (HardSwish) Forward(x *tensor.Tensor) *tensor.Tensor { return x.HardSwish() }
func (HardSwish) Parameters() []*tensor.Tensor            { return nil }

// Mish module.
type Mish struct{}

func (Mish) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Mish() }
func (Mish) Parameters() []*tensor.Tensor            { return nil }

// Softplus module.
type Softplus struct{}

func (Softplus) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Softplus() }
func (Softplus) Parameters() []*tensor.Tensor            { return nil }

// Softsign module.
type Softsign struct{}

func (Softsign) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Softsign() }
func (Softsign) Parameters() []*tensor.Tensor            { return nil }

// LogSigmoid module.
type LogSigmoid struct{}

func (LogSigmoid) Forward(x *tensor.Tensor) *tensor.Tensor { return x.LogSigmoid() }
func (LogSigmoid) Parameters() []*tensor.Tensor            { return nil }

// Hardshrink module with threshold Lambda.
type Hardshrink struct{ Lambda float64 }

func (h Hardshrink) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Hardshrink(h.Lambda) }
func (Hardshrink) Parameters() []*tensor.Tensor              { return nil }

// Softshrink module with threshold Lambda.
type Softshrink struct{ Lambda float64 }

func (s Softshrink) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Softshrink(s.Lambda) }
func (Softshrink) Parameters() []*tensor.Tensor              { return nil }

// Tanhshrink module.
type Tanhshrink struct{}

func (Tanhshrink) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Tanhshrink() }
func (Tanhshrink) Parameters() []*tensor.Tensor            { return nil }

// Threshold module: x if x > Thresh else Value.
type Threshold struct {
	Thresh float64
	Value  float64
}

func (th Threshold) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Threshold(th.Thresh, th.Value) }
func (Threshold) Parameters() []*tensor.Tensor               { return nil }

// CELU module with parameter Alpha.
type CELU struct{ Alpha float64 }

func (c CELU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.CELU(c.Alpha) }
func (CELU) Parameters() []*tensor.Tensor              { return nil }
