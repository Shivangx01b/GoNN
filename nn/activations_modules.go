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
