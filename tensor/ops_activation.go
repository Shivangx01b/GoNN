package tensor

import "math"

// Activation functions (forward + autograd backward).

// ReLU returns max(0, t).
func (t *Tensor) ReLU() *Tensor {
	return unaryOp(t, "ReLU",
		func(v float64) float64 {
			if v > 0 {
				return v
			}
			return 0
		},
		func(g, x, y float64) float64 {
			if x > 0 {
				return g
			}
			return 0
		})
}

// LeakyReLU returns x if x>0 else alpha*x.
func (t *Tensor) LeakyReLU(alpha float64) *Tensor {
	return unaryOp(t, "LeakyReLU",
		func(v float64) float64 {
			if v > 0 {
				return v
			}
			return alpha * v
		},
		func(g, x, y float64) float64 {
			if x > 0 {
				return g
			}
			return g * alpha
		})
}

// ELU returns x if x>0 else alpha*(e^x-1).
func (t *Tensor) ELU(alpha float64) *Tensor {
	return unaryOp(t, "ELU",
		func(v float64) float64 {
			if v > 0 {
				return v
			}
			return alpha * (math.Exp(v) - 1)
		},
		func(g, x, y float64) float64 {
			if x > 0 {
				return g
			}
			return g * (y + alpha)
		})
}

// SELU activation.
func (t *Tensor) SELU() *Tensor {
	const lambda = 1.0507009873554805
	const alpha = 1.6732632423543772
	return unaryOp(t, "SELU",
		func(v float64) float64 {
			if v > 0 {
				return lambda * v
			}
			return lambda * alpha * (math.Exp(v) - 1)
		},
		func(g, x, y float64) float64 {
			if x > 0 {
				return g * lambda
			}
			return g * (y + lambda*alpha)
		})
}

// Sigmoid returns 1/(1+e^-t).
func (t *Tensor) Sigmoid() *Tensor {
	return unaryOp(t, "Sigmoid",
		func(v float64) float64 { return 1 / (1 + math.Exp(-v)) },
		func(g, x, y float64) float64 { return g * y * (1 - y) })
}

// Tanh returns tanh(t).
func (t *Tensor) Tanh() *Tensor {
	return unaryOp(t, "Tanh", math.Tanh,
		func(g, x, y float64) float64 { return g * (1 - y*y) })
}

// HardTanh clamps to [-1, 1].
func (t *Tensor) HardTanh() *Tensor {
	return unaryOp(t, "HardTanh",
		func(v float64) float64 {
			if v < -1 {
				return -1
			}
			if v > 1 {
				return 1
			}
			return v
		},
		func(g, x, y float64) float64 {
			if x < -1 || x > 1 {
				return 0
			}
			return g
		})
}

// HardSigmoid is a piecewise linear approx to sigmoid.
func (t *Tensor) HardSigmoid() *Tensor {
	return unaryOp(t, "HardSigmoid",
		func(v float64) float64 {
			if v <= -3 {
				return 0
			}
			if v >= 3 {
				return 1
			}
			return v/6 + 0.5
		},
		func(g, x, y float64) float64 {
			if x <= -3 || x >= 3 {
				return 0
			}
			return g / 6
		})
}

// Softplus returns ln(1+e^t).
func (t *Tensor) Softplus() *Tensor {
	return unaryOp(t, "Softplus",
		func(v float64) float64 {
			if v > 20 {
				return v
			}
			return math.Log(1 + math.Exp(v))
		},
		func(g, x, y float64) float64 { return g / (1 + math.Exp(-x)) })
}

// Softsign returns x/(1+|x|).
func (t *Tensor) Softsign() *Tensor {
	return unaryOp(t, "Softsign",
		func(v float64) float64 { return v / (1 + math.Abs(v)) },
		func(g, x, y float64) float64 {
			d := 1 + math.Abs(x)
			return g / (d * d)
		})
}

// GELU using the tanh approximation (PyTorch default-compatible).
func (t *Tensor) GELU() *Tensor {
	const c = 0.7978845608028654 // sqrt(2/pi)
	const k = 0.044715
	return unaryOp(t, "GELU",
		func(v float64) float64 {
			return 0.5 * v * (1 + math.Tanh(c*(v+k*v*v*v)))
		},
		func(g, x, y float64) float64 {
			inner := c * (x + k*x*x*x)
			tanh := math.Tanh(inner)
			sech2 := 1 - tanh*tanh
			dInner := c * (1 + 3*k*x*x)
			return g * (0.5*(1+tanh) + 0.5*x*sech2*dInner)
		})
}

// SiLU (Swish) returns x * sigmoid(x).
func (t *Tensor) SiLU() *Tensor {
	return unaryOp(t, "SiLU",
		func(v float64) float64 { return v / (1 + math.Exp(-v)) },
		func(g, x, y float64) float64 {
			s := 1 / (1 + math.Exp(-x))
			return g * (s + x*s*(1-s))
		})
}

// Swish is an alias for SiLU.
func (t *Tensor) Swish() *Tensor { return t.SiLU() }

// HardSwish returns x*ReLU6(x+3)/6.
func (t *Tensor) HardSwish() *Tensor {
	return unaryOp(t, "HardSwish",
		func(v float64) float64 {
			if v <= -3 {
				return 0
			}
			if v >= 3 {
				return v
			}
			return v * (v + 3) / 6
		},
		func(g, x, y float64) float64 {
			if x <= -3 {
				return 0
			}
			if x >= 3 {
				return g
			}
			return g * (2*x + 3) / 6
		})
}

// Mish returns x * tanh(softplus(x)).
func (t *Tensor) Mish() *Tensor {
	return unaryOp(t, "Mish",
		func(v float64) float64 {
			sp := math.Log(1 + math.Exp(v))
			return v * math.Tanh(sp)
		},
		func(g, x, y float64) float64 {
			sp := math.Log(1 + math.Exp(x))
			th := math.Tanh(sp)
			dSp := 1 / (1 + math.Exp(-x))
			return g * (th + x*(1-th*th)*dSp)
		})
}

// ReLU6 returns min(max(0, x), 6).
func (t *Tensor) ReLU6() *Tensor {
	return unaryOp(t, "ReLU6",
		func(v float64) float64 {
			if v < 0 {
				return 0
			}
			if v > 6 {
				return 6
			}
			return v
		},
		func(g, x, y float64) float64 {
			if x <= 0 || x >= 6 {
				return 0
			}
			return g
		})
}

// Softmax along axis. Numerically stable (subtracts max).
func (t *Tensor) Softmax(axis int) *Tensor {
	if axis < 0 {
		axis += len(t.Shape)
	}
	mx := t.MaxAxis(axis, true)
	shifted := t.Sub(mx)
	exp := shifted.Exp()
	return exp.Div(exp.SumAxis(axis, true))
}

// LogSoftmax along axis.
func (t *Tensor) LogSoftmax(axis int) *Tensor {
	if axis < 0 {
		axis += len(t.Shape)
	}
	mx := t.MaxAxis(axis, true)
	shifted := t.Sub(mx)
	logSum := shifted.Exp().SumAxis(axis, true).Log()
	return shifted.Sub(logSum)
}
