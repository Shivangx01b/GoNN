package tensor

import (
	"math"
	"math/rand"
)

// Activation functions (forward + autograd backward), defined once as
// UnaryOpDefs. Fixed activations are registered with the registry
// (registry.go), so nn's generic Activation module and (*Tensor).Unary reach
// the exact same definitions as the fluent methods below; the ones with GPU
// kernels carry the matching dispatch Kind. Parameterized activations
// (LeakyReLU(alpha), Threshold(t, v), ...) build ephemeral defs capturing
// their parameters. All closures are unchanged from the original
// implementations — numerics are identical.

var (
	reluDef = UnaryOpDef{Name: "ReLU", Kind: UnaryReLU,
		Fwd: func(v float64) float64 {
			if v > 0 {
				return v
			}
			return 0
		},
		Bwd: func(g, x, y float64) float64 {
			if x > 0 {
				return g
			}
			return 0
		}}

	seluDef = UnaryOpDef{Name: "SELU", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			const lambda = 1.0507009873554805
			const alpha = 1.6732632423543772
			if v > 0 {
				return lambda * v
			}
			return lambda * alpha * (math.Exp(v) - 1)
		},
		Bwd: func(g, x, y float64) float64 {
			const lambda = 1.0507009873554805
			const alpha = 1.6732632423543772
			if x > 0 {
				return g * lambda
			}
			return g * (y + lambda*alpha)
		}}

	sigmoidDef = UnaryOpDef{Name: "Sigmoid", Kind: UnarySigmoid,
		Fwd: func(v float64) float64 { return 1 / (1 + math.Exp(-v)) },
		Bwd: func(g, x, y float64) float64 { return g * y * (1 - y) }}

	tanhDef = UnaryOpDef{Name: "Tanh", Kind: UnaryTanh,
		Fwd: math.Tanh,
		Bwd: func(g, x, y float64) float64 { return g * (1 - y*y) }}

	hardTanhDef = UnaryOpDef{Name: "HardTanh", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v < -1 {
				return -1
			}
			if v > 1 {
				return 1
			}
			return v
		},
		Bwd: func(g, x, y float64) float64 {
			if x < -1 || x > 1 {
				return 0
			}
			return g
		}}

	hardSigmoidDef = UnaryOpDef{Name: "HardSigmoid", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v <= -3 {
				return 0
			}
			if v >= 3 {
				return 1
			}
			return v/6 + 0.5
		},
		Bwd: func(g, x, y float64) float64 {
			if x <= -3 || x >= 3 {
				return 0
			}
			return g / 6
		}}

	softplusDef = UnaryOpDef{Name: "Softplus", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v > 20 {
				return v
			}
			return math.Log(1 + math.Exp(v))
		},
		Bwd: func(g, x, y float64) float64 { return g / (1 + math.Exp(-x)) }}

	softsignDef = UnaryOpDef{Name: "Softsign", Kind: UnaryNone,
		Fwd: func(v float64) float64 { return v / (1 + math.Abs(v)) },
		Bwd: func(g, x, y float64) float64 {
			d := 1 + math.Abs(x)
			return g / (d * d)
		}}

	// GELU uses the tanh approximation (PyTorch default-compatible);
	// 0.7978845608028654 = sqrt(2/pi).
	geluDef = UnaryOpDef{Name: "GELU", Kind: UnaryGELU,
		Fwd: func(v float64) float64 {
			const c = 0.7978845608028654
			const k = 0.044715
			return 0.5 * v * (1 + math.Tanh(c*(v+k*v*v*v)))
		},
		Bwd: func(g, x, y float64) float64 {
			const c = 0.7978845608028654
			const k = 0.044715
			inner := c * (x + k*x*x*x)
			tanh := math.Tanh(inner)
			sech2 := 1 - tanh*tanh
			dInner := c * (1 + 3*k*x*x)
			return g * (0.5*(1+tanh) + 0.5*x*sech2*dInner)
		}}

	siluDef = UnaryOpDef{Name: "SiLU", Kind: UnarySiLU,
		Fwd: func(v float64) float64 { return v / (1 + math.Exp(-v)) },
		Bwd: func(g, x, y float64) float64 {
			s := 1 / (1 + math.Exp(-x))
			return g * (s + x*s*(1-s))
		}}

	hardSwishDef = UnaryOpDef{Name: "HardSwish", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v <= -3 {
				return 0
			}
			if v >= 3 {
				return v
			}
			return v * (v + 3) / 6
		},
		Bwd: func(g, x, y float64) float64 {
			if x <= -3 {
				return 0
			}
			if x >= 3 {
				return g
			}
			return g * (2*x + 3) / 6
		}}

	mishDef = UnaryOpDef{Name: "Mish", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			sp := math.Log(1 + math.Exp(v))
			return v * math.Tanh(sp)
		},
		Bwd: func(g, x, y float64) float64 {
			sp := math.Log(1 + math.Exp(x))
			th := math.Tanh(sp)
			dSp := 1 / (1 + math.Exp(-x))
			return g * (th + x*(1-th*th)*dSp)
		}}

	relu6Def = UnaryOpDef{Name: "ReLU6", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v < 0 {
				return 0
			}
			if v > 6 {
				return 6
			}
			return v
		},
		Bwd: func(g, x, y float64) float64 {
			if x <= 0 || x >= 6 {
				return 0
			}
			return g
		}}

	// LogSigmoid returns log(sigmoid(x)) = -softplus(-x), computed stably:
	//   log(sigmoid(x)) = -log(1 + exp(-x)) = min(0, x) - log(1 + exp(-|x|)).
	// Backward: d/dx log(sigmoid(x)) = 1 - sigmoid(x) = 1/(1+exp(x)).
	logSigmoidDef = UnaryOpDef{Name: "LogSigmoid", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			// stable: -softplus(-v)
			if -v > 20 {
				// softplus(-v) ~= -v
				return v
			}
			return -math.Log(1 + math.Exp(-v))
		},
		Bwd: func(g, x, y float64) float64 { return g / (1 + math.Exp(x)) }}

	// Tanhshrink returns x - tanh(x); d/dx = tanh^2(x).
	tanhshrinkDef = UnaryOpDef{Name: "Tanhshrink", Kind: UnaryNone,
		Fwd: func(v float64) float64 { return v - math.Tanh(v) },
		Bwd: func(g, x, y float64) float64 {
			th := math.Tanh(x)
			return g * th * th
		}}
)

func init() {
	RegisterUnary(reluDef)
	RegisterUnary(seluDef)
	RegisterUnary(sigmoidDef)
	RegisterUnary(tanhDef)
	RegisterUnary(hardTanhDef)
	RegisterUnary(hardSigmoidDef)
	RegisterUnary(softplusDef)
	RegisterUnary(softsignDef)
	RegisterUnary(geluDef)
	RegisterUnary(siluDef)
	RegisterUnary(hardSwishDef)
	RegisterUnary(mishDef)
	RegisterUnary(relu6Def)
	RegisterUnary(logSigmoidDef)
	RegisterUnary(tanhshrinkDef)
}

// ReLU returns max(0, t).
func (t *Tensor) ReLU() *Tensor { return applyUnary(t, reluDef) }

// SELU activation.
func (t *Tensor) SELU() *Tensor { return applyUnary(t, seluDef) }

// Sigmoid returns 1/(1+e^-t).
func (t *Tensor) Sigmoid() *Tensor { return applyUnary(t, sigmoidDef) }

// Tanh returns tanh(t).
func (t *Tensor) Tanh() *Tensor { return applyUnary(t, tanhDef) }

// HardTanh clamps to [-1, 1].
func (t *Tensor) HardTanh() *Tensor { return applyUnary(t, hardTanhDef) }

// HardSigmoid is a piecewise linear approx to sigmoid.
func (t *Tensor) HardSigmoid() *Tensor { return applyUnary(t, hardSigmoidDef) }

// Softplus returns ln(1+e^t).
func (t *Tensor) Softplus() *Tensor { return applyUnary(t, softplusDef) }

// Softsign returns x/(1+|x|).
func (t *Tensor) Softsign() *Tensor { return applyUnary(t, softsignDef) }

// GELU using the tanh approximation (PyTorch default-compatible).
func (t *Tensor) GELU() *Tensor { return applyUnary(t, geluDef) }

// SiLU (Swish) returns x * sigmoid(x).
func (t *Tensor) SiLU() *Tensor { return applyUnary(t, siluDef) }

// Swish is an alias for SiLU.
func (t *Tensor) Swish() *Tensor { return t.SiLU() }

// HardSwish returns x*ReLU6(x+3)/6.
func (t *Tensor) HardSwish() *Tensor { return applyUnary(t, hardSwishDef) }

// Mish returns x * tanh(softplus(x)).
func (t *Tensor) Mish() *Tensor { return applyUnary(t, mishDef) }

// ReLU6 returns min(max(0, x), 6).
func (t *Tensor) ReLU6() *Tensor { return applyUnary(t, relu6Def) }

// LogSigmoid returns log(sigmoid(x)), computed stably.
func (t *Tensor) LogSigmoid() *Tensor { return applyUnary(t, logSigmoidDef) }

// Tanhshrink returns x - tanh(x).
func (t *Tensor) Tanhshrink() *Tensor { return applyUnary(t, tanhshrinkDef) }

// ---- Parameterized activations (ephemeral defs; closures capture params) ---

// LeakyReLU returns x if x>0 else alpha*x.
func (t *Tensor) LeakyReLU(alpha float64) *Tensor {
	return applyUnary(t, UnaryOpDef{Name: "LeakyReLU", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v > 0 {
				return v
			}
			return alpha * v
		},
		Bwd: func(g, x, y float64) float64 {
			if x > 0 {
				return g
			}
			return g * alpha
		}})
}

// ELU returns x if x>0 else alpha*(e^x-1).
func (t *Tensor) ELU(alpha float64) *Tensor {
	return applyUnary(t, UnaryOpDef{Name: "ELU", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v > 0 {
				return v
			}
			return alpha * (math.Exp(v) - 1)
		},
		Bwd: func(g, x, y float64) float64 {
			if x > 0 {
				return g
			}
			return g * (y + alpha)
		}})
}

// Hardshrink: x if |x| > lambda else 0. Gradient is 1 in the same region.
func (t *Tensor) Hardshrink(lambda float64) *Tensor {
	return applyUnary(t, UnaryOpDef{Name: "Hardshrink", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v > lambda || v < -lambda {
				return v
			}
			return 0
		},
		Bwd: func(g, x, y float64) float64 {
			if x > lambda || x < -lambda {
				return g
			}
			return 0
		}})
}

// Softshrink: sign(x) * max(|x| - lambda, 0). Gradient is 1 where |x| > lambda.
func (t *Tensor) Softshrink(lambda float64) *Tensor {
	return applyUnary(t, UnaryOpDef{Name: "Softshrink", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v > lambda {
				return v - lambda
			}
			if v < -lambda {
				return v + lambda
			}
			return 0
		},
		Bwd: func(g, x, y float64) float64 {
			if x > lambda || x < -lambda {
				return g
			}
			return 0
		}})
}

// Threshold: x if x > thresh else value. Gradient is 1 where x > thresh.
func (t *Tensor) Threshold(thresh, value float64) *Tensor {
	return applyUnary(t, UnaryOpDef{Name: "Threshold", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v > thresh {
				return v
			}
			return value
		},
		Bwd: func(g, x, y float64) float64 {
			if x > thresh {
				return g
			}
			return 0
		}})
}

// CELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1)).
// d/dx = 1 for x > 0; otherwise exp(x/alpha).
func (t *Tensor) CELU(alpha float64) *Tensor {
	if alpha == 0 {
		opError("CELU", "alpha must be non-zero")
	}
	return applyUnary(t, UnaryOpDef{Name: "CELU", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v > 0 {
				return v
			}
			return alpha * (math.Exp(v/alpha) - 1)
		},
		Bwd: func(g, x, y float64) float64 {
			if x > 0 {
				return g
			}
			return g * math.Exp(x/alpha)
		}})
}

// RReLU: Randomized leaky ReLU. PyTorch samples a random slope per element
// from U(lower, upper) at training time. To keep the single fwd/bwd-closure
// shape (keyed on (x, y)), this implementation is deterministic and uses the
// midpoint slope = (lower+upper)/2, mirroring PyTorch's eval-mode behavior.
// The rng parameter is accepted for API parity but currently unused; pass nil
// if you don't care.
func (t *Tensor) RReLU(lower, upper float64, rng *rand.Rand) *Tensor {
	_ = rng
	slope := 0.5 * (lower + upper)
	return applyUnary(t, UnaryOpDef{Name: "RReLU", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v >= 0 {
				return v
			}
			return slope * v
		},
		Bwd: func(g, x, y float64) float64 {
			if x >= 0 {
				return g
			}
			return g * slope
		}})
}

// ---- Axis-reduction "activations" (not elementwise; composed from ops) -----

// Softmax along axis. Numerically stable (subtracts max).
func (t *Tensor) Softmax(axis int) *Tensor {
	axis = normalizeAxis("Softmax", axis, len(t.Shape))
	mx := t.MaxAxis(axis, true)
	shifted := t.Sub(mx)
	exp := shifted.Exp()
	return exp.Div(exp.SumAxis(axis, true))
}

// LogSoftmax along axis.
func (t *Tensor) LogSoftmax(axis int) *Tensor {
	axis = normalizeAxis("LogSoftmax", axis, len(t.Shape))
	mx := t.MaxAxis(axis, true)
	shifted := t.Sub(mx)
	logSum := shifted.Exp().SumAxis(axis, true).Log()
	return shifted.Sub(logSum)
}
