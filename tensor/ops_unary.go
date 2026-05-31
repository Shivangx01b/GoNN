package tensor

import "math"

// Unary elementwise ops with autograd.

func unaryOp(t *Tensor, name string, fwd func(float64) float64, bwd func(grad, x, y float64) float64) *Tensor {
	out := Zeros(t.Shape...)
	for i, v := range t.Data {
		out.Data[i] = fwd(v)
	}
	out.Dtype = t.Dtype
	castInPlace(out)
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		input := t
		outData := out.Data
		out.creator = &Function{
			Name:   name,
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				g := Zeros(grad.Shape...)
				for i := range g.Data {
					g.Data[i] = bwd(grad.Data[i], input.Data[i], outData[i])
				}
				return []*Tensor{g}
			},
		}
	}
	return out
}

// Exp returns e^t.
func (t *Tensor) Exp() *Tensor {
	return unaryOp(t, "Exp", math.Exp, func(g, x, y float64) float64 { return g * y })
}

// Log returns ln(t).
func (t *Tensor) Log() *Tensor {
	return unaryOp(t, "Log", math.Log, func(g, x, y float64) float64 { return g / x })
}

// Sqrt returns sqrt(t).
func (t *Tensor) Sqrt() *Tensor {
	return unaryOp(t, "Sqrt", math.Sqrt, func(g, x, y float64) float64 { return g * 0.5 / y })
}

// Sin returns sin(t).
func (t *Tensor) Sin() *Tensor {
	return unaryOp(t, "Sin", math.Sin, func(g, x, y float64) float64 { return g * math.Cos(x) })
}

// Cos returns cos(t).
func (t *Tensor) Cos() *Tensor {
	return unaryOp(t, "Cos", math.Cos, func(g, x, y float64) float64 { return -g * math.Sin(x) })
}

// Tan returns tan(t).
func (t *Tensor) Tan() *Tensor {
	return unaryOp(t, "Tan", math.Tan, func(g, x, y float64) float64 {
		c := math.Cos(x)
		return g / (c * c)
	})
}

// Abs returns |t|.
func (t *Tensor) Abs() *Tensor {
	return unaryOp(t, "Abs", math.Abs, func(g, x, y float64) float64 {
		if x > 0 {
			return g
		}
		if x < 0 {
			return -g
		}
		return 0
	})
}

// Reciprocal returns 1/t.
func (t *Tensor) Reciprocal() *Tensor {
	return unaryOp(t, "Reciprocal", func(v float64) float64 { return 1 / v },
		func(g, x, y float64) float64 { return -g / (x * x) })
}

// Pow returns t^p (scalar exponent).
func (t *Tensor) Pow(p float64) *Tensor {
	return unaryOp(t, "Pow", func(v float64) float64 { return math.Pow(v, p) },
		func(g, x, y float64) float64 {
			if x == 0 {
				// d/dx x^p at 0 is 0 for p>1, 1 for p==1, and undefined
				// (+Inf) for p<1. Guard against the Inf; the finite cases
				// fall through to the general formula.
				if p == 1 {
					return g
				}
				if p > 1 {
					return 0
				}
			}
			return g * p * math.Pow(x, p-1)
		})
}

// Square returns t*t (more efficient than Pow(2)).
func (t *Tensor) Square() *Tensor {
	return unaryOp(t, "Square", func(v float64) float64 { return v * v },
		func(g, x, y float64) float64 { return g * 2 * x })
}

// Clip clamps each element to [min, max].
func (t *Tensor) Clip(minV, maxV float64) *Tensor {
	return unaryOp(t, "Clip",
		func(v float64) float64 {
			if v < minV {
				return minV
			}
			if v > maxV {
				return maxV
			}
			return v
		},
		func(g, x, y float64) float64 {
			if x < minV || x > maxV {
				return 0
			}
			return g
		},
	)
}
