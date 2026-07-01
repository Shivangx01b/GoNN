package tensor

import "math"

// Elementwise unary math ops. Fixed ops are UnaryOpDefs registered with the
// registry (registry.go) — one definition drives the fluent method, the
// name-based (*Tensor).Unary lookup, and GPU dispatch. Parameterized ops
// (Pow, Clip) build ephemeral defs whose closures capture the parameters.
// All forward/backward closures are unchanged from the original
// implementations — numerics are identical.

var (
	expDef = UnaryOpDef{Name: "Exp", Kind: UnaryExp,
		Fwd: math.Exp,
		Bwd: func(g, x, y float64) float64 { return g * y }}

	logDef = UnaryOpDef{Name: "Log", Kind: UnaryLog,
		Fwd: math.Log,
		Bwd: func(g, x, y float64) float64 { return g / x }}

	sqrtDef = UnaryOpDef{Name: "Sqrt", Kind: UnaryNone,
		Fwd: math.Sqrt,
		Bwd: func(g, x, y float64) float64 { return g * 0.5 / y }}

	sinDef = UnaryOpDef{Name: "Sin", Kind: UnaryNone,
		Fwd: math.Sin,
		Bwd: func(g, x, y float64) float64 { return g * math.Cos(x) }}

	cosDef = UnaryOpDef{Name: "Cos", Kind: UnaryNone,
		Fwd: math.Cos,
		Bwd: func(g, x, y float64) float64 { return -g * math.Sin(x) }}

	tanDef = UnaryOpDef{Name: "Tan", Kind: UnaryNone,
		Fwd: math.Tan,
		Bwd: func(g, x, y float64) float64 {
			c := math.Cos(x)
			return g / (c * c)
		}}

	absDef = UnaryOpDef{Name: "Abs", Kind: UnaryNone,
		Fwd: math.Abs,
		Bwd: func(g, x, y float64) float64 {
			if x > 0 {
				return g
			}
			if x < 0 {
				return -g
			}
			return 0
		}}

	reciprocalDef = UnaryOpDef{Name: "Reciprocal", Kind: UnaryNone,
		Fwd: func(v float64) float64 { return 1 / v },
		Bwd: func(g, x, y float64) float64 { return -g / (x * x) }}

	squareDef = UnaryOpDef{Name: "Square", Kind: UnaryNone,
		Fwd: func(v float64) float64 { return v * v },
		Bwd: func(g, x, y float64) float64 { return g * 2 * x }}
)

func init() {
	RegisterUnary(expDef)
	RegisterUnary(logDef)
	RegisterUnary(sqrtDef)
	RegisterUnary(sinDef)
	RegisterUnary(cosDef)
	RegisterUnary(tanDef)
	RegisterUnary(absDef)
	RegisterUnary(reciprocalDef)
	RegisterUnary(squareDef)
}

// Exp returns e^t.
func (t *Tensor) Exp() *Tensor { return applyUnary(t, expDef) }

// Log returns ln(t).
func (t *Tensor) Log() *Tensor { return applyUnary(t, logDef) }

// Sqrt returns sqrt(t).
func (t *Tensor) Sqrt() *Tensor { return applyUnary(t, sqrtDef) }

// Sin returns sin(t).
func (t *Tensor) Sin() *Tensor { return applyUnary(t, sinDef) }

// Cos returns cos(t).
func (t *Tensor) Cos() *Tensor { return applyUnary(t, cosDef) }

// Tan returns tan(t).
func (t *Tensor) Tan() *Tensor { return applyUnary(t, tanDef) }

// Abs returns |t|.
func (t *Tensor) Abs() *Tensor { return applyUnary(t, absDef) }

// Reciprocal returns 1/t.
func (t *Tensor) Reciprocal() *Tensor { return applyUnary(t, reciprocalDef) }

// Square returns t*t (more efficient than Pow(2)).
func (t *Tensor) Square() *Tensor { return applyUnary(t, squareDef) }

// Pow returns t^p (scalar exponent).
func (t *Tensor) Pow(p float64) *Tensor {
	return applyUnary(t, UnaryOpDef{Name: "Pow", Kind: UnaryNone,
		Fwd: func(v float64) float64 { return math.Pow(v, p) },
		Bwd: func(g, x, y float64) float64 {
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
		}})
}

// Clip clamps each element to [min, max].
func (t *Tensor) Clip(minV, maxV float64) *Tensor {
	return applyUnary(t, UnaryOpDef{Name: "Clip", Kind: UnaryNone,
		Fwd: func(v float64) float64 {
			if v < minV {
				return minV
			}
			if v > maxV {
				return maxV
			}
			return v
		},
		Bwd: func(g, x, y float64) float64 {
			if x < minV || x > maxV {
				return 0
			}
			return g
		}})
}
