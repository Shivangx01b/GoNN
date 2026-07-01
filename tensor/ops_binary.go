package tensor

// Binary elementwise ops with broadcasting + autograd.

// Add returns t + o.
func (t *Tensor) Add(o *Tensor) *Tensor { return binOp(t, o, "Add") }

// Sub returns t - o.
func (t *Tensor) Sub(o *Tensor) *Tensor { return binOp(t, o, "Sub") }

// Mul returns t * o (elementwise).
func (t *Tensor) Mul(o *Tensor) *Tensor { return binOp(t, o, "Mul") }

// Div returns t / o (elementwise).
func (t *Tensor) Div(o *Tensor) *Tensor { return binOp(t, o, "Div") }

// AddScalar returns t + v.
func (t *Tensor) AddScalar(v float64) *Tensor { return t.Add(Scalar(v)) }

// SubScalar returns t - v.
func (t *Tensor) SubScalar(v float64) *Tensor { return t.Sub(Scalar(v)) }

// MulScalar returns t * v.
func (t *Tensor) MulScalar(v float64) *Tensor { return t.Mul(Scalar(v)) }

// DivScalar returns t / v.
func (t *Tensor) DivScalar(v float64) *Tensor { return t.Div(Scalar(v)) }

// Neg returns -t.
func (t *Tensor) Neg() *Tensor { return t.MulScalar(-1) }

func binOp(a, b *Tensor, op string) *Tensor {
	outShape := broadcastShapes(a.Shape, b.Shape)
	ax, bx := expandTo(a, outShape), expandTo(b, outShape)
	out := Zeros(outShape...)
	if !dispatchBinary(binaryKindOf(op), ax.Data, bx.Data, out.Data) {
		switch op {
		case "Add":
			for i := range out.Data {
				out.Data[i] = ax.Data[i] + bx.Data[i]
			}
		case "Sub":
			for i := range out.Data {
				out.Data[i] = ax.Data[i] - bx.Data[i]
			}
		case "Mul":
			for i := range out.Data {
				out.Data[i] = ax.Data[i] * bx.Data[i]
			}
		case "Div":
			for i := range out.Data {
				out.Data[i] = ax.Data[i] / bx.Data[i]
			}
		}
	}
	finishOp(out, promote(a.Dtype, b.Dtype))
	if a.RequiresGrad || b.RequiresGrad || a.creator != nil || b.creator != nil {
		out.RequiresGrad = true
		out.creator = &Function{
			Name:   op,
			Inputs: []*Tensor{a, b},
			Saved:  []interface{}{ax, bx, op},
			Backward: func(grad *Tensor, saved []interface{}, inputs []*Tensor) []*Tensor {
				ax := saved[0].(*Tensor)
				bx := saved[1].(*Tensor)
				op := saved[2].(string)
				ga := Zeros(grad.Shape...)
				gb := Zeros(grad.Shape...)
				switch op {
				case "Add":
					copy(ga.Data, grad.Data)
					copy(gb.Data, grad.Data)
				case "Sub":
					copy(ga.Data, grad.Data)
					for i := range gb.Data {
						gb.Data[i] = -grad.Data[i]
					}
				case "Mul":
					for i := range ga.Data {
						ga.Data[i] = grad.Data[i] * bx.Data[i]
						gb.Data[i] = grad.Data[i] * ax.Data[i]
					}
				case "Div":
					for i := range ga.Data {
						ga.Data[i] = grad.Data[i] / bx.Data[i]
						gb.Data[i] = -grad.Data[i] * ax.Data[i] / (bx.Data[i] * bx.Data[i])
					}
				}
				return []*Tensor{ga, gb}
			},
		}
	}
	return out
}

