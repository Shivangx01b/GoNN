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

// MatMul performs matrix multiplication. Supports 2D x 2D for now.
// (Batched matmul would handle leading dims with broadcasting.)
func (t *Tensor) MatMul(o *Tensor) *Tensor {
	if len(t.Shape) != 2 || len(o.Shape) != 2 {
		panic("MatMul: both tensors must be 2D")
	}
	m, k := t.Shape[0], t.Shape[1]
	k2, n := o.Shape[0], o.Shape[1]
	if k != k2 {
		panic("MatMul: inner dims do not match")
	}
	out := Zeros(m, n)
	for i := 0; i < m; i++ {
		for kk := 0; kk < k; kk++ {
			a := t.Data[i*k+kk]
			if a == 0 {
				continue
			}
			for j := 0; j < n; j++ {
				out.Data[i*n+j] += a * o.Data[kk*n+j]
			}
		}
	}
	if t.RequiresGrad || o.RequiresGrad || t.creator != nil || o.creator != nil {
		out.RequiresGrad = true
		out.creator = &Function{
			Name:   "MatMul",
			Inputs: []*Tensor{t, o},
			Backward: func(grad *Tensor, _ []interface{}, inputs []*Tensor) []*Tensor {
				a, b := inputs[0], inputs[1]
				// dA = grad @ B^T,  dB = A^T @ grad
				ga := matmul2D(grad.Data, transpose2D(b.Data, b.Shape[0], b.Shape[1]),
					grad.Shape[0], grad.Shape[1], b.Shape[0])
				gb := matmul2D(transpose2D(a.Data, a.Shape[0], a.Shape[1]), grad.Data,
					a.Shape[1], a.Shape[0], grad.Shape[1])
				return []*Tensor{
					New(ga, a.Shape...),
					New(gb, b.Shape...),
				}
			},
		}
	}
	return out
}

func matmul2D(A, B []float64, m, k, n int) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for kk := 0; kk < k; kk++ {
			a := A[i*k+kk]
			if a == 0 {
				continue
			}
			for j := 0; j < n; j++ {
				out[i*n+j] += a * B[kk*n+j]
			}
		}
	}
	return out
}

func transpose2D(A []float64, m, n int) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out[j*m+i] = A[i*n+j]
		}
	}
	return out
}
