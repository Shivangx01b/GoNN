package ml

import (
	"gonum.org/v1/gonum/mat"
)

// LinearRegression fits y = X*w + b via the normal equations.
type LinearRegression struct {
	Weights []float64
	Bias    float64
}

func (m *LinearRegression) Fit(X [][]float64, y []float64) {
	n, d := shapeOf(X)
	// Augment X with a bias column.
	aug := make([]float64, n*(d+1))
	for i := 0; i < n; i++ {
		aug[i*(d+1)] = 1.0
		for j := 0; j < d; j++ {
			aug[i*(d+1)+1+j] = X[i][j]
		}
	}
	A := mat.NewDense(n, d+1, aug)
	b := mat.NewVecDense(n, append([]float64(nil), y...))

	var w mat.VecDense
	// Solve normal equations A^T A w = A^T b
	var AtA mat.Dense
	AtA.Mul(A.T(), A)
	var Atb mat.VecDense
	Atb.MulVec(A.T(), b)
	if err := w.SolveVec(&AtA, &Atb); err != nil {
		// Fall back to pseudo-inverse via SVD.
		var svd mat.SVD
		if !svd.Factorize(A, mat.SVDThin) {
			panic("LinearRegression: SVD failed")
		}
		var x mat.Dense
		// Use full rank (min of dims). Third arg is rank (int), not tolerance.
		rank := d + 1
		if n < rank {
			rank = n
		}
		svd.SolveTo(&x, mat.NewDense(n, 1, append([]float64(nil), y...)), rank)
		coefs := make([]float64, d+1)
		for i := 0; i < d+1; i++ {
			coefs[i] = x.At(i, 0)
		}
		m.Bias = coefs[0]
		m.Weights = coefs[1:]
		return
	}
	m.Bias = w.AtVec(0)
	m.Weights = make([]float64, d)
	for i := 0; i < d; i++ {
		m.Weights[i] = w.AtVec(i + 1)
	}
}

func (m *LinearRegression) Predict(X [][]float64) []float64 {
	n := len(X)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = m.Bias + dot(X[i], m.Weights)
	}
	return out
}

// Ridge is L2-regularized linear regression with closed-form solution.
type Ridge struct {
	Alpha   float64
	Weights []float64
	Bias    float64
}

func (m *Ridge) Fit(X [][]float64, y []float64) {
	if m.Alpha == 0 {
		m.Alpha = 1.0
	}
	n, d := shapeOf(X)
	// Center to avoid regularizing intercept.
	mx := meanCol(X)
	my := 0.0
	for _, v := range y {
		my += v
	}
	my /= float64(n)

	Xc := make([]float64, n*d)
	yc := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			Xc[i*d+j] = X[i][j] - mx[j]
		}
		yc[i] = y[i] - my
	}
	A := mat.NewDense(n, d, Xc)
	b := mat.NewVecDense(n, yc)
	var AtA mat.Dense
	AtA.Mul(A.T(), A)
	// Add alpha * I
	for i := 0; i < d; i++ {
		AtA.Set(i, i, AtA.At(i, i)+m.Alpha)
	}
	var Atb mat.VecDense
	Atb.MulVec(A.T(), b)
	var w mat.VecDense
	if err := w.SolveVec(&AtA, &Atb); err != nil {
		panic("Ridge: solve failed")
	}
	m.Weights = make([]float64, d)
	for i := 0; i < d; i++ {
		m.Weights[i] = w.AtVec(i)
	}
	// Recover intercept
	b0 := my
	for j := 0; j < d; j++ {
		b0 -= m.Weights[j] * mx[j]
	}
	m.Bias = b0
}

func (m *Ridge) Predict(X [][]float64) []float64 {
	n := len(X)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = m.Bias + dot(X[i], m.Weights)
	}
	return out
}
