package ml

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// BayesianRidge is Bayesian linear regression with Gamma priors on the noise
// precision (alpha) and weight precision (lambda). Hyperparameters are
// estimated via the closed-form evidence (marginal likelihood) updates that
// scikit-learn uses.
type BayesianRidge struct {
	Alpha1, Alpha2   float64 // shape and rate priors on noise precision
	Lambda1, Lambda2 float64 // shape and rate priors on weight precision
	MaxIter          int
	Tol              float64

	AlphaPrec  float64 // noise precision
	LambdaPrec float64 // weight precision
	Weights    []float64
	Bias       float64
	// Covariance of weight posterior (excluding bias).
	Sigma *mat.Dense
}

func (m *BayesianRidge) Fit(X [][]float64, y []float64) {
	if m.Alpha1 == 0 {
		m.Alpha1 = 1e-6
	}
	if m.Alpha2 == 0 {
		m.Alpha2 = 1e-6
	}
	if m.Lambda1 == 0 {
		m.Lambda1 = 1e-6
	}
	if m.Lambda2 == 0 {
		m.Lambda2 = 1e-6
	}
	if m.MaxIter == 0 {
		m.MaxIter = 300
	}
	if m.Tol == 0 {
		m.Tol = 1e-3
	}
	n, d := shapeOf(X)
	// Center to handle bias separately.
	meanX := meanCol(X)
	meanY := 0.0
	for _, v := range y {
		meanY += v
	}
	meanY /= float64(n)
	Xc := mat.NewDense(n, d, nil)
	yc := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			Xc.Set(i, j, X[i][j]-meanX[j])
		}
		yc[i] = y[i] - meanY
	}

	// X^T X eigendecomposition so per-iteration updates are O(d^2) instead of O(d^3).
	var XtX mat.Dense
	XtX.Mul(Xc.T(), Xc)
	data := make([]float64, d*d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			data[i*d+j] = 0.5 * (XtX.At(i, j) + XtX.At(j, i))
		}
	}
	XtXSym := mat.NewSymDense(d, data)
	var eig mat.EigenSym
	if ok := eig.Factorize(XtXSym, true); !ok {
		panic("BayesianRidge: eigendecomposition failed")
	}
	eigvals := eig.Values(nil)
	var V mat.Dense
	eig.VectorsTo(&V)
	// XtY
	yVec := mat.NewVecDense(n, yc)
	var XtY mat.VecDense
	XtY.MulVec(Xc.T(), yVec)
	// VtXtY = V^T * (X^T y)
	VtXtY := make([]float64, d)
	for i := 0; i < d; i++ {
		s := 0.0
		for j := 0; j < d; j++ {
			s += V.At(j, i) * XtY.AtVec(j)
		}
		VtXtY[i] = s
	}

	alpha := 1.0
	lambda := 1.0
	w := make([]float64, d)
	prevW := make([]float64, d)

	for iter := 0; iter < m.MaxIter; iter++ {
		// Posterior mean in eigen-basis: w_eig[i] = alpha * eigvals[i] / (lambda + alpha*eigvals[i]) * VtXtY[i] / eigvals[i]
		// equivalently mu = V * (alpha / (lambda + alpha*lam_i)) * VtXtY[i]
		coeffs := make([]float64, d)
		for i := 0; i < d; i++ {
			denom := lambda + alpha*eigvals[i]
			if denom < 1e-12 {
				denom = 1e-12
			}
			coeffs[i] = alpha * VtXtY[i] / denom
		}
		// w = V * coeffs
		for i := 0; i < d; i++ {
			s := 0.0
			for j := 0; j < d; j++ {
				s += V.At(i, j) * coeffs[j]
			}
			w[i] = s
		}
		// gamma = sum(alpha * lam_i / (lambda + alpha*lam_i))
		gamma := 0.0
		for i := 0; i < d; i++ {
			denom := lambda + alpha*eigvals[i]
			if denom < 1e-12 {
				denom = 1e-12
			}
			gamma += alpha * eigvals[i] / denom
		}
		// Update lambda and alpha.
		wNormSq := 0.0
		for _, v := range w {
			wNormSq += v * v
		}
		newLambda := (gamma + 2*m.Lambda1) / (wNormSq + 2*m.Lambda2 + 1e-12)
		// Residual sum of squares
		var pred mat.VecDense
		pred.MulVec(Xc, mat.NewVecDense(d, w))
		rss := 0.0
		for i := 0; i < n; i++ {
			df := yc[i] - pred.AtVec(i)
			rss += df * df
		}
		newAlpha := (float64(n) - gamma + 2*m.Alpha1) / (rss + 2*m.Alpha2 + 1e-12)

		// Convergence check.
		maxDiff := 0.0
		for i := 0; i < d; i++ {
			df := math.Abs(w[i] - prevW[i])
			if df > maxDiff {
				maxDiff = df
			}
		}
		copy(prevW, w)
		alpha = newAlpha
		lambda = newLambda
		if maxDiff < m.Tol {
			break
		}
	}

	m.AlphaPrec = alpha
	m.LambdaPrec = lambda
	m.Weights = w
	// Bias.
	b := meanY
	for j := 0; j < d; j++ {
		b -= w[j] * meanX[j]
	}
	m.Bias = b
	// Posterior covariance Sigma = (lambda*I + alpha * X^T X)^{-1}
	A := mat.NewDense(d, d, nil)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			A.Set(i, j, alpha*XtX.At(i, j))
		}
		A.Set(i, i, A.At(i, i)+lambda)
	}
	var sig mat.Dense
	if err := sig.Inverse(A); err == nil {
		m.Sigma = &sig
	}
}

func (m *BayesianRidge) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i, x := range X {
		out[i] = m.Bias + dot(x, m.Weights)
	}
	return out
}

// PredictWithStd returns predictions and per-sample posterior standard deviations.
func (m *BayesianRidge) PredictWithStd(X [][]float64) ([]float64, []float64) {
	preds := m.Predict(X)
	stds := make([]float64, len(X))
	d := len(m.Weights)
	for i, x := range X {
		// var = 1/alpha + x^T Sigma x
		v := 1.0 / m.AlphaPrec
		if m.Sigma != nil {
			for a := 0; a < d; a++ {
				s := 0.0
				for b := 0; b < d; b++ {
					s += m.Sigma.At(a, b) * x[b]
				}
				v += x[a] * s
			}
		}
		if v < 0 {
			v = 0
		}
		stds[i] = math.Sqrt(v)
	}
	return preds, stds
}
