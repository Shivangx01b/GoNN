package ml

import "math"

// softThreshold is the proximal operator for L1: sign(z) * max(|z| - gamma, 0).
func softThreshold(z, gamma float64) float64 {
	if z > gamma {
		return z - gamma
	}
	if z < -gamma {
		return z + gamma
	}
	return 0.0
}

// Lasso fits L1-regularized linear regression via coordinate descent.
//
// Minimizes (1 / (2n)) * ||y - Xw - b||^2 + Alpha * ||w||_1.
type Lasso struct {
	Alpha   float64
	MaxIter int
	Tol     float64

	Weights []float64
	Bias    float64
}

func (m *Lasso) Fit(X [][]float64, y []float64) {
	if m.Alpha == 0 {
		m.Alpha = 1.0
	}
	if m.MaxIter == 0 {
		m.MaxIter = 1000
	}
	if m.Tol == 0 {
		m.Tol = 1e-4
	}
	n, d := shapeOf(X)
	// Center features and target so we don't penalize the intercept.
	meanX := meanCol(X)
	meanY := 0.0
	for _, v := range y {
		meanY += v
	}
	meanY /= float64(n)

	Xc := make([][]float64, n)
	yc := make([]float64, n)
	for i := 0; i < n; i++ {
		Xc[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			Xc[i][j] = X[i][j] - meanX[j]
		}
		yc[i] = y[i] - meanY
	}
	// Precompute per-column squared norms.
	col2 := make([]float64, d)
	for j := 0; j < d; j++ {
		s := 0.0
		for i := 0; i < n; i++ {
			s += Xc[i][j] * Xc[i][j]
		}
		col2[j] = s
	}

	w := make([]float64, d)
	// Residual r = yc - Xc*w (starts as yc).
	r := make([]float64, n)
	copy(r, yc)

	nf := float64(n)
	for iter := 0; iter < m.MaxIter; iter++ {
		maxChange := 0.0
		for j := 0; j < d; j++ {
			if col2[j] == 0 {
				continue
			}
			// Add back contribution of feature j: r += Xc[:,j] * w[j]
			wOld := w[j]
			if wOld != 0 {
				for i := 0; i < n; i++ {
					r[i] += Xc[i][j] * wOld
				}
			}
			// Compute rho = Xc[:,j] . r
			rho := 0.0
			for i := 0; i < n; i++ {
				rho += Xc[i][j] * r[i]
			}
			wNew := softThreshold(rho/nf, m.Alpha) / (col2[j] / nf)
			w[j] = wNew
			if wNew != 0 {
				for i := 0; i < n; i++ {
					r[i] -= Xc[i][j] * wNew
				}
			}
			diff := math.Abs(wNew - wOld)
			if diff > maxChange {
				maxChange = diff
			}
		}
		if maxChange < m.Tol {
			break
		}
	}

	m.Weights = w
	b := meanY
	for j := 0; j < d; j++ {
		b -= w[j] * meanX[j]
	}
	m.Bias = b
}

func (m *Lasso) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i, x := range X {
		out[i] = m.Bias + dot(x, m.Weights)
	}
	return out
}

// ElasticNet mixes L1 and L2 regularization via coordinate descent.
//
// Minimizes
//   (1 / (2n)) * ||y - Xw - b||^2
//   + Alpha * L1Ratio * ||w||_1
//   + 0.5 * Alpha * (1 - L1Ratio) * ||w||_2^2.
type ElasticNet struct {
	Alpha   float64
	L1Ratio float64
	MaxIter int
	Tol     float64

	Weights []float64
	Bias    float64
}

func (m *ElasticNet) Fit(X [][]float64, y []float64) {
	if m.Alpha == 0 {
		m.Alpha = 1.0
	}
	if m.L1Ratio == 0 {
		m.L1Ratio = 0.5
	}
	if m.MaxIter == 0 {
		m.MaxIter = 1000
	}
	if m.Tol == 0 {
		m.Tol = 1e-4
	}
	n, d := shapeOf(X)
	meanX := meanCol(X)
	meanY := 0.0
	for _, v := range y {
		meanY += v
	}
	meanY /= float64(n)

	Xc := make([][]float64, n)
	yc := make([]float64, n)
	for i := 0; i < n; i++ {
		Xc[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			Xc[i][j] = X[i][j] - meanX[j]
		}
		yc[i] = y[i] - meanY
	}
	col2 := make([]float64, d)
	for j := 0; j < d; j++ {
		s := 0.0
		for i := 0; i < n; i++ {
			s += Xc[i][j] * Xc[i][j]
		}
		col2[j] = s
	}

	w := make([]float64, d)
	r := make([]float64, n)
	copy(r, yc)

	nf := float64(n)
	l1 := m.Alpha * m.L1Ratio
	l2 := m.Alpha * (1 - m.L1Ratio)
	for iter := 0; iter < m.MaxIter; iter++ {
		maxChange := 0.0
		for j := 0; j < d; j++ {
			if col2[j] == 0 {
				continue
			}
			wOld := w[j]
			if wOld != 0 {
				for i := 0; i < n; i++ {
					r[i] += Xc[i][j] * wOld
				}
			}
			rho := 0.0
			for i := 0; i < n; i++ {
				rho += Xc[i][j] * r[i]
			}
			wNew := softThreshold(rho/nf, l1) / (col2[j]/nf + l2)
			w[j] = wNew
			if wNew != 0 {
				for i := 0; i < n; i++ {
					r[i] -= Xc[i][j] * wNew
				}
			}
			diff := math.Abs(wNew - wOld)
			if diff > maxChange {
				maxChange = diff
			}
		}
		if maxChange < m.Tol {
			break
		}
	}

	m.Weights = w
	b := meanY
	for j := 0; j < d; j++ {
		b -= w[j] * meanX[j]
	}
	m.Bias = b
}

func (m *ElasticNet) Predict(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i, x := range X {
		out[i] = m.Bias + dot(x, m.Weights)
	}
	return out
}
