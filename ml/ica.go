package ml

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// FastICA performs Independent Component Analysis using the FastICA algorithm
// with non-linearity g(u) = tanh(u).
//
// Fit() computes a whitening matrix and an unmixing matrix W such that
//   S = W * Z  where  Z = whitening * (X - mean)
type FastICA struct {
	NComponents int
	MaxIter     int
	Tol         float64
	Seed        int64

	Mean       []float64
	Whitening  *mat.Dense // [NComponents][d]
	Components *mat.Dense // unmixing matrix W [NComponents][NComponents]
}

func (m *FastICA) Fit(X [][]float64) {
	if m.MaxIter == 0 {
		m.MaxIter = 200
	}
	if m.Tol == 0 {
		m.Tol = 1e-4
	}
	n, d := shapeOf(X)
	if m.NComponents == 0 || m.NComponents > d {
		m.NComponents = d
	}
	K := m.NComponents

	// Center.
	m.Mean = meanCol(X)
	Xc := mat.NewDense(d, n, nil) // columns are samples
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			Xc.Set(j, i, X[i][j]-m.Mean[j])
		}
	}

	// Whitening via eigen of covariance.
	var cov mat.Dense
	cov.Mul(Xc, Xc.T())
	cov.Scale(1.0/float64(n-1), &cov)

	covData := make([]float64, d*d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			covData[i*d+j] = 0.5 * (cov.At(i, j) + cov.At(j, i))
		}
	}
	covSym := mat.NewSymDense(d, covData)
	var eig mat.EigenSym
	if ok := eig.Factorize(covSym, true); !ok {
		panic("FastICA: eigendecomposition failed")
	}
	vals := eig.Values(nil)
	var vecs mat.Dense
	eig.VectorsTo(&vecs)
	// Sort descending.
	idx := make([]int, d)
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < d; i++ {
		best := i
		for j := i + 1; j < d; j++ {
			if vals[idx[j]] > vals[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	// Build whitening matrix W_w = diag(1/sqrt(lambda)) * V^T, top K rows.
	whit := mat.NewDense(K, d, nil)
	for i := 0; i < K; i++ {
		col := idx[i]
		lam := vals[col]
		if lam < 1e-12 {
			lam = 1e-12
		}
		s := 1.0 / math.Sqrt(lam)
		for j := 0; j < d; j++ {
			whit.Set(i, j, s*vecs.At(j, col))
		}
	}
	m.Whitening = whit

	// Z = whitening * Xc -> [K, n]
	var Z mat.Dense
	Z.Mul(whit, Xc)

	rng := rand.New(rand.NewSource(m.Seed))

	// Initialize W with random orthonormal rows.
	Wdata := make([]float64, K*K)
	for i := range Wdata {
		Wdata[i] = rng.NormFloat64()
	}
	W := mat.NewDense(K, K, Wdata)
	W = symDecorrelate(W)

	for iter := 0; iter < m.MaxIter; iter++ {
		// WZ = W * Z, [K, n]
		var WZ mat.Dense
		WZ.Mul(W, &Z)
		// g(u) = tanh(u), g'(u) = 1 - tanh(u)^2
		gWZ := mat.NewDense(K, n, nil)
		gPrimeMean := make([]float64, K)
		for i := 0; i < K; i++ {
			for j := 0; j < n; j++ {
				t := math.Tanh(WZ.At(i, j))
				gWZ.Set(i, j, t)
				gPrimeMean[i] += 1.0 - t*t
			}
			gPrimeMean[i] /= float64(n)
		}
		// W_new = (gWZ * Z^T) / n - diag(gPrimeMean) * W
		var gZ mat.Dense
		gZ.Mul(gWZ, Z.T())
		gZ.Scale(1.0/float64(n), &gZ)
		Wnew := mat.NewDense(K, K, nil)
		for i := 0; i < K; i++ {
			for j := 0; j < K; j++ {
				Wnew.Set(i, j, gZ.At(i, j)-gPrimeMean[i]*W.At(i, j))
			}
		}
		Wnew = symDecorrelate(Wnew)
		// Convergence: max |(|w_i . w_i_new|) - 1|
		maxDiff := 0.0
		for i := 0; i < K; i++ {
			s := 0.0
			for j := 0; j < K; j++ {
				s += W.At(i, j) * Wnew.At(i, j)
			}
			d := math.Abs(math.Abs(s) - 1)
			if d > maxDiff {
				maxDiff = d
			}
		}
		W = Wnew
		if maxDiff < m.Tol {
			break
		}
	}
	m.Components = W
}

// symDecorrelate returns W_new = (W * W^T)^{-1/2} * W via eigen of W*W^T.
func symDecorrelate(W *mat.Dense) *mat.Dense {
	k, _ := W.Dims()
	var WWt mat.Dense
	WWt.Mul(W, W.T())
	data := make([]float64, k*k)
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			data[i*k+j] = 0.5 * (WWt.At(i, j) + WWt.At(j, i))
		}
	}
	sym := mat.NewSymDense(k, data)
	var eig mat.EigenSym
	if ok := eig.Factorize(sym, true); !ok {
		panic("FastICA: decorrelation eigendecomposition failed")
	}
	vals := eig.Values(nil)
	var vecs mat.Dense
	eig.VectorsTo(&vecs)
	// Build D^{-1/2}.
	dInv := mat.NewDense(k, k, nil)
	for i := 0; i < k; i++ {
		v := vals[i]
		if v < 1e-12 {
			v = 1e-12
		}
		dInv.Set(i, i, 1.0/math.Sqrt(v))
	}
	// (WWt)^{-1/2} = V * D^{-1/2} * V^T
	var tmp, inv mat.Dense
	tmp.Mul(&vecs, dInv)
	inv.Mul(&tmp, vecs.T())
	var out mat.Dense
	out.Mul(&inv, W)
	// Wrap result into new Dense.
	r, c := out.Dims()
	d := mat.NewDense(r, c, nil)
	d.Copy(&out)
	return d
}

// Transform projects new data into the independent-component space.
func (m *FastICA) Transform(X [][]float64) [][]float64 {
	n := len(X)
	d := len(m.Mean)
	Xc := mat.NewDense(d, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			Xc.Set(j, i, X[i][j]-m.Mean[j])
		}
	}
	var Z mat.Dense
	Z.Mul(m.Whitening, Xc)
	var S mat.Dense
	S.Mul(m.Components, &Z)
	K, _ := S.Dims()
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, K)
		for k := 0; k < K; k++ {
			out[i][k] = S.At(k, i)
		}
	}
	return out
}

// FitTransform fits then projects.
func (m *FastICA) FitTransform(X [][]float64) [][]float64 {
	m.Fit(X)
	return m.Transform(X)
}
