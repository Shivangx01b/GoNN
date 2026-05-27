package ml

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// KernelPCA performs PCA in a reproducing-kernel Hilbert space.
//
// Supported kernels: "rbf" (default), "poly", "linear".
type KernelPCA struct {
	NComponents int
	Kernel      string
	Gamma       float64
	Degree      int
	Coef0       float64

	// Training data needed to evaluate kernel on new points.
	XTrain    [][]float64
	alphas    [][]float64 // [NComponents][n] -- eigenvectors of centered Gram
	lambdas   []float64   // top eigenvalues
	rowMean   []float64   // row means of K
	totalMean float64     // total mean of K
}

func (m *KernelPCA) kernelFn(a, b []float64) float64 {
	switch m.Kernel {
	case "linear":
		return dot(a, b)
	case "poly":
		deg := m.Degree
		if deg == 0 {
			deg = 3
		}
		gamma := m.Gamma
		if gamma == 0 {
			gamma = 1.0 / float64(len(a))
		}
		return math.Pow(gamma*dot(a, b)+m.Coef0, float64(deg))
	default: // "rbf"
		gamma := m.Gamma
		if gamma == 0 {
			gamma = 1.0 / float64(len(a))
		}
		return math.Exp(-gamma * squaredEuclidean(a, b))
	}
}

func (m *KernelPCA) Fit(X [][]float64) {
	if m.Kernel == "" {
		m.Kernel = "rbf"
	}
	n, _ := shapeOf(X)
	m.XTrain = copyMatrix(X)
	// Compute Gram matrix K.
	K := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := m.kernelFn(X[i], X[j])
			K.Set(i, j, v)
			if i != j {
				K.Set(j, i, v)
			}
		}
	}
	// Center K: K_c = K - 1n K - K 1n + 1n K 1n.
	m.rowMean = make([]float64, n)
	for i := 0; i < n; i++ {
		s := 0.0
		for j := 0; j < n; j++ {
			s += K.At(i, j)
		}
		m.rowMean[i] = s / float64(n)
	}
	total := 0.0
	for _, v := range m.rowMean {
		total += v
	}
	m.totalMean = total / float64(n)
	Kc := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Kc.Set(i, j, K.At(i, j)-m.rowMean[i]-m.rowMean[j]+m.totalMean)
		}
	}
	// Symmetric eigendecomposition.
	var sym mat.SymDense
	sym.SymRankOne(mat.NewSymDense(n, nil), 0, mat.NewVecDense(n, nil)) // ensure zero
	sym.Reset()
	// Build symmetric matrix from Kc (it is symmetric by construction).
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			data[i*n+j] = 0.5 * (Kc.At(i, j) + Kc.At(j, i))
		}
	}
	S := mat.NewSymDense(n, data)
	var eig mat.EigenSym
	if ok := eig.Factorize(S, true); !ok {
		panic("KernelPCA: eigendecomposition failed")
	}
	vals := eig.Values(nil)
	var vecs mat.Dense
	eig.VectorsTo(&vecs)
	// Sort by descending eigenvalue.
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < n; i++ {
		best := i
		for j := i + 1; j < n; j++ {
			if vals[idx[j]] > vals[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
	k := m.NComponents
	if k == 0 || k > n {
		k = n
	}
	m.alphas = make([][]float64, k)
	m.lambdas = make([]float64, k)
	for c := 0; c < k; c++ {
		col := idx[c]
		lam := vals[col]
		if lam < 1e-12 {
			lam = 1e-12
		}
		m.lambdas[c] = lam
		alpha := make([]float64, n)
		// Normalize so that lambda * <alpha, alpha> = 1 (sklearn convention).
		norm := 1.0 / math.Sqrt(lam)
		for i := 0; i < n; i++ {
			alpha[i] = vecs.At(i, col) * norm
		}
		m.alphas[c] = alpha
	}
	m.NComponents = k
}

// Transform projects new data into the kernel principal-component space.
func (m *KernelPCA) Transform(X [][]float64) [][]float64 {
	nTrain := len(m.XTrain)
	out := make([][]float64, len(X))
	// We need to center the kernel evaluations the same way as during fit:
	// K_new_centered[i, j] = K(x_new_i, x_train_j) - mean_j(K(x_new_i, .))
	//                       - rowMean[j] + totalMean
	for i, x := range X {
		krow := make([]float64, nTrain)
		s := 0.0
		for j := 0; j < nTrain; j++ {
			krow[j] = m.kernelFn(x, m.XTrain[j])
			s += krow[j]
		}
		newRowMean := s / float64(nTrain)
		for j := 0; j < nTrain; j++ {
			krow[j] = krow[j] - newRowMean - m.rowMean[j] + m.totalMean
		}
		proj := make([]float64, m.NComponents)
		for c := 0; c < m.NComponents; c++ {
			proj[c] = dot(krow, m.alphas[c])
		}
		out[i] = proj
	}
	return out
}

// FitTransform is the common helper.
func (m *KernelPCA) FitTransform(X [][]float64) [][]float64 {
	m.Fit(X)
	return m.Transform(X)
}
