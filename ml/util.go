package ml

import (
	"math"
	"math/rand"
	"sort"
)

// shapeOf returns rows, cols. Panics on empty X.
func shapeOf(X [][]float64) (int, int) {
	n := len(X)
	if n == 0 {
		return 0, 0
	}
	return n, len(X[0])
}

// flatten flattens a [n][d] matrix into a row-major slice of length n*d.
func flatten(X [][]float64) []float64 {
	n, d := shapeOf(X)
	out := make([]float64, n*d)
	for i := 0; i < n; i++ {
		copy(out[i*d:(i+1)*d], X[i])
	}
	return out
}

// reshape converts a flat row-major slice into [n][d].
func reshape(data []float64, n, d int) [][]float64 {
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, d)
		copy(out[i], data[i*d:(i+1)*d])
	}
	return out
}

// euclidean returns the L2 distance between two vectors.
func euclidean(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s)
}

// squaredEuclidean returns the squared L2 distance.
func squaredEuclidean(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// argmaxF returns the index of the largest value.
func argmaxF(v []float64) int {
	idx := 0
	best := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > best {
			best = v[i]
			idx = i
		}
	}
	return idx
}

// argminF returns the index of the smallest value.
func argminF(v []float64) int {
	idx := 0
	best := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] < best {
			best = v[i]
			idx = i
		}
	}
	return idx
}

// uniqueInts returns sorted unique values of y.
func uniqueInts(y []int) []int {
	seen := map[int]struct{}{}
	for _, v := range y {
		seen[v] = struct{}{}
	}
	out := make([]int, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

// sigmoid returns 1 / (1 + exp(-x)).
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1.0 + e)
}

// softmax returns a softmax distribution over a row.
func softmax(v []float64) []float64 {
	maxv := v[0]
	for _, x := range v[1:] {
		if x > maxv {
			maxv = x
		}
	}
	out := make([]float64, len(v))
	sum := 0.0
	for i, x := range v {
		out[i] = math.Exp(x - maxv)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// dot returns the dot product.
func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// copyMatrix returns a deep copy.
func copyMatrix(X [][]float64) [][]float64 {
	out := make([][]float64, len(X))
	for i := range X {
		out[i] = make([]float64, len(X[i]))
		copy(out[i], X[i])
	}
	return out
}

// randPerm returns a permutation of [0, n) using r.
func randPerm(n int, r *rand.Rand) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	r.Shuffle(n, func(i, j int) { p[i], p[j] = p[j], p[i] })
	return p
}

// meanCol computes per-column means.
func meanCol(X [][]float64) []float64 {
	n, d := shapeOf(X)
	m := make([]float64, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			m[j] += X[i][j]
		}
	}
	for j := range m {
		m[j] /= float64(n)
	}
	return m
}
