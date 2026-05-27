package ml

import (
	"math"
	"math/rand"
)

// KMeans performs k-means clustering with k-means++ initialization.
type KMeans struct {
	K         int
	MaxIter   int
	Tol       float64
	Seed      int64
	Centroids [][]float64
}

func (m *KMeans) defaults() {
	if m.MaxIter == 0 {
		m.MaxIter = 300
	}
	if m.Tol == 0 {
		m.Tol = 1e-4
	}
}

// initPlusPlus performs k-means++ centroid initialization.
func (m *KMeans) initPlusPlus(X [][]float64, r *rand.Rand) [][]float64 {
	n, d := shapeOf(X)
	centroids := make([][]float64, m.K)
	first := r.Intn(n)
	centroids[0] = append([]float64(nil), X[first]...)
	dist2 := make([]float64, n)
	for i := 0; i < n; i++ {
		dist2[i] = squaredEuclidean(X[i], centroids[0])
	}
	for c := 1; c < m.K; c++ {
		sum := 0.0
		for _, v := range dist2 {
			sum += v
		}
		var chosen int
		if sum == 0 {
			chosen = r.Intn(n)
		} else {
			target := r.Float64() * sum
			acc := 0.0
			for i, v := range dist2 {
				acc += v
				if acc >= target {
					chosen = i
					break
				}
			}
		}
		centroids[c] = append([]float64(nil), X[chosen]...)
		for i := 0; i < n; i++ {
			d2 := squaredEuclidean(X[i], centroids[c])
			if d2 < dist2[i] {
				dist2[i] = d2
			}
		}
	}
	_ = d
	return centroids
}

func (m *KMeans) Fit(X [][]float64) {
	m.defaults()
	if m.K <= 0 {
		panic("KMeans: K must be > 0")
	}
	n, d := shapeOf(X)
	r := rand.New(rand.NewSource(m.Seed))
	m.Centroids = m.initPlusPlus(X, r)
	labels := make([]int, n)
	for it := 0; it < m.MaxIter; it++ {
		// Assign
		changed := false
		for i := 0; i < n; i++ {
			best := 0
			bd := squaredEuclidean(X[i], m.Centroids[0])
			for c := 1; c < m.K; c++ {
				v := squaredEuclidean(X[i], m.Centroids[c])
				if v < bd {
					bd = v
					best = c
				}
			}
			if labels[i] != best {
				labels[i] = best
				changed = true
			}
		}
		// Update
		sums := make([][]float64, m.K)
		counts := make([]int, m.K)
		for c := 0; c < m.K; c++ {
			sums[c] = make([]float64, d)
		}
		for i := 0; i < n; i++ {
			c := labels[i]
			counts[c]++
			for j := 0; j < d; j++ {
				sums[c][j] += X[i][j]
			}
		}
		shift := 0.0
		for c := 0; c < m.K; c++ {
			if counts[c] == 0 {
				// Reinit empty cluster to a random point.
				m.Centroids[c] = append([]float64(nil), X[r.Intn(n)]...)
				continue
			}
			newC := make([]float64, d)
			for j := 0; j < d; j++ {
				newC[j] = sums[c][j] / float64(counts[c])
			}
			shift += euclidean(newC, m.Centroids[c])
			m.Centroids[c] = newC
		}
		if !changed || shift < m.Tol {
			break
		}
	}
	_ = math.Inf
}

func (m *KMeans) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		best := 0
		bd := squaredEuclidean(x, m.Centroids[0])
		for c := 1; c < len(m.Centroids); c++ {
			v := squaredEuclidean(x, m.Centroids[c])
			if v < bd {
				bd = v
				best = c
			}
		}
		out[i] = best
	}
	return out
}
