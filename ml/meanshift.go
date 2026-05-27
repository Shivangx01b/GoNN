package ml

import "math"

// MeanShift clusters points by iteratively shifting each toward the local mean
// of a Gaussian-kernel window. Points that converge to the same mode form a
// cluster.
type MeanShift struct {
	Bandwidth float64
	MaxIter   int
	Tol       float64

	Centers [][]float64 // discovered cluster centers
	labels_ []int       // per training point cluster assignment
}

func (m *MeanShift) Fit(X [][]float64) {
	if m.MaxIter == 0 {
		m.MaxIter = 300
	}
	if m.Tol == 0 {
		m.Tol = 1e-3
	}
	if m.Bandwidth == 0 {
		m.Bandwidth = estimateBandwidth(X)
	}
	n, d := shapeOf(X)
	bw := m.Bandwidth
	bw2 := bw * bw

	shifted := make([][]float64, n)
	for i := 0; i < n; i++ {
		shifted[i] = append([]float64(nil), X[i]...)
	}

	// Run the shift independently for each point.
	for i := 0; i < n; i++ {
		cur := shifted[i]
		for iter := 0; iter < m.MaxIter; iter++ {
			num := make([]float64, d)
			den := 0.0
			for j := 0; j < n; j++ {
				dist2 := 0.0
				for k := 0; k < d; k++ {
					df := cur[k] - X[j][k]
					dist2 += df * df
				}
				w := math.Exp(-dist2 / (2 * bw2))
				den += w
				for k := 0; k < d; k++ {
					num[k] += w * X[j][k]
				}
			}
			if den == 0 {
				break
			}
			newCur := make([]float64, d)
			shift := 0.0
			for k := 0; k < d; k++ {
				newCur[k] = num[k] / den
				df := newCur[k] - cur[k]
				shift += df * df
			}
			cur = newCur
			if math.Sqrt(shift) < m.Tol {
				break
			}
		}
		shifted[i] = cur
	}

	// Merge nearby modes into cluster centers.
	mergeRadius := bw * 0.5
	m.Centers = nil
	m.labels_ = make([]int, n)
	for i := 0; i < n; i++ {
		assigned := -1
		for c, center := range m.Centers {
			if euclidean(shifted[i], center) < mergeRadius {
				assigned = c
				break
			}
		}
		if assigned == -1 {
			m.Centers = append(m.Centers, append([]float64(nil), shifted[i]...))
			assigned = len(m.Centers) - 1
		}
		m.labels_[i] = assigned
	}
}

// estimateBandwidth uses the mean pairwise distance as a reasonable default.
func estimateBandwidth(X [][]float64) float64 {
	n, _ := shapeOf(X)
	if n < 2 {
		return 1.0
	}
	// Sample at most 50 pairs.
	maxN := n
	if maxN > 50 {
		maxN = 50
	}
	sum := 0.0
	count := 0
	for i := 0; i < maxN; i++ {
		for j := i + 1; j < maxN; j++ {
			sum += euclidean(X[i], X[j])
			count++
		}
	}
	if count == 0 {
		return 1.0
	}
	return sum / float64(count) / 2.0
}

func (m *MeanShift) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		best := 0
		bd := euclidean(x, m.Centers[0])
		for c := 1; c < len(m.Centers); c++ {
			d := euclidean(x, m.Centers[c])
			if d < bd {
				bd = d
				best = c
			}
		}
		out[i] = best
	}
	return out
}
