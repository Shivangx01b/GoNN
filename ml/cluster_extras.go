package ml

import (
	"math"
)

// DBSCAN density-based clustering. Predict labels: cluster id (>=0) or -1 (noise).
type DBSCAN struct {
	Eps        float64
	MinSamples int
	labels     []int
	Xtrain     [][]float64
}

func (m *DBSCAN) Fit(X [][]float64) {
	if m.MinSamples == 0 {
		m.MinSamples = 5
	}
	n := len(X)
	m.Xtrain = copyMatrix(X)
	m.labels = make([]int, n)
	for i := range m.labels {
		m.labels[i] = -2 // unvisited
	}
	clusterID := 0
	neighbors := func(idx int) []int {
		var nb []int
		for j := 0; j < n; j++ {
			if euclidean(X[idx], X[j]) <= m.Eps {
				nb = append(nb, j)
			}
		}
		return nb
	}
	for i := 0; i < n; i++ {
		if m.labels[i] != -2 {
			continue
		}
		nb := neighbors(i)
		if len(nb) < m.MinSamples {
			m.labels[i] = -1 // noise (may be reclassified later)
			continue
		}
		m.labels[i] = clusterID
		// Expand cluster
		seeds := append([]int(nil), nb...)
		for k := 0; k < len(seeds); k++ {
			q := seeds[k]
			if m.labels[q] == -1 {
				m.labels[q] = clusterID
			}
			if m.labels[q] != -2 {
				continue
			}
			m.labels[q] = clusterID
			qnb := neighbors(q)
			if len(qnb) >= m.MinSamples {
				seeds = append(seeds, qnb...)
			}
		}
		clusterID++
	}
}

// Predict for DBSCAN: assigns each x to the cluster of its nearest training point
// if within Eps, else -1.
func (m *DBSCAN) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		best := -1
		bd := math.Inf(1)
		for j, xt := range m.Xtrain {
			d := euclidean(x, xt)
			if d < bd {
				bd = d
				best = j
			}
		}
		if best == -1 || bd > m.Eps {
			out[i] = -1
		} else {
			out[i] = m.labels[best]
		}
	}
	return out
}

// Labels returns the cluster labels assigned during Fit.
func (m *DBSCAN) Labels() []int { return m.labels }

// AgglomerativeClustering performs bottom-up hierarchical clustering.
type AgglomerativeClustering struct {
	NClusters int
	Linkage   string // "single", "complete", "average"

	labels []int
	Xtrain [][]float64
}

func (m *AgglomerativeClustering) Fit(X [][]float64) {
	if m.Linkage == "" {
		m.Linkage = "average"
	}
	n := len(X)
	m.Xtrain = copyMatrix(X)
	// Initialize each point as its own cluster.
	clusters := make([][]int, n)
	for i := 0; i < n; i++ {
		clusters[i] = []int{i}
	}
	// Pairwise distance between current clusters.
	dist := make([][]float64, n)
	for i := 0; i < n; i++ {
		dist[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			dist[i][j] = euclidean(X[i], X[j])
		}
	}
	active := make([]bool, n)
	for i := range active {
		active[i] = true
	}
	numActive := n
	clusterDist := func(a, b int) float64 {
		ca := clusters[a]
		cb := clusters[b]
		switch m.Linkage {
		case "single":
			best := math.Inf(1)
			for _, i := range ca {
				for _, j := range cb {
					if dist[i][j] < best {
						best = dist[i][j]
					}
				}
			}
			return best
		case "complete":
			best := math.Inf(-1)
			for _, i := range ca {
				for _, j := range cb {
					if dist[i][j] > best {
						best = dist[i][j]
					}
				}
			}
			return best
		default: // average
			s := 0.0
			cnt := 0
			for _, i := range ca {
				for _, j := range cb {
					s += dist[i][j]
					cnt++
				}
			}
			return s / float64(cnt)
		}
	}
	for numActive > m.NClusters {
		bestD := math.Inf(1)
		bestA, bestB := -1, -1
		for a := 0; a < n; a++ {
			if !active[a] {
				continue
			}
			for b := a + 1; b < n; b++ {
				if !active[b] {
					continue
				}
				d := clusterDist(a, b)
				if d < bestD {
					bestD = d
					bestA = a
					bestB = b
				}
			}
		}
		if bestA == -1 {
			break
		}
		clusters[bestA] = append(clusters[bestA], clusters[bestB]...)
		clusters[bestB] = nil
		active[bestB] = false
		numActive--
	}
	m.labels = make([]int, n)
	cid := 0
	for i := 0; i < n; i++ {
		if !active[i] {
			continue
		}
		for _, p := range clusters[i] {
			m.labels[p] = cid
		}
		cid++
	}
}

// Predict assigns each new point to the cluster of its nearest training point.
func (m *AgglomerativeClustering) Predict(X [][]float64) []int {
	out := make([]int, len(X))
	for i, x := range X {
		best := 0
		bd := euclidean(x, m.Xtrain[0])
		for j := 1; j < len(m.Xtrain); j++ {
			d := euclidean(x, m.Xtrain[j])
			if d < bd {
				bd = d
				best = j
			}
		}
		out[i] = m.labels[best]
	}
	return out
}

// Labels returns the labels assigned during Fit.
func (m *AgglomerativeClustering) Labels() []int { return m.labels }
