package ml

import (
	"math"
	"sort"
)

// StandardScaler scales features to zero mean and unit variance.
type StandardScaler struct {
	Mean []float64
	Std  []float64
}

func (s *StandardScaler) Fit(X [][]float64) {
	n, d := shapeOf(X)
	s.Mean = make([]float64, d)
	s.Std = make([]float64, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			s.Mean[j] += X[i][j]
		}
	}
	for j := 0; j < d; j++ {
		s.Mean[j] /= float64(n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			diff := X[i][j] - s.Mean[j]
			s.Std[j] += diff * diff
		}
	}
	for j := 0; j < d; j++ {
		s.Std[j] = math.Sqrt(s.Std[j] / float64(n))
		if s.Std[j] == 0 {
			s.Std[j] = 1
		}
	}
}

func (s *StandardScaler) Transform(X [][]float64) [][]float64 {
	n, d := shapeOf(X)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			out[i][j] = (X[i][j] - s.Mean[j]) / s.Std[j]
		}
	}
	return out
}

func (s *StandardScaler) FitTransform(X [][]float64) [][]float64 {
	s.Fit(X)
	return s.Transform(X)
}

// MinMaxScaler scales features to [0, 1].
type MinMaxScaler struct {
	Min []float64
	Max []float64
}

func (s *MinMaxScaler) Fit(X [][]float64) {
	n, d := shapeOf(X)
	s.Min = make([]float64, d)
	s.Max = make([]float64, d)
	for j := 0; j < d; j++ {
		s.Min[j] = math.Inf(1)
		s.Max[j] = math.Inf(-1)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			if X[i][j] < s.Min[j] {
				s.Min[j] = X[i][j]
			}
			if X[i][j] > s.Max[j] {
				s.Max[j] = X[i][j]
			}
		}
	}
}

func (s *MinMaxScaler) Transform(X [][]float64) [][]float64 {
	n, d := shapeOf(X)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			rng := s.Max[j] - s.Min[j]
			if rng == 0 {
				out[i][j] = 0
			} else {
				out[i][j] = (X[i][j] - s.Min[j]) / rng
			}
		}
	}
	return out
}

func (s *MinMaxScaler) FitTransform(X [][]float64) [][]float64 {
	s.Fit(X)
	return s.Transform(X)
}

// OneHotEncoder one-hot encodes integer labels.
type OneHotEncoder struct {
	Classes []int
	classIdx map[int]int
}

func (e *OneHotEncoder) Fit(y []int) {
	e.Classes = uniqueInts(y)
	e.classIdx = map[int]int{}
	for i, c := range e.Classes {
		e.classIdx[c] = i
	}
}

func (e *OneHotEncoder) Transform(y []int) [][]float64 {
	out := make([][]float64, len(y))
	K := len(e.Classes)
	for i, v := range y {
		out[i] = make([]float64, K)
		if idx, ok := e.classIdx[v]; ok {
			out[i][idx] = 1
		}
	}
	return out
}

func (e *OneHotEncoder) FitTransform(y []int) [][]float64 {
	e.Fit(y)
	return e.Transform(y)
}

// LabelEncoder maps labels to integers [0, K).
type LabelEncoder struct {
	Classes []int
	classIdx map[int]int
}

func (e *LabelEncoder) Fit(y []int) {
	e.Classes = uniqueInts(y)
	e.classIdx = map[int]int{}
	for i, c := range e.Classes {
		e.classIdx[c] = i
	}
}

func (e *LabelEncoder) Transform(y []int) []int {
	out := make([]int, len(y))
	for i, v := range y {
		out[i] = e.classIdx[v]
	}
	return out
}

func (e *LabelEncoder) InverseTransform(y []int) []int {
	out := make([]int, len(y))
	for i, v := range y {
		if v >= 0 && v < len(e.Classes) {
			out[i] = e.Classes[v]
		}
	}
	return out
}

// PolynomialFeatures generates polynomial combinations up to Degree (1..3).
// Includes the bias term (1) and all monomials with total degree <= Degree.
type PolynomialFeatures struct {
	Degree int
	// combos lists feature-index multisets defining each output column.
	combos [][]int
	dIn    int
}

func (p *PolynomialFeatures) Fit(X [][]float64) {
	_, d := shapeOf(X)
	p.dIn = d
	if p.Degree <= 0 {
		p.Degree = 2
	}
	p.combos = genCombos(d, p.Degree)
}

// genCombos returns sorted index multisets of size 0..deg from feature alphabet of size d.
func genCombos(d, deg int) [][]int {
	var out [][]int
	out = append(out, []int{}) // bias (degree 0)
	var rec func(prefix []int, start, remaining int)
	rec = func(prefix []int, start, remaining int) {
		if remaining == 0 {
			return
		}
		for i := start; i < d; i++ {
			next := append(append([]int(nil), prefix...), i)
			out = append(out, next)
			rec(next, i, remaining-1)
		}
	}
	rec(nil, 0, deg)
	// Sort by length then lex
	sort.SliceStable(out, func(i, j int) bool {
		if len(out[i]) != len(out[j]) {
			return len(out[i]) < len(out[j])
		}
		for k := 0; k < len(out[i]); k++ {
			if out[i][k] != out[j][k] {
				return out[i][k] < out[j][k]
			}
		}
		return false
	})
	return out
}

func (p *PolynomialFeatures) Transform(X [][]float64) [][]float64 {
	n := len(X)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, len(p.combos))
		for k, combo := range p.combos {
			v := 1.0
			for _, idx := range combo {
				v *= X[i][idx]
			}
			out[i][k] = v
		}
	}
	return out
}

func (p *PolynomialFeatures) FitTransform(X [][]float64) [][]float64 {
	p.Fit(X)
	return p.Transform(X)
}
