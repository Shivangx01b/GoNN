package ml

import (
	"math"
	"sort"
)

// Accuracy returns fraction of correct predictions.
func Accuracy(yTrue, yPred []int) float64 {
	if len(yTrue) == 0 {
		return 0
	}
	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue))
}

// ConfusionMatrix returns an nClasses x nClasses matrix (rows = true, cols = pred).
func ConfusionMatrix(yTrue, yPred []int, nClasses int) [][]int {
	cm := make([][]int, nClasses)
	for i := range cm {
		cm[i] = make([]int, nClasses)
	}
	for i := range yTrue {
		if yTrue[i] >= 0 && yTrue[i] < nClasses && yPred[i] >= 0 && yPred[i] < nClasses {
			cm[yTrue[i]][yPred[i]]++
		}
	}
	return cm
}

func uniqueClasses(yTrue, yPred []int) []int {
	seen := map[int]struct{}{}
	for _, v := range yTrue {
		seen[v] = struct{}{}
	}
	for _, v := range yPred {
		seen[v] = struct{}{}
	}
	out := make([]int, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

// perClassStats returns tp, fp, fn, support per class label list.
func perClassStats(yTrue, yPred []int) (classes []int, tp, fp, fn, support []int) {
	classes = uniqueClasses(yTrue, yPred)
	K := len(classes)
	tp = make([]int, K)
	fp = make([]int, K)
	fn = make([]int, K)
	support = make([]int, K)
	idx := map[int]int{}
	for i, c := range classes {
		idx[c] = i
	}
	for i := range yTrue {
		t := idx[yTrue[i]]
		p := idx[yPred[i]]
		support[t]++
		if t == p {
			tp[t]++
		} else {
			fp[p]++
			fn[t]++
		}
	}
	return
}

// Precision returns precision; average = "macro" or "weighted".
func Precision(yTrue, yPred []int, average string) float64 {
	_, tp, fp, _, support := perClassStats(yTrue, yPred)
	K := len(tp)
	prec := make([]float64, K)
	for i := 0; i < K; i++ {
		if tp[i]+fp[i] > 0 {
			prec[i] = float64(tp[i]) / float64(tp[i]+fp[i])
		}
	}
	return aggregate(prec, support, average)
}

// Recall returns recall.
func Recall(yTrue, yPred []int, average string) float64 {
	_, tp, _, fn, support := perClassStats(yTrue, yPred)
	K := len(tp)
	rec := make([]float64, K)
	for i := 0; i < K; i++ {
		if tp[i]+fn[i] > 0 {
			rec[i] = float64(tp[i]) / float64(tp[i]+fn[i])
		}
	}
	return aggregate(rec, support, average)
}

// F1 returns F1 score.
func F1(yTrue, yPred []int, average string) float64 {
	_, tp, fp, fn, support := perClassStats(yTrue, yPred)
	K := len(tp)
	f := make([]float64, K)
	for i := 0; i < K; i++ {
		p := 0.0
		if tp[i]+fp[i] > 0 {
			p = float64(tp[i]) / float64(tp[i]+fp[i])
		}
		r := 0.0
		if tp[i]+fn[i] > 0 {
			r = float64(tp[i]) / float64(tp[i]+fn[i])
		}
		if p+r > 0 {
			f[i] = 2 * p * r / (p + r)
		}
	}
	return aggregate(f, support, average)
}

func aggregate(vals []float64, support []int, average string) float64 {
	if len(vals) == 0 {
		return 0
	}
	if average == "weighted" {
		total := 0.0
		w := 0.0
		for i, v := range vals {
			total += v * float64(support[i])
			w += float64(support[i])
		}
		if w == 0 {
			return 0
		}
		return total / w
	}
	// macro
	s := 0.0
	for _, v := range vals {
		s += v
	}
	return s / float64(len(vals))
}

// MeanSquaredError returns the average squared error.
func MeanSquaredError(yTrue, yPred []float64) float64 {
	if len(yTrue) == 0 {
		return 0
	}
	s := 0.0
	for i := range yTrue {
		d := yTrue[i] - yPred[i]
		s += d * d
	}
	return s / float64(len(yTrue))
}

// MeanAbsoluteError returns the average absolute error.
func MeanAbsoluteError(yTrue, yPred []float64) float64 {
	if len(yTrue) == 0 {
		return 0
	}
	s := 0.0
	for i := range yTrue {
		s += math.Abs(yTrue[i] - yPred[i])
	}
	return s / float64(len(yTrue))
}

// R2Score returns the coefficient of determination.
func R2Score(yTrue, yPred []float64) float64 {
	if len(yTrue) == 0 {
		return 0
	}
	mean := 0.0
	for _, v := range yTrue {
		mean += v
	}
	mean /= float64(len(yTrue))
	ssRes, ssTot := 0.0, 0.0
	for i := range yTrue {
		d := yTrue[i] - yPred[i]
		ssRes += d * d
		m := yTrue[i] - mean
		ssTot += m * m
	}
	if ssTot == 0 {
		return 0
	}
	return 1 - ssRes/ssTot
}

// SilhouetteScore returns the mean silhouette coefficient.
func SilhouetteScore(X [][]float64, labels []int) float64 {
	n := len(X)
	if n == 0 {
		return 0
	}
	// Group indices by label.
	groups := map[int][]int{}
	for i, l := range labels {
		groups[l] = append(groups[l], i)
	}
	if len(groups) < 2 {
		return 0
	}
	total := 0.0
	for i := 0; i < n; i++ {
		own := groups[labels[i]]
		if len(own) <= 1 {
			continue
		}
		a := 0.0
		for _, j := range own {
			if j == i {
				continue
			}
			a += euclidean(X[i], X[j])
		}
		a /= float64(len(own) - 1)
		b := math.Inf(1)
		for lbl, members := range groups {
			if lbl == labels[i] {
				continue
			}
			s := 0.0
			for _, j := range members {
				s += euclidean(X[i], X[j])
			}
			avg := s / float64(len(members))
			if avg < b {
				b = avg
			}
		}
		denom := math.Max(a, b)
		if denom > 0 {
			total += (b - a) / denom
		}
	}
	return total / float64(n)
}

// ROCAUC returns the binary ROC AUC for scores. yTrue must be 0/1.
func ROCAUC(yTrue []int, yScore []float64) float64 {
	n := len(yTrue)
	type pair struct {
		s float64
		y int
	}
	ps := make([]pair, n)
	for i := range ps {
		ps[i] = pair{yScore[i], yTrue[i]}
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].s > ps[j].s })
	var nPos, nNeg int
	for _, p := range ps {
		if p.y == 1 {
			nPos++
		} else {
			nNeg++
		}
	}
	if nPos == 0 || nNeg == 0 {
		return 0
	}
	// Sum of ranks of positives (using average ranks for ties).
	rankSum := 0.0
	i := 0
	for i < n {
		j := i
		for j < n && ps[j].s == ps[i].s {
			j++
		}
		// Ranks i+1 .. j (1-based). Average = ((i+1)+j)/2.
		avgRank := float64(i+1+j) / 2.0
		for k := i; k < j; k++ {
			if ps[k].y == 1 {
				rankSum += avgRank
			}
		}
		i = j
	}
	// We sorted descending by score, but rank convention is ascending.
	// To get ascending ranks, reverse: rankAsc = (n+1) - rankDesc.
	// Equivalent transformation:
	rankSumAsc := float64(nPos)*float64(n+1) - rankSum
	auc := (rankSumAsc - float64(nPos)*float64(nPos+1)/2.0) / (float64(nPos) * float64(nNeg))
	return auc
}
