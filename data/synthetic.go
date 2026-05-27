package data

import (
	"math"
	"math/rand"

	"gonn/tensor"
)

// MakeRegression generates a synthetic linear regression dataset:
//
//	y = X @ w + b + noise * eps
//
// where X ~ N(0, 1), w ~ N(0, 1), b ~ N(0, 1), and eps ~ N(0, 1). Returned
// shapes are X(nSamples, nFeatures) and y(nSamples,).
func MakeRegression(nSamples, nFeatures int, noise float64, seed int64) (X, y *tensor.Tensor) {
	if nSamples <= 0 || nFeatures <= 0 {
		panic("data.MakeRegression: nSamples and nFeatures must be > 0")
	}
	r := newRand(seed)

	xData := make([]float64, nSamples*nFeatures)
	for i := range xData {
		xData[i] = r.NormFloat64()
	}
	w := make([]float64, nFeatures)
	for i := range w {
		w[i] = r.NormFloat64()
	}
	b := r.NormFloat64()

	yData := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		s := b
		base := i * nFeatures
		for j := 0; j < nFeatures; j++ {
			s += xData[base+j] * w[j]
		}
		if noise > 0 {
			s += noise * r.NormFloat64()
		}
		yData[i] = s
	}

	return tensor.New(xData, nSamples, nFeatures), tensor.New(yData, nSamples)
}

// MakeClassification generates a synthetic classification dataset by
// drawing nClasses centroids from N(0, I) (scaled by a fixed separation)
// and sampling nSamples points around them with isotropic Gaussian noise.
// Returned shapes are X(nSamples, nFeatures) and y(nSamples,) where
// labels are float64 class indices in [0, nClasses).
func MakeClassification(nSamples, nFeatures, nClasses int, seed int64) (X, y *tensor.Tensor) {
	if nSamples <= 0 || nFeatures <= 0 || nClasses <= 0 {
		panic("data.MakeClassification: nSamples, nFeatures, nClasses must be > 0")
	}
	const sep = 3.0
	r := newRand(seed)

	centers := make([]float64, nClasses*nFeatures)
	for i := range centers {
		centers[i] = sep * r.NormFloat64()
	}

	xData := make([]float64, nSamples*nFeatures)
	yData := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		c := r.Intn(nClasses)
		yData[i] = float64(c)
		cBase := c * nFeatures
		xBase := i * nFeatures
		for j := 0; j < nFeatures; j++ {
			xData[xBase+j] = centers[cBase+j] + r.NormFloat64()
		}
	}

	return tensor.New(xData, nSamples, nFeatures), tensor.New(yData, nSamples)
}

// MakeBlobs generates isotropic Gaussian blobs around nCenters randomly
// drawn cluster centers. Samples are distributed as evenly as possible
// across centers. Returned shapes are X(nSamples, nFeatures) and
// y(nSamples,) with float64 cluster indices.
func MakeBlobs(nSamples, nFeatures, nCenters int, seed int64) (X, y *tensor.Tensor) {
	if nSamples <= 0 || nFeatures <= 0 || nCenters <= 0 {
		panic("data.MakeBlobs: nSamples, nFeatures, nCenters must be > 0")
	}
	const sep = 10.0
	const clusterStd = 1.0
	r := newRand(seed)

	centers := make([]float64, nCenters*nFeatures)
	for i := range centers {
		centers[i] = sep * r.Float64()
	}

	xData := make([]float64, nSamples*nFeatures)
	yData := make([]float64, nSamples)
	per := nSamples / nCenters
	rem := nSamples % nCenters

	idx := 0
	for c := 0; c < nCenters; c++ {
		count := per
		if c < rem {
			count++
		}
		cBase := c * nFeatures
		for k := 0; k < count; k++ {
			xBase := idx * nFeatures
			for j := 0; j < nFeatures; j++ {
				xData[xBase+j] = centers[cBase+j] + clusterStd*r.NormFloat64()
			}
			yData[idx] = float64(c)
			idx++
		}
	}

	return tensor.New(xData, nSamples, nFeatures), tensor.New(yData, nSamples)
}

// MakeMoons generates the classic two-moons 2-D classification dataset.
// nSamples is split evenly between the two classes (with the second class
// taking the remainder when odd). Returned shapes are X(nSamples, 2) and
// y(nSamples,).
func MakeMoons(nSamples int, noise float64, seed int64) (X, y *tensor.Tensor) {
	if nSamples <= 0 {
		panic("data.MakeMoons: nSamples must be > 0")
	}
	r := newRand(seed)

	nA := nSamples / 2
	nB := nSamples - nA

	xData := make([]float64, nSamples*2)
	yData := make([]float64, nSamples)

	for i := 0; i < nA; i++ {
		theta := math.Pi * float64(i) / float64(maxInt(nA-1, 1))
		x := math.Cos(theta)
		yv := math.Sin(theta)
		if noise > 0 {
			x += noise * r.NormFloat64()
			yv += noise * r.NormFloat64()
		}
		xData[2*i] = x
		xData[2*i+1] = yv
		yData[i] = 0
	}
	for i := 0; i < nB; i++ {
		theta := math.Pi * float64(i) / float64(maxInt(nB-1, 1))
		x := 1.0 - math.Cos(theta)
		yv := 0.5 - math.Sin(theta)
		if noise > 0 {
			x += noise * r.NormFloat64()
			yv += noise * r.NormFloat64()
		}
		xData[2*(nA+i)] = x
		xData[2*(nA+i)+1] = yv
		yData[nA+i] = 1
	}

	return tensor.New(xData, nSamples, 2), tensor.New(yData, nSamples)
}

func newRand(seed int64) *rand.Rand {
	if seed == 0 {
		return rand.New(rand.NewSource(rand.Int63()))
	}
	return rand.New(rand.NewSource(seed))
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
