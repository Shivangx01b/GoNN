// Demonstrates classical ML algorithms: KMeans, LinearRegression, RandomForest, PCA.
package main

import (
	"fmt"
	"math/rand"

	"gonn/ml"
)

func main() {
	rand.Seed(0)

	fmt.Println("=== Linear Regression ===")
	xs := make([][]float64, 100)
	ys := make([]float64, 100)
	for i := range xs {
		x := rand.Float64() * 10
		xs[i] = []float64{x}
		ys[i] = 2.5*x + 1.7 + rand.NormFloat64()*0.3
	}
	lr := &ml.LinearRegression{}
	lr.Fit(xs, ys)
	fmt.Printf("  learned: y = %.3f*x + %.3f  (truth: 2.5*x + 1.7)\n", lr.Weights[0], lr.Bias)

	fmt.Println("\n=== KMeans ===")
	X := makeBlobs(200, 3)
	km := &ml.KMeans{K: 3, MaxIter: 100, Seed: 1}
	km.Fit(X)
	labels := km.Predict(X)
	fmt.Printf("  fit %d points into %d clusters\n", len(labels), len(km.Centroids))
	for i, c := range km.Centroids {
		fmt.Printf("  centroid[%d] = [%.2f, %.2f]\n", i, c[0], c[1])
	}

	fmt.Println("\n=== PCA ===")
	Xpca := makeBlobs(100, 2)
	pca := &ml.PCA{NComponents: 1}
	pca.Fit(Xpca)
	red := pca.Transform(Xpca)
	fmt.Printf("  reduced %dx%d -> %dx%d\n",
		len(Xpca), len(Xpca[0]), len(red), len(red[0]))
}

func makeBlobs(n, centers int) [][]float64 {
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		c := i % centers
		cx := float64(c)*5 - 5
		cy := float64(c)*3 - 3
		out[i] = []float64{
			cx + rand.NormFloat64()*0.5,
			cy + rand.NormFloat64()*0.5,
		}
	}
	return out
}
