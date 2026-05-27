// Linear regression on a synthetic dataset, trained with autograd + SGD.
package main

import (
	"fmt"
	"math/rand"

	"gonn/optim"
	"gonn/tensor"
)

func main() {
	rand.Seed(42)

	// Generate y = 3x + 2 + noise
	const n = 200
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rand.Float64() * 10
		yData[i] = 3*xData[i] + 2 + rand.NormFloat64()*0.5
	}
	X := tensor.New(xData, n, 1)
	Y := tensor.New(yData, n, 1)

	// Parameters
	W := tensor.Randn(1, 1).SetRequiresGrad(true)
	b := tensor.Zeros(1).SetRequiresGrad(true)
	opt := optim.NewSGD([]*tensor.Tensor{W, b}, 0.01)

	for epoch := 0; epoch < 200; epoch++ {
		opt.ZeroGrad()
		pred := X.MatMul(W).Add(b)
		diff := pred.Sub(Y)
		loss := diff.Square().Mean()
		loss.Backward()
		opt.Step()
		if epoch%20 == 0 {
			fmt.Printf("epoch %3d  loss=%.4f  W=%.4f  b=%.4f\n",
				epoch, loss.Data[0], W.Data[0], b.Data[0])
		}
	}
	fmt.Printf("\nlearned: y = %.3f*x + %.3f  (truth: 3*x + 2)\n", W.Data[0], b.Data[0])
}
