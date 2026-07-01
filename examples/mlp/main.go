// MLP classifier on a synthetic 2-class spiral dataset, trained with Adam.
// Demonstrates: nn.Sequential, nn.Linear, nn.ReLU, nn.CrossEntropyLoss, optim.Adam.
package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonn/nn"
	"gonn/optim"
	"gonn/tensor"
)

func main() {
	rand.Seed(7)

	X, Y := makeBlobs(300, 3, 2)
	model := nn.NewSequential(
		nn.NewLinear(2, 32, true),
		nn.ReLU(),
		nn.NewLinear(32, 16, true),
		nn.ReLU(),
		nn.NewLinear(16, 3, true),
	)
	opt := optim.NewAdam(model.Parameters(), 0.01)

	for epoch := 0; epoch < 200; epoch++ {
		opt.ZeroGrad()
		logits := model.Forward(X)
		loss := nn.CrossEntropyLoss(logits, Y)
		loss.Backward()
		opt.Step()
		if epoch%20 == 0 {
			acc := accuracy(logits, Y)
			fmt.Printf("epoch %3d  loss=%.4f  acc=%.2f%%\n", epoch, loss.Data[0], acc*100)
		}
	}
	finalLogits := model.Forward(X)
	fmt.Printf("\nfinal accuracy = %.2f%%\n", accuracy(finalLogits, Y)*100)
}

// makeBlobs returns (X, y) — X: (n, 2), y: (n,) ints as float64.
func makeBlobs(n, classes, _ int) (*tensor.Tensor, *tensor.Tensor) {
	xData := make([]float64, n*2)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		c := i % classes
		ang := float64(c) * 2 * math.Pi / float64(classes)
		cx, cy := 3*math.Cos(ang), 3*math.Sin(ang)
		xData[i*2] = cx + rand.NormFloat64()*0.5
		xData[i*2+1] = cy + rand.NormFloat64()*0.5
		yData[i] = float64(c)
	}
	return tensor.New(xData, n, 2), tensor.New(yData, n)
}

func accuracy(logits, targets *tensor.Tensor) float64 {
	preds := logits.ArgMax(1)
	correct := 0
	for i, p := range preds.Data {
		if int(p) == int(targets.Data[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(targets.Data))
}
