// Small CNN classifier on synthetic 1x8x8 images. Demonstrates Conv2d,
// MaxPool2d, AdaptiveAvgPool2d, and end-to-end training.
package main

import (
	"fmt"
	"math/rand"

	"gonn/nn"
	"gonn/optim"
	"gonn/tensor"
)

func main() {
	rand.Seed(3)

	X, Y := makeImgs(120, 3) // 3 classes
	conv1 := nn.NewConv2d(1, 8, 3, 1, 1, true)
	pool1 := &nn.MaxPool2d{KH: 2, KW: 2, StrideH: 2, StrideW: 2}
	conv2 := nn.NewConv2d(8, 16, 3, 1, 1, true)
	gap := &nn.AdaptiveAvgPool2d{OutH: 1, OutW: 1}
	head := nn.NewLinear(16, 3, true)

	params := []*tensor.Tensor{}
	params = append(params, conv1.Parameters()...)
	params = append(params, conv2.Parameters()...)
	params = append(params, head.Parameters()...)
	opt := optim.NewAdam(params, 5e-3)

	for epoch := 0; epoch < 40; epoch++ {
		opt.ZeroGrad()
		x := conv1.Forward(X).ReLU()
		x = pool1.Forward(x)
		x = conv2.Forward(x).ReLU()
		x = gap.Forward(x)         // (N, 16, 1, 1)
		x = x.Reshape(x.Shape[0], 16)
		logits := head.Forward(x)  // (N, 3)
		loss := nn.CrossEntropyLoss(logits, Y)
		loss.Backward()
		opt.Step()
		if epoch%5 == 0 {
			acc := accuracy(logits, Y)
			fmt.Printf("epoch %2d  loss=%.4f  acc=%.2f%%\n", epoch, loss.Data[0], acc*100)
		}
	}
}

func makeImgs(n, classes int) (*tensor.Tensor, *tensor.Tensor) {
	x := make([]float64, n*1*8*8)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		c := i % classes
		// each class places a bright square in a different corner
		for r := 0; r < 8; r++ {
			for col := 0; col < 8; col++ {
				idx := i*64 + r*8 + col
				x[idx] = rand.NormFloat64() * 0.1
				inCorner := false
				switch c {
				case 0:
					inCorner = r < 3 && col < 3
				case 1:
					inCorner = r >= 5 && col >= 5
				case 2:
					inCorner = r < 3 && col >= 5
				}
				if inCorner {
					x[idx] += 2.0
				}
			}
		}
		y[i] = float64(c)
	}
	return tensor.New(x, n, 1, 8, 8), tensor.New(y, n)
}

func accuracy(logits, targets *tensor.Tensor) float64 {
	preds := logits.ArgMax(1)
	c := 0
	for i, p := range preds.Data {
		if int(p) == int(targets.Data[i]) {
			c++
		}
	}
	return float64(c) / float64(len(targets.Data))
}
