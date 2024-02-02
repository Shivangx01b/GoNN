# GoNN
A go based neural network for basic stuffs

## What is GONN ?

GoNN is Go Lang based Neural Network for deep learning (work in progress!). This project does not compete with any exsisting neural network library which is world famouse like Pytroch, Tensorflow etc; but still it's a deep learning framework in future (I feel so ?)

## Example usage

- Using GoNN
```go
package main

import (
	"fmt"
	"gonn/tensor"
)

type Tensor struct {
	data []float64
}

func main() {
	// Example of how to create a tensor and perform an operation
	xData := []float64{1.0, 2.0, 3.0}
	yData := []float64{4.0, 5.0, 6.0}
	xTensor := Tensor{data: xData}
	yTensor := Tensor{data: yData}
	mul := tensor.Mul{}
	ctx := tensor.Context{}
	result := mul.Forward(&ctx, xTensor.data, yTensor.data)
	fmt.Println("Result of multiplication:", result)
}
```

### TODO 
1) Impelement Gonum for matrix multiplication
2) Impelemnt 2D and 3D vectors
3) Better Backwards capability (looks like shit for now)
