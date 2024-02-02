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
