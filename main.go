// GoNN entry point. Demonstrates the core Tensor API with autograd.
// For richer examples see examples/.
package main

import (
	"fmt"

	"gonn/tensor"
)

func main() {
	fmt.Println("GoNN — Go neural-network framework")
	fmt.Println("==================================")
	fmt.Println()

	demoBasicOps()
	demoAutograd()
	demoMatmul()
	demoActivations()
}

func demoBasicOps() {
	fmt.Println("[1] Basic ops")
	x := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	y := tensor.New([]float64{7, 8, 9, 10, 11, 12}, 2, 3)
	fmt.Printf("  x = %v\n", x)
	fmt.Printf("  y = %v\n", y)
	fmt.Printf("  x + y = %v\n", x.Add(y))
	fmt.Printf("  x * y = %v\n", x.Mul(y))
	fmt.Printf("  sum(x)= %v\n", x.Sum())
	fmt.Println()
}

func demoAutograd() {
	fmt.Println("[2] Autograd: d/dx [sum((Wx)^2)] for x = [1,2,3]")
	x := tensor.New([]float64{1, 2, 3}, 3, 1).SetRequiresGrad(true)
	W := tensor.New([]float64{2, -1, 0.5}, 1, 3).SetRequiresGrad(true)
	y := W.MatMul(x).Square().Sum()
	y.Backward()
	fmt.Printf("  y = %v\n", y)
	fmt.Printf("  dy/dx = %v\n", x.Grad)
	fmt.Printf("  dy/dW = %v\n", W.Grad)
	fmt.Println()
}

func demoMatmul() {
	fmt.Println("[3] MatMul")
	A := tensor.New([]float64{1, 2, 3, 4}, 2, 2)
	B := tensor.New([]float64{5, 6, 7, 8}, 2, 2)
	fmt.Printf("  A @ B = %v\n", A.MatMul(B))
	fmt.Println()
}

func demoActivations() {
	fmt.Println("[4] Activations")
	z := tensor.New([]float64{-2, -1, 0, 1, 2}, 5)
	fmt.Printf("  z         = %v\n", z)
	fmt.Printf("  ReLU(z)   = %v\n", z.ReLU())
	fmt.Printf("  Sigmoid(z)= %v\n", z.Sigmoid())
	fmt.Printf("  Tanh(z)   = %v\n", z.Tanh())
	fmt.Printf("  GELU(z)   = %v\n", z.GELU())
	fmt.Printf("  Softmax(z)= %v\n", z.Softmax(0))
	fmt.Println()
}
