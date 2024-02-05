package main

import (
	"fmt"
	"gonn/tensor"

	"gonum.org/v1/gonum/mat"
)

func main() {
	test1D()
	test2D()
	test3D()
	testAdd1D()
	testAdd2D()
	testAdd3D()
}

func test1D() {
	// Initialize 1D tensors
	xData := mat.NewVecDense(3, []float64{1.0, 2.0, 3.0})
	yData := mat.NewVecDense(3, []float64{4.0, 5.0, 6.0})

	// Perform element-wise multiplication
	ctx := &tensor.Context{}
	mulOp := &tensor.Mul{}
	result := mulOp.Forward(ctx, xData, yData).(*mat.VecDense) // Assuming Forward returns *mat.VecDense for 1D

	fmt.Println("Result of 1D multiplication:", result.RawVector().Data)
}

func test2D() {
	// Initialize 2D tensors
	xData := mat.NewDense(2, 3, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	yData := mat.NewDense(2, 3, []float64{7.0, 8.0, 9.0, 10.0, 11.0, 12.0})

	// Perform element-wise multiplication
	ctx := &tensor.Context{}
	mulOp := &tensor.Mul{}
	result := mulOp.Forward(ctx, xData, yData).(*mat.Dense) // Assuming Forward returns *mat.Dense for 2D

	fmt.Printf("Result of 2D multiplication:\n%v\n", mat.Formatted(result))
}

func test3D() {
	// Initialize 3D tensors as slices of 2D tensors
	xData := []*mat.Dense{
		mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
		mat.NewDense(2, 2, []float64{5.0, 6.0, 7.0, 8.0}),
	}
	yData := []*mat.Dense{
		mat.NewDense(2, 2, []float64{9.0, 10.0, 11.0, 12.0}),
		mat.NewDense(2, 2, []float64{13.0, 14.0, 15.0, 16.0}),
	}

	// Perform element-wise multiplication
	ctx := &tensor.Context{}
	mulOp := &tensor.Mul{}
	result := mulOp.Forward(ctx, xData, yData).([]*mat.Dense) // Assuming Forward returns []*mat.Dense for 3D

	fmt.Println("Result of 3D multiplication:")
	for i, m := range result {
		fmt.Printf("Layer %d:\n%v\n", i+1, mat.Formatted(m))
	}
}

func testAdd1D() {
	// Initialize 1D tensors
	xData := mat.NewVecDense(3, []float64{1.0, 2.0, 3.0})
	yData := mat.NewVecDense(3, []float64{4.0, 5.0, 6.0})

	// Perform element-wise addition
	ctx := &tensor.Context{}
	addOp := &tensor.Add{}
	result := addOp.Forward(ctx, xData, yData).(*mat.VecDense)

	fmt.Println("Result of 1D addition:", result.RawVector().Data)
}

func testAdd2D() {
	// Initialize 2D tensors
	xData := mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0})
	yData := mat.NewDense(2, 2, []float64{5.0, 6.0, 7.0, 8.0})

	// Perform element-wise addition
	ctx := &tensor.Context{}
	addOp := &tensor.Add{}
	result := addOp.Forward(ctx, xData, yData).(*mat.Dense)

	fmt.Println("Result of 2D addition:")
	fmt.Println(mat.Formatted(result))
}

func testAdd3D() {
	// Initialize 3D tensors as slices of 2D tensors
	xData := []*mat.Dense{
		mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0}),
		mat.NewDense(2, 2, []float64{9.0, 8.0, 7.0, 6.0}),
	}
	yData := []*mat.Dense{
		mat.NewDense(2, 2, []float64{5.0, 6.0, 7.0, 8.0}),
		mat.NewDense(2, 2, []float64{5.0, 4.0, 3.0, 2.0}),
	}

	// Perform element-wise addition
	ctx := &tensor.Context{}
	addOp := &tensor.Add{}
	result := addOp.Forward(ctx, xData, yData).([]*mat.Dense)

	fmt.Println("Result of 3D addition:")
	for i, m := range result {
		fmt.Printf("Layer %d:\n%v\n", i+1, mat.Formatted(m))
	}
}
