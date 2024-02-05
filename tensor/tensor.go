package tensor

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Context struct to hold operation context
type Context struct {
	parents      []*Tensor
	savedTensors []interface{} // Interface type to save any tensor dimension
}

// Save tensors for backward pass
func (c *Context) SaveForBackward(tensors ...interface{}) {
	c.savedTensors = append(c.savedTensors, tensors...)
}

// Tensor struct represents a tensor
type Tensor struct {
	data  interface{} // Can be *mat.VecDense, *mat.Dense, or []*mat.Dense for 1D, 2D, or 3D tensors respectively
	grad  interface{} // Same as data
	shape []int       // Shape of the tensor
	ctx   *Context
}

// Function interface for operations
type Function interface {
	Forward(ctx *Context, inputs ...interface{}) interface{}
	Backward(ctx *Context, gradOutput interface{}) []interface{}
}

// Mul operation for element-wise multiplication
type Mul struct{}

func (m *Mul) Forward(ctx *Context, inputs ...interface{}) interface{} {
	x, y := inputs[0], inputs[1]
	ctx.SaveForBackward(x, y)

	switch xt := x.(type) {
	case *mat.VecDense:
		yt := y.(*mat.VecDense)
		result := mat.NewVecDense(xt.Len(), nil)
		for i := 0; i < xt.Len(); i++ {
			result.SetVec(i, xt.AtVec(i)*yt.AtVec(i))
		}
		return result
	case *mat.Dense:
		yt := y.(*mat.Dense)
		r, c := xt.Dims()
		result := mat.NewDense(r, c, nil)
		result.MulElem(xt, yt)
		return result
	case []*mat.Dense:
		yt := y.([]*mat.Dense)
		result := make([]*mat.Dense, len(xt))
		for i := range xt {
			r, c := xt[i].Dims()
			result[i] = mat.NewDense(r, c, nil)
			result[i].MulElem(xt[i], yt[i])
		}
		return result
	default:
		panic("unsupported tensor type")
	}
}

func (m *Mul) Backward(ctx *Context, gradOutput interface{}) []interface{} {
	savedTensors := ctx.savedTensors
	x, y := savedTensors[0], savedTensors[1]

	// Determine the type of tensors to apply correct differentiation logic
	switch xt := x.(type) {
	case *mat.VecDense:
		yt := y.(*mat.VecDense)
		gradOutputVec := gradOutput.(*mat.VecDense)

		// Calculate gradients for 1D tensors
		gradX := mat.NewVecDense(xt.Len(), nil)
		gradY := mat.NewVecDense(yt.Len(), nil)
		for i := 0; i < xt.Len(); i++ {
			gradX.SetVec(i, yt.AtVec(i)*gradOutputVec.AtVec(i))
			gradY.SetVec(i, xt.AtVec(i)*gradOutputVec.AtVec(i))
		}
		return []interface{}{gradX, gradY}

	case *mat.Dense:
		yt := y.(*mat.Dense)
		gradOutputDense := gradOutput.(*mat.Dense)

		// Calculate gradients for 2D tensors
		r, c := xt.Dims()
		gradX := mat.NewDense(r, c, nil)
		gradY := mat.NewDense(r, c, nil)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				gradX.Set(i, j, yt.At(i, j)*gradOutputDense.At(i, j))
				gradY.Set(i, j, xt.At(i, j)*gradOutputDense.At(i, j))
			}
		}
		return []interface{}{gradX, gradY}

	case []*mat.Dense:
		yt := y.([]*mat.Dense)
		gradOutputDense := gradOutput.([]*mat.Dense)

		// Calculate gradients for 3D tensors
		gradX := make([]*mat.Dense, len(xt))
		gradY := make([]*mat.Dense, len(yt))
		for k := range xt {
			r, c := xt[k].Dims()
			gradX[k] = mat.NewDense(r, c, nil)
			gradY[k] = mat.NewDense(r, c, nil)
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					gradX[k].Set(i, j, yt[k].At(i, j)*gradOutputDense[k].At(i, j))
					gradY[k].Set(i, j, xt[k].At(i, j)*gradOutputDense[k].At(i, j))
				}
			}
		}
		return []interface{}{gradX, gradY}

	default:
		panic("unsupported tensor type in backward pass")
	}
}

// Add operation supports 1D, 2D, and 3D tensors
type Add struct{}

func (a *Add) Forward(ctx *Context, inputs ...interface{}) interface{} {
	x, y := inputs[0], inputs[1]

	// Handle 1D tensors
	if xv, ok := x.(*mat.VecDense); ok {
		yv := y.(*mat.VecDense)
		result := mat.NewVecDense(xv.Len(), nil)
		result.AddVec(xv, yv)
		return result
	}

	// Handle 2D tensors
	if xd, ok := x.(*mat.Dense); ok {
		yd := y.(*mat.Dense)
		r, c := xd.Dims()                 // Capture the number of rows and columns separately
		result := mat.NewDense(r, c, nil) // Use the separate row and column counts here
		result.Add(xd, yd)
		return result
	}

	// Handle 3D tensors as slices of *mat.Dense
	if x3d, ok := x.([]*mat.Dense); ok {
		y3d := y.([]*mat.Dense)
		result := make([]*mat.Dense, len(x3d))
		for i, xd := range x3d {
			r, c := xd.Dims()                   // Capture the dimensions of the current 2D tensor
			result[i] = mat.NewDense(r, c, nil) // Initialize a new *mat.Dense with the correct dimensions
			result[i].Add(xd, y3d[i])           // Perform element-wise addition
		}
		return result
	}

	panic("unsupported tensor type in Add operation")
}

func (a *Add) Backward(ctx *Context, gradOutput interface{}) []interface{} {
	// The gradient of an addition operation is simply passed through to both inputs.
	// This logic is the same regardless of the tensor dimensionality, but we need to match the type.

	// Handle 1D tensors
	if goVec, ok := gradOutput.(*mat.VecDense); ok {
		return []interface{}{goVec, goVec}
	}

	// Handle 2D tensors
	if goDense, ok := gradOutput.(*mat.Dense); ok {
		return []interface{}{goDense, goDense}
	}

	// Handle 3D tensors
	if go3d, ok := gradOutput.([]*mat.Dense); ok {
		gradX := make([]*mat.Dense, len(go3d))
		gradY := make([]*mat.Dense, len(go3d))
		for i, goDense := range go3d {
			gradX[i] = goDense
			gradY[i] = goDense
		}
		return []interface{}{gradX, gradY}
	}

	panic("unsupported gradient output type in Add operation backward pass")
}

// ReLU operation
type ReLU struct{}

func (r *ReLU) Forward(ctx *Context, inputs ...[]float64) []float64 {
	input := inputs[0]
	result := make([]float64, len(input))
	for i, val := range input {
		if val > 0 {
			result[i] = val
		}
	}
	ctx.SaveForBackward(input)
	return result
}

func (r *ReLU) Backward(ctx *Context, gradOutput []float64) [][]float64 {
	input := ctx.savedTensors[0].([]float64)
	gradInput := make([]float64, len(input))
	for i, val := range input {
		if val > 0 {
			gradInput[i] = gradOutput[i]
		}
	}
	return [][]float64{gradInput}
}

// Dot operation (simplified for 1D vectors)
type Dot struct{}

func (d *Dot) Forward(ctx *Context, inputs ...[]float64) []float64 {
	x, y := inputs[0], inputs[1]
	if len(x) != len(y) {
		panic("Dot: input vectors must be of the same length")
	}
	var result float64
	for i := range x {
		result += x[i] * y[i]
	}
	ctx.SaveForBackward(x, y)
	return []float64{result}
}

func (d *Dot) Backward(ctx *Context, gradOutput []float64) [][]float64 {
	x, y := ctx.savedTensors[0].([]float64), ctx.savedTensors[1].([]float64)
	gradX := make([]float64, len(x))
	gradY := make([]float64, len(y))
	for i := range x {
		gradX[i] = y[i] * gradOutput[0]
		gradY[i] = x[i] * gradOutput[0]
	}
	return [][]float64{gradX, gradY}
}

// Sum operation (simplified version)
type Sum struct{}

func (s *Sum) Forward(ctx *Context, inputs ...[]float64) []float64 {
	input := inputs[0]
	var sum float64
	for _, val := range input {
		sum += val
	}
	ctx.SaveForBackward(input)
	return []float64{sum}
}

func (s *Sum) Backward(ctx *Context, gradOutput []float64) [][]float64 {
	input := ctx.savedTensors[0].([]float64)
	gradInput := make([]float64, len(input))
	for i := range gradInput {
		gradInput[i] = gradOutput[0]
	}
	return [][]float64{gradInput}
}

// LogSoftmax operation (simplified version)
type LogSoftmax struct{}

func (l *LogSoftmax) Forward(ctx *Context, inputs ...[]float64) []float64 {
	input := inputs[0]
	maxVal := max(input)
	stableInput := make([]float64, len(input))
	for i, val := range input {
		stableInput[i] = val - maxVal
	}
	expSum := sumExp(stableInput)
	logSoftmax := make([]float64, len(input))
	for i, val := range stableInput {
		logSoftmax[i] = val - math.Log(expSum)
	}
	ctx.SaveForBackward(logSoftmax)
	return logSoftmax
}

func (l *LogSoftmax) Backward(ctx *Context, gradOutput []float64) [][]float64 {
	output := ctx.savedTensors[0].([]float64)
	gradInput := make([]float64, len(output))
	expOutput := exp(output)
	sumGradOutput := sum(gradOutput)
	for i := range output {
		gradInput[i] = gradOutput[i] - expOutput[i]*sumGradOutput
	}
	return [][]float64{gradInput}
}

// Helper functions for LogSoftmax
func max(a []float64) float64 {
	maxVal := a[0]
	for _, val := range a[1:] {
		if val > maxVal {
			maxVal = val
		}
	}
	return maxVal
}

func sumExp(a []float64) float64 {
	var sum float64
	for _, val := range a {
		sum += math.Exp(val)
	}
	return sum
}

func exp(a []float64) []float64 {
	result := make([]float64, len(a))
	for i, val := range a {
		result[i] = math.Exp(val)
	}
	return result
}

func sum(a []float64) float64 {
	var sum float64
	for _, val := range a {
		sum += val
	}
	return sum
}
