package tensor

import (
	"math"
)

// Context struct to hold operation context
type Context struct {
	parents      []*Tensor
	savedTensors []interface{}
}

// Save tensors for backward pass
func (c *Context) SaveForBackward(tensors ...interface{}) {
	c.savedTensors = append(c.savedTensors, tensors...)
}

// Tensor struct represents a tensor
type Tensor struct {
	data  []float64
	grad  []float64
	shape []int
	ctx   *Context
}

// Function interface for operations
type Function interface {
	Forward(ctx *Context, inputs ...[]float64) []float64
	Backward(ctx *Context, gradOutput []float64) [][]float64
}

// Mul operation
type Mul struct{}

func (m *Mul) Forward(ctx *Context, inputs ...[]float64) []float64 {
	x, y := inputs[0], inputs[1]
	ctx.SaveForBackward(x, y)
	result := make([]float64, len(x))
	for i := range x {
		result[i] = x[i] * y[i]
	}
	return result
}

func (m *Mul) Backward(ctx *Context, gradOutput []float64) [][]float64 {
	savedTensors := ctx.savedTensors
	x, y := savedTensors[0].([]float64), savedTensors[1].([]float64)
	gradX := make([]float64, len(x))
	gradY := make([]float64, len(y))
	for i := range gradOutput {
		gradX[i] = y[i] * gradOutput[i]
		gradY[i] = x[i] * gradOutput[i]
	}
	return [][]float64{gradX, gradY}
}

// Add operation
type Add struct{}

func (a *Add) Forward(ctx *Context, inputs ...[]float64) []float64 {
	x, y := inputs[0], inputs[1]
	result := make([]float64, len(x))
	for i := range x {
		result[i] = x[i] + y[i]
	}
	return result
}

func (a *Add) Backward(ctx *Context, gradOutput []float64) [][]float64 {
	return [][]float64{gradOutput, gradOutput}
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
