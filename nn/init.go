package nn

import (
	"math"
	"math/rand"

	"gonn/tensor"
)

// XavierUniform fills t with U(-a, a) where a = sqrt(6 / (fanIn+fanOut)).
func XavierUniform(t *tensor.Tensor, fanIn, fanOut int) {
	a := math.Sqrt(6.0 / float64(fanIn+fanOut))
	for i := range t.Data {
		t.Data[i] = -a + rand.Float64()*(2*a)
	}
}

// XavierNormal fills t with N(0, std^2) where std = sqrt(2 / (fanIn+fanOut)).
func XavierNormal(t *tensor.Tensor, fanIn, fanOut int) {
	std := math.Sqrt(2.0 / float64(fanIn+fanOut))
	for i := range t.Data {
		t.Data[i] = rand.NormFloat64() * std
	}
}

// KaimingUniform fills t with U(-bound, bound) where bound = sqrt(6/fanIn).
func KaimingUniform(t *tensor.Tensor, fanIn int) {
	bound := math.Sqrt(6.0 / float64(fanIn))
	for i := range t.Data {
		t.Data[i] = -bound + rand.Float64()*(2*bound)
	}
}

// KaimingNormal fills t with N(0, std^2) where std = sqrt(2/fanIn).
func KaimingNormal(t *tensor.Tensor, fanIn int) {
	std := math.Sqrt(2.0 / float64(fanIn))
	for i := range t.Data {
		t.Data[i] = rand.NormFloat64() * std
	}
}

// Constant fills t with v.
func Constant(t *tensor.Tensor, v float64) {
	for i := range t.Data {
		t.Data[i] = v
	}
}

// ZerosInit fills t with 0.
func ZerosInit(t *tensor.Tensor) { Constant(t, 0) }

// OnesInit fills t with 1.
func OnesInit(t *tensor.Tensor) { Constant(t, 1) }

// Uniform fills t with samples from U(low, high).
func Uniform(t *tensor.Tensor, low, high float64) {
	for i := range t.Data {
		t.Data[i] = low + rand.Float64()*(high-low)
	}
}

// Normal fills t with samples from N(mean, std^2).
func Normal(t *tensor.Tensor, mean, std float64) {
	for i := range t.Data {
		t.Data[i] = mean + rand.NormFloat64()*std
	}
}

// Orthogonal fills the last two dims of t with an orthogonal matrix scaled by gain.
// Uses QR via Gram-Schmidt on a random Gaussian matrix.
func Orthogonal(t *tensor.Tensor, gain float64) {
	if len(t.Shape) < 2 {
		// Fallback: just fill with normals scaled by gain.
		for i := range t.Data {
			t.Data[i] = rand.NormFloat64() * gain
		}
		return
	}
	rows := t.Shape[0]
	cols := 1
	for i := 1; i < len(t.Shape); i++ {
		cols *= t.Shape[i]
	}
	// Generate random rows x cols, then Gram-Schmidt orthonormalize the rows
	// (or columns, whichever has fewer entries — but rows is simplest).
	flat := make([]float64, rows*cols)
	for i := range flat {
		flat[i] = rand.NormFloat64()
	}
	// Treat as (rows, cols). Orthonormalize rows if rows<=cols, else cols.
	if rows <= cols {
		for i := 0; i < rows; i++ {
			for j := 0; j < i; j++ {
				var dot float64
				for k := 0; k < cols; k++ {
					dot += flat[i*cols+k] * flat[j*cols+k]
				}
				for k := 0; k < cols; k++ {
					flat[i*cols+k] -= dot * flat[j*cols+k]
				}
			}
			var norm float64
			for k := 0; k < cols; k++ {
				norm += flat[i*cols+k] * flat[i*cols+k]
			}
			norm = math.Sqrt(norm)
			if norm > 0 {
				for k := 0; k < cols; k++ {
					flat[i*cols+k] /= norm
				}
			}
		}
	} else {
		for j := 0; j < cols; j++ {
			for jj := 0; jj < j; jj++ {
				var dot float64
				for k := 0; k < rows; k++ {
					dot += flat[k*cols+j] * flat[k*cols+jj]
				}
				for k := 0; k < rows; k++ {
					flat[k*cols+j] -= dot * flat[k*cols+jj]
				}
			}
			var norm float64
			for k := 0; k < rows; k++ {
				norm += flat[k*cols+j] * flat[k*cols+j]
			}
			norm = math.Sqrt(norm)
			if norm > 0 {
				for k := 0; k < rows; k++ {
					flat[k*cols+j] /= norm
				}
			}
		}
	}
	for i := range t.Data {
		t.Data[i] = flat[i] * gain
	}
}
