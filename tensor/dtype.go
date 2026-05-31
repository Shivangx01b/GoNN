package tensor

import "math"

// DType is the logical element type of a Tensor. Storage is always float64;
// DType controls the *numeric precision and range* of values: results are
// rounded to the representable set of the dtype at op boundaries, so a Float16
// tensor behaves exactly like IEEE-754 binary16 (limited precision, overflow to
// Inf, subnormals), and Float32 like binary32.
//
// This is precision/range emulation on float64 storage — it gives correct
// mixed-precision *numerics* (useful for matching fp32 results, testing range
// robustness, simulating mixed precision). It does NOT reduce memory, and it
// does not model reduced-precision *accumulation* inside matmul/reductions
// (those still accumulate in float64, then round the result). For true fp16
// storage and tensor-core compute on the GPU, see backend/cuda's DeviceBufferF16.
type DType uint8

const (
	// Float64 is the default (full precision); rounding to it is a no-op, so
	// tensors created without a dtype behave exactly as before.
	Float64 DType = iota
	Float32
	Float16
)

// String returns the PyTorch-style name.
func (d DType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float16:
		return "float16"
	default:
		return "float64"
	}
}

// promote returns the higher-precision (wider) of two dtypes, matching numpy/
// torch type promotion: float16+float32 -> float32, float32+float64 -> float64.
// Precision rank is Float64 > Float32 > Float16; the constants are ordered so
// the more-precise dtype has the smaller value.
func promote(a, b DType) DType {
	if a < b {
		return a
	}
	return b
}

// roundTo rounds v to the nearest value representable in dtype d.
func roundTo(v float64, d DType) float64 {
	switch d {
	case Float32:
		return float64(float32(v))
	case Float16:
		return float16ToFloat64(float32ToFloat16bits(float32(v)))
	default:
		return v
	}
}

// castInPlace rounds every element of t to t.Dtype (no-op for Float64).
func castInPlace(t *Tensor) {
	if t.Dtype == Float64 {
		return
	}
	for i, v := range t.Data {
		t.Data[i] = roundTo(v, t.Dtype)
	}
}

// ---- IEEE-754 binary16 <-> binary32, round-to-nearest-even ------------------

// float32ToFloat16bits converts a float32 to the bit pattern of the nearest
// IEEE-754 binary16, rounding to nearest, ties to even. Handles Inf/NaN,
// overflow (-> Inf), and subnormals.
func float32ToFloat16bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	rawExp := int32((b >> 23) & 0xff)
	mant := b & 0x7fffff

	if rawExp == 0xff { // Inf / NaN
		if mant != 0 {
			return sign | 0x7e00 // NaN (quiet)
		}
		return sign | 0x7c00 // Inf
	}
	if (b & 0x7fffffff) == 0 {
		return sign // +/- 0
	}

	exp := rawExp - 127 + 15 // rebias to half
	if exp >= 31 {
		return sign | 0x7c00 // overflow -> Inf
	}
	if exp <= 0 {
		// Subnormal half (or underflow to zero).
		if exp < -10 {
			return sign
		}
		m := mant | 0x800000 // restore implicit leading 1 (24-bit significand)
		shift := uint(14 - exp)
		half := m >> shift
		roundBit := (m >> (shift - 1)) & 1
		sticky := m & ((1 << (shift - 1)) - 1)
		if roundBit == 1 && (sticky != 0 || (half&1) == 1) {
			half++
		}
		return sign | uint16(half)
	}
	// Normal half.
	half := uint16(exp<<10) | uint16(mant>>13)
	roundBit := (mant >> 12) & 1
	sticky := mant & 0xfff
	if roundBit == 1 && (sticky != 0 || (half&1) == 1) {
		half++ // carry into exponent (and to Inf at exp 31) is handled by the add
	}
	return sign | half
}

// float16ToFloat64 expands a binary16 bit pattern to float64 (exact).
func float16ToFloat64(h uint16) float64 {
	sign := float64(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	exp := int((h >> 10) & 0x1f)
	mant := int(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return sign * 0
		}
		return sign * float64(mant) * math.Pow(2, -24) // subnormal
	case 0x1f:
		if mant == 0 {
			return sign * math.Inf(1)
		}
		return math.NaN()
	default:
		return sign * (1 + float64(mant)/1024) * math.Pow(2, float64(exp-15))
	}
}

// ---- Tensor dtype API -------------------------------------------------------

// DType returns the tensor's logical element type.
func (t *Tensor) DType() DType { return t.Dtype }

// NewTyped creates a tensor of the given dtype, rounding data to its precision.
func NewTyped(data []float64, dt DType, shape ...int) *Tensor {
	t := New(append([]float64(nil), data...), shape...)
	t.Dtype = dt
	castInPlace(t)
	return t
}

// AsType returns a copy of t with values rounded to dtype dt. It is
// differentiable (straight-through: gradient passes to the input unchanged).
// To is an alias.
func (t *Tensor) AsType(dt DType) *Tensor {
	out := &Tensor{
		Data:    make([]float64, len(t.Data)),
		Shape:   append([]int(nil), t.Shape...),
		Strides: append([]int(nil), t.Strides...),
		Dtype:   dt,
	}
	for i, v := range t.Data {
		out.Data[i] = roundTo(v, dt)
	}
	if t.RequiresGrad || t.creator != nil {
		out.RequiresGrad = true
		out.creator = &Function{
			Name:   "AsType",
			Inputs: []*Tensor{t},
			Backward: func(grad *Tensor, _ []interface{}, _ []*Tensor) []*Tensor {
				return []*Tensor{grad}
			},
		}
	}
	return out
}

// To is an alias for AsType (PyTorch parity: tensor.to(dtype)).
func (t *Tensor) To(dt DType) *Tensor { return t.AsType(dt) }
