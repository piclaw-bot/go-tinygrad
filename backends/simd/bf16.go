package simd

// BF16 native operations: work directly on []uint16 BF16 data.
// Accumulate in F32 internally but inputs/outputs are BF16.
// This halves memory bandwidth vs the F32 emulation path.

import (
	"math"
	"unsafe"
)

// BF16 is a bfloat16 value stored as uint16.
type BF16 = uint16

// BF16ToF32 converts a single BF16 to float32.
func BF16ToF32(b BF16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// F32ToBF16 converts a single float32 to BF16 (truncate, no rounding).
func F32ToBF16(f float32) BF16 {
	return BF16(math.Float32bits(f) >> 16)
}

// BF16DotF32 computes dot(bf16_x, f32_y) accumulating in F32.
// x is BF16 ([]uint16), y is F32 ([]float32). Returns F32 scalar.
// Used for: BF16 hidden @ F32 weight (mixed precision GEMV).
func BF16DotF32(x []uint16, y []float32) float32 {
	n := len(x)
	if n != len(y) {
		if len(y) < n {
			n = len(y)
		}
	}
	sum := float32(0)
	// Process 4 at a time for better ILP
	i := 0
	for ; i+3 < n; i += 4 {
		sum += BF16ToF32(x[i]) * y[i]
		sum += BF16ToF32(x[i+1]) * y[i+1]
		sum += BF16ToF32(x[i+2]) * y[i+2]
		sum += BF16ToF32(x[i+3]) * y[i+3]
	}
	for ; i < n; i++ {
		sum += BF16ToF32(x[i]) * y[i]
	}
	return sum
}

// BF16Dot computes dot(bf16_x, bf16_y) accumulating in F32.
func BF16Dot(x, y []uint16) float32 {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	sum := float32(0)
	i := 0
	for ; i+3 < n; i += 4 {
		sum += BF16ToF32(x[i]) * BF16ToF32(y[i])
		sum += BF16ToF32(x[i+1]) * BF16ToF32(y[i+1])
		sum += BF16ToF32(x[i+2]) * BF16ToF32(y[i+2])
		sum += BF16ToF32(x[i+3]) * BF16ToF32(y[i+3])
	}
	for ; i < n; i++ {
		sum += BF16ToF32(x[i]) * BF16ToF32(y[i])
	}
	return sum
}

// BF16RMSNorm computes RMSNorm in-place on BF16 data with BF16 weights.
// Accumulates sum-of-squares in F32, outputs BF16.
func BF16RMSNorm(x, w []uint16, eps float32) {
	n := len(x)
	if n == 0 || len(w) < n {
		return
	}
	// Sum of squares in F32
	ss := float32(0)
	for _, v := range x {
		f := BF16ToF32(v)
		ss += f * f
	}
	invRMS := float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))
	// Apply norm
	for i := range x {
		f := BF16ToF32(x[i]) * invRMS * BF16ToF32(w[i])
		x[i] = F32ToBF16(f)
	}
}

// BF16VecAdd computes dst[i] = BF16(BF16ToF32(a[i]) + BF16ToF32(b[i]))
func BF16VecAdd(dst, a, b []uint16) {
	n := len(dst)
	if len(a) < n {
		n = len(a)
	}
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		dst[i] = F32ToBF16(BF16ToF32(a[i]) + BF16ToF32(b[i]))
	}
}

// BF16GemvNT computes out[j] = dot(x, w[j*inDim:(j+1)*inDim]) for BF16 x, F32 w.
// This is the mixed-precision GEMV: BF16 activations × F32 weights → BF16 output.
func BF16GemvNT(out []uint16, x []uint16, w []float32, inDim, outDim int) {
	if inDim <= 0 || outDim <= 0 || len(out) < outDim || len(x) < inDim || len(w) < inDim*outDim {
		return
	}
	for j := 0; j < outDim; j++ {
		row := w[j*inDim : (j+1)*inDim]
		sum := BF16DotF32(x, row)
		out[j] = F32ToBF16(sum)
	}
}

// BF16FromF32Slice converts []float32 to []uint16 (BF16) in-place reinterpretation.
// Returns a new []uint16 backed by new memory.
func BF16FromF32Slice(f32 []float32) []uint16 {
	out := make([]uint16, len(f32))
	for i, v := range f32 {
		out[i] = F32ToBF16(v)
	}
	return out
}

// BF16ToF32Slice converts []uint16 (BF16) to []float32.
func BF16ToF32Slice(bf16 []uint16) []float32 {
	out := make([]float32, len(bf16))
	for i, v := range bf16 {
		out[i] = BF16ToF32(v)
	}
	return out
}

// BF16SlicePtr returns unsafe pointer for passing to assembly.
func BF16SlicePtr(s []uint16) unsafe.Pointer {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Pointer(&s[0])
}
