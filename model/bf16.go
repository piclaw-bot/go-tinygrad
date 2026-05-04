package model

// BF16 precision emulation for models trained in BF16 (Gemma3).
//
// BF16 has 8 exponent bits + 7 mantissa bits (vs FP32's 23 mantissa bits).
// Models trained in BF16 rely on the implicit precision clamping.
// Running them in FP32 causes value explosion from compounding errors.
//
// Solution: round all intermediate values to BF16 precision by zeroing
// the lower 16 bits of the FP32 mantissa.

import "math"

// toBF16 rounds a float32 to BF16 precision.
func toBF16(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) & 0xFFFF0000)
}

// bf16Slice rounds all elements in a slice to BF16 precision in-place.
func bf16Slice(x []float32) {
	for i := range x {
		x[i] = toBF16(x[i])
	}
}

// rmsNormBF16 is rmsNormInPlace with BF16 output precision.
func rmsNormBF16(x, weight []float32, eps float32) {
	h := len(x)
	ss := float32(0)
	for _, v := range x {
		ss += v * v // accumulate in FP32
	}
	ss = float32(1.0 / math.Sqrt(float64(ss/float32(h)+eps)))
	for i := range x {
		x[i] = toBF16(weight[i] * x[i] * ss)
	}
}

// gemvBF16 is like gemvNT but rounds output to BF16.
func gemvBF16(out, x, w []float32, inDim, outDim int) {
	for j := 0; j < outDim; j++ {
		sum := float32(0)
		row := w[j*inDim : (j+1)*inDim]
		for p := 0; p < inDim; p++ {
			sum += x[p] * row[p] // FP32 accumulate
		}
		out[j] = toBF16(sum)
	}
}

// geluTanhBF16 applies GELU with BF16 output.
func geluTanhBF16(x float32) float32 {
	x3 := x * x * x
	inner := float32(0.7978845608) * (x + 0.044715*x3)
	return toBF16(0.5 * x * (1.0 + float32(math.Tanh(float64(inner)))))
}
