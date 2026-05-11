package model

// BF16 precision for models trained in BF16 (Gemma3).
// Now delegates to SIMD assembly implementations.

import "github.com/rcarmo/go-pherence/backends/simd"

func toBF16(x float32) float32 {
	buf := []float32{x}
	simd.ToBF16(buf)
	return buf[0]
}

func bf16Slice(x []float32) {
	simd.ToBF16(x)
}

func rmsNormBF16(x, weight []float32, eps float32) {
	simd.RMSNormBF16(x, weight, eps)
}
