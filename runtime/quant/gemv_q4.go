package quant

// GemvQ4Sym computes out = x @ W^T where W is stored as GPTQ INT4 symmetric.
// This dequantizes on-the-fly during the dot product, avoiding the full F32 expansion.
//
// qweight: [inDim/8, outDim] packed int32
// scales:  [numGroups, outDim] float32
// gIdx:    [inDim] int32
// x:       [inDim] float32
// out:     [outDim] float32
func GemvQ4Sym(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) {
	if err := ValidateGemvQ4Sym(out, x, qweight, gIdx, scales, inDim, outDim); err != nil {
		return
	}
	for j := 0; j < outDim; j++ {
		var sum float32
		for packIdx := 0; packIdx < inDim/8; packIdx++ {
			packed := qweight[packIdx*outDim+j]
			for bit := 0; bit < 8; bit++ {
				i := packIdx*8 + bit
				qw := (packed >> (uint(bit) * 4)) & 0xF
				g := int(gIdx[i])
				scale := scales[g*outDim+j]
				sum += x[i] * scale * float32(qw-8)
			}
		}
		out[j] = sum
	}
}
