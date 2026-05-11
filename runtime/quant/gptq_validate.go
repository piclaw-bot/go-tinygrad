package quant

import "fmt"

// ValidateGPTQ checks GPTQ tensor lengths and dimensions before dequantization.
func ValidateGPTQ(qweight, qzeros, gIdx []int32, scales []float32, inFeatures, outFeatures int, sym bool) error {
	if inFeatures <= 0 || outFeatures <= 0 {
		return fmt.Errorf("invalid GPTQ dims in=%d out=%d", inFeatures, outFeatures)
	}
	if inFeatures%8 != 0 {
		return fmt.Errorf("GPTQ inFeatures=%d is not divisible by 8", inFeatures)
	}
	if !sym && outFeatures%8 != 0 {
		return fmt.Errorf("GPTQ outFeatures=%d is not divisible by 8 for qzeros", outFeatures)
	}
	wantQWeight := (inFeatures / 8) * outFeatures
	if len(qweight) < wantQWeight {
		return fmt.Errorf("GPTQ qweight length=%d, expected at least %d", len(qweight), wantQWeight)
	}
	if len(gIdx) < inFeatures {
		return fmt.Errorf("GPTQ g_idx length=%d, expected at least %d", len(gIdx), inFeatures)
	}

	maxGroup := -1
	for i := 0; i < inFeatures; i++ {
		g := int(gIdx[i])
		if g < 0 {
			return fmt.Errorf("GPTQ g_idx[%d]=%d is negative", i, g)
		}
		if g > maxGroup {
			maxGroup = g
		}
	}
	if maxGroup < 0 {
		return fmt.Errorf("GPTQ g_idx has no groups")
	}

	wantScales := (maxGroup + 1) * outFeatures
	if len(scales) < wantScales {
		return fmt.Errorf("GPTQ scales length=%d, expected at least %d for %d groups", len(scales), wantScales, maxGroup+1)
	}
	if !sym {
		wantQZeros := (maxGroup + 1) * (outFeatures / 8)
		if len(qzeros) < wantQZeros {
			return fmt.Errorf("GPTQ qzeros length=%d, expected at least %d for %d groups", len(qzeros), wantQZeros, maxGroup+1)
		}
	}
	return nil
}

// ValidateGPTQSym checks symmetric GPTQ tensor lengths and dimensions.
func ValidateGPTQSym(qweight, gIdx []int32, scales []float32, inFeatures, outFeatures int) error {
	return ValidateGPTQ(qweight, nil, gIdx, scales, inFeatures, outFeatures, true)
}
