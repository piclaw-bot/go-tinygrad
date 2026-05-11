package quant

import "fmt"

// ValidateGemvQ4Sym checks inputs for GemvQ4Sym.
func ValidateGemvQ4Sym(out, x []float32, qweight, gIdx []int32, scales []float32, inDim, outDim int) error {
	if len(out) < outDim {
		return fmt.Errorf("GemvQ4Sym out length=%d, expected at least %d", len(out), outDim)
	}
	if len(x) < inDim {
		return fmt.Errorf("GemvQ4Sym x length=%d, expected at least %d", len(x), inDim)
	}
	if err := ValidateGPTQSym(qweight, gIdx, scales, inDim, outDim); err != nil {
		return fmt.Errorf("GemvQ4Sym GPTQ validation: %w", err)
	}
	return nil
}
