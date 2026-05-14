package quant

import "fmt"

// NVFP4Weight holds the TensorRT Model Optimizer / NVFP4 safetensors layout
// seen in public Qwen3 and Gemma4 checkpoints.
//
// Observed tensor set for quantized linear weights:
//
//	<prefix>.weight         U8       [outDim, inDim/2] packed FP4, two weights/byte
//	<prefix>.weight_scale   F8_E4M3  [outDim, inDim/groupSize] per-block scale bytes
//	<prefix>.weight_scale_2 F32      [] global/secondary scale
//	<prefix>.input_scale    F32      [] activation scale, optional for some tensors
//
// This type is a layout container only. Decode semantics are implemented in the
// later correctness fallback once the exact FP4 code mapping and scale formula
// are locked down with golden vectors.
type NVFP4Weight struct {
	Weight        []byte  // [outDim, inDim/2] U8 packed FP4 nibbles
	WeightScale   []byte  // [outDim, groups] raw F8_E4M3 bytes
	WeightScale2  float32 // scalar secondary/global scale
	InputScale    float32 // optional scalar activation scale
	HasInputScale bool
	OutDim        int
	InDim         int
	Groups        int
	GroupSize     int // observed ModelOpt NVFP4 group size is 16
}

// ValidateNVFP4Weight checks the observed NVFP4 packed-weight and scale layout.
func ValidateNVFP4Weight(qw *NVFP4Weight) error {
	if qw == nil {
		return fmt.Errorf("nil NVFP4 quant weight")
	}
	if qw.OutDim <= 0 || qw.InDim <= 0 || qw.GroupSize <= 0 || qw.Groups <= 0 {
		return fmt.Errorf("invalid NVFP4 dims out=%d in=%d groupSize=%d groups=%d", qw.OutDim, qw.InDim, qw.GroupSize, qw.Groups)
	}
	if qw.InDim%2 != 0 {
		return fmt.Errorf("NVFP4 inDim=%d is not divisible by packed FP4 factor=2", qw.InDim)
	}
	if qw.InDim%qw.GroupSize != 0 || qw.Groups != qw.InDim/qw.GroupSize {
		return fmt.Errorf("NVFP4 group layout mismatch inDim=%d groupSize=%d groups=%d", qw.InDim, qw.GroupSize, qw.Groups)
	}
	wantWeight, ok := checkedMulInt(qw.OutDim, qw.InDim/2)
	if !ok {
		return fmt.Errorf("NVFP4 weight size overflows out=%d in=%d", qw.OutDim, qw.InDim)
	}
	wantScale, ok := checkedMulInt(qw.OutDim, qw.Groups)
	if !ok {
		return fmt.Errorf("NVFP4 scale size overflows out=%d groups=%d", qw.OutDim, qw.Groups)
	}
	if len(qw.Weight) < wantWeight {
		return fmt.Errorf("NVFP4 weight length=%d, expected at least %d", len(qw.Weight), wantWeight)
	}
	if len(qw.WeightScale) < wantScale {
		return fmt.Errorf("NVFP4 weight_scale length=%d, expected at least %d", len(qw.WeightScale), wantScale)
	}
	return nil
}
