package quant

import (
	"fmt"
	"math"
)

// NVFP4Weight holds the TensorRT Model Optimizer / NVFP4 safetensors layout
// seen in public Qwen3 and Gemma4 checkpoints.
//
// Observed tensor set for quantized linear weights:
//
//	<prefix>.weight         U8       [outDim, inDim/2] packed FP4, two weights/byte
//	<prefix>.weight_scale   F8_E4M3  [outDim, inDim/groupSize] per-block scale bytes
//	<prefix>.weight_scale_2 F32      [] global/secondary scale
//	<prefix>.input_scale    F32      [] activation scale, optional for some tensors
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

// DecodeFP4E2M1 decodes NVIDIA FP4 E2M1 values as used by NVFP4 packed
// weights. Codes 0..7 map to positive {0, 0.5, 1, 1.5, 2, 3, 4, 6}; bit 3 is
// the sign bit.
func DecodeFP4E2M1(code byte) float32 {
	mag := [...]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}[code&0x7]
	if code&0x8 != 0 {
		return -mag
	}
	return mag
}

// DecodeF8E4M3 decodes finite IEEE-style E4M3 bytes used by safetensors
// F8_E4M3 scale tensors. NaN/Inf encodings are preserved for diagnostics, but
// ModelOpt scale tensors are expected to be finite.
func DecodeF8E4M3(code byte) float32 {
	sign := code & 0x80
	exp := (code >> 3) & 0x0f
	mant := code & 0x07
	var v float32
	if exp == 0 {
		if mant == 0 {
			v = 0
		} else {
			v = float32(mant) / 8 * float32(math.Ldexp(1, -6))
		}
	} else if exp == 0x0f {
		if mant == 0 {
			v = float32(math.Inf(1))
		} else {
			v = float32(math.NaN())
		}
	} else {
		v = (1 + float32(mant)/8) * float32(math.Ldexp(1, int(exp)-7))
	}
	if sign != 0 {
		return -v
	}
	return v
}

// UnpackNVFP4 expands packed low-nibble-first FP4 bytes into decoded E2M1
// values. It is primarily a test/prototype helper; production paths should
// dequantize directly from packed bytes.
func UnpackNVFP4(packed []byte, count int) []float32 {
	if count < 0 || count > len(packed)*2 {
		return nil
	}
	out := make([]float32, count)
	for i := 0; i < count; i++ {
		b := packed[i/2]
		code := b & 0x0f
		if i%2 == 1 {
			code = b >> 4
		}
		out[i] = DecodeFP4E2M1(code)
	}
	return out
}

// DequantNVFP4 dequantizes the observed ModelOpt NVFP4 layout to F32 using
// per-block E4M3 scales and the scalar weight_scale_2 multiplier. It returns
// [outDim, inDim] row-major F32 values and is intended as the correctness-first
// CPU reference path for tests and future fallback code.
func DequantNVFP4(qw *NVFP4Weight) []float32 {
	if err := ValidateNVFP4Weight(qw); err != nil {
		return nil
	}
	outLen, ok := checkedMulInt(qw.OutDim, qw.InDim)
	if !ok {
		return nil
	}
	out := make([]float32, outLen)
	for row := 0; row < qw.OutDim; row++ {
		for col := 0; col < qw.InDim; col++ {
			out[row*qw.InDim+col] = nvfp4At(qw, row, col)
		}
	}
	return out
}

// GemvNVFP4 performs a correctness-first matrix-vector multiply directly from
// packed NVFP4 weights: out[outDim] = W_nvfp4[outDim, inDim] · x[inDim].
func GemvNVFP4(out, x []float32, qw *NVFP4Weight) {
	if err := ValidateNVFP4Weight(qw); err != nil || len(out) < qw.OutDim || len(x) < qw.InDim {
		return
	}
	for row := 0; row < qw.OutDim; row++ {
		sum := float32(0)
		for col := 0; col < qw.InDim; col++ {
			sum += nvfp4At(qw, row, col) * x[col]
		}
		out[row] = sum
	}
}

func nvfp4At(qw *NVFP4Weight, row, col int) float32 {
	rowPacked := row * (qw.InDim / 2)
	rowScale := row * qw.Groups
	group := col / qw.GroupSize
	scale := DecodeF8E4M3(qw.WeightScale[rowScale+group]) * qw.WeightScale2
	b := qw.Weight[rowPacked+col/2]
	code := b & 0x0f
	if col%2 == 1 {
		code = b >> 4
	}
	return DecodeFP4E2M1(code) * scale
}
