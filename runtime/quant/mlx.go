package quant

// MLX quantization support.
//
// MLX affine 4-bit format:
//   weight[outDim, inDim/8] uint32 — 8 packed 4-bit values per uint32, LSB-first
//   scales[outDim, inDim/group_size] float32/float16 — per-group scale
//   biases[outDim, inDim/group_size] float32/float16 — per-group bias
//
// Dequantization:
//   for each group g in row r:
//     for each element e in group:
//       val = (packed >> (e*4)) & 0xF
//       weight[r][g*group_size + e] = val * scales[r][g] + biases[r][g]
//
// Key differences from GPTQ:
//   - Layout is [outDim, inDim/8] not [inDim/8, outDim]
//   - Bias (additive) instead of zero-point (subtractive)
//   - No g_idx permutation — groups are sequential
//   - Embeddings may also be quantized

import (
	"encoding/binary"
	"fmt"
	"math"
)

// MLXQuantWeight holds MLX affine quantized weight data.
type MLXQuantWeight struct {
	Weight    []uint32  // [outDim, inDim/8] packed 4-bit
	Scales    []float32 // [outDim, numGroups]
	Biases    []float32 // [outDim, numGroups]
	OutDim    int
	InDim     int
	Groups    int // numGroups = inDim / groupSize
	GroupSize int
	Bits      int
}

// DequantMLX dequantizes an MLX affine quantized weight to F32.
// weight[outDim, inDim/packFactor] × scales[outDim, numGroups] + biases[outDim, numGroups]
// Returns [outDim, inDim] float32.
func DequantMLX(qw *MLXQuantWeight) []float32 {
	out := make([]float32, qw.OutDim*qw.InDim)
	packFactor := 32 / qw.Bits
	mask := uint32((1 << qw.Bits) - 1)

	for row := 0; row < qw.OutDim; row++ {
		rowOff := row * qw.InDim
		packedOff := row * (qw.InDim / packFactor)
		scaleOff := row * qw.Groups

		for g := 0; g < qw.Groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			gStart := g * qw.GroupSize

			for e := 0; e < qw.GroupSize; e++ {
				idx := gStart + e
				packIdx := idx / packFactor
				bitPos := uint(idx%packFactor) * uint(qw.Bits)
				val := (qw.Weight[packedOff+packIdx] >> bitPos) & mask
				out[rowOff+idx] = float32(val)*scale + bias
			}
		}
	}
	return out
}

// GemvMLQ performs matrix-vector multiply with MLX quantized weight.
// out[outDim] = W_mlx[outDim, inDim] · x[inDim] (dequantized on-the-fly)
func GemvMLQ(out, x []float32, qw *MLXQuantWeight) {
	packFactor := 32 / qw.Bits
	mask := uint32((1 << qw.Bits) - 1)

	for row := 0; row < qw.OutDim; row++ {
		packedOff := row * (qw.InDim / packFactor)
		scaleOff := row * qw.Groups
		sum := float32(0)

		for g := 0; g < qw.Groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			gStart := g * qw.GroupSize

			gsum := float32(0)
			xsum := float32(0) // for bias accumulation

			for e := 0; e < qw.GroupSize; e++ {
				idx := gStart + e
				packIdx := idx / packFactor
				bitPos := uint(idx%packFactor) * uint(qw.Bits)
				val := float32((qw.Weight[packedOff+packIdx] >> bitPos) & mask)
				gsum += val * x[idx]
				xsum += x[idx]
			}
			sum += gsum*scale + xsum*bias
		}
		out[row] = sum
	}
}

// LoadMLXWeight loads an MLX affine quantized weight from safetensors.
// prefix is e.g. "model.layers.0.self_attn.q_proj"
func LoadMLXWeight(f interface {
	GetFloat32(name string) ([]float32, []int, error)
	GetRaw(name string) ([]byte, string, []int, error)
}, prefix string, outDim, inDim, groupSize, bits int) (*MLXQuantWeight, error) {
	packFactor := 32 / bits
	numGroups := inDim / groupSize

	// Load packed weight: [outDim, inDim/packFactor] as uint32
	raw, dtype, shape, err := f.GetRaw(prefix + ".weight")
	if err != nil {
		return nil, fmt.Errorf("load %s.weight: %w", prefix, err)
	}

	var weight []uint32
	if dtype == "U32" || dtype == "I32" {
		n := len(raw) / 4
		weight = make([]uint32, n)
		for i := 0; i < n; i++ {
			weight[i] = binary.LittleEndian.Uint32(raw[i*4:])
		}
	} else {
		return nil, fmt.Errorf("MLX weight dtype %s not supported (expected U32/I32)", dtype)
	}

	// Verify shape
	expectedN := outDim * (inDim / packFactor)
	if len(weight) != expectedN {
		// Try to infer dims from shape
		if len(shape) == 2 {
			outDim = shape[0]
			inDim = shape[1] * packFactor
			numGroups = inDim / groupSize
			expectedN = outDim * (inDim / packFactor)
		}
		if len(weight) != expectedN {
			return nil, fmt.Errorf("MLX weight shape mismatch: got %d, expected %d (%dx%d)", len(weight), expectedN, outDim, inDim/packFactor)
		}
	}

	// Load scales: [outDim, numGroups]
	scales, err := loadMLXFloat(f, prefix+".scales", outDim*numGroups)
	if err != nil {
		return nil, err
	}

	// Load biases: [outDim, numGroups]
	biases, err := loadMLXFloat(f, prefix+".biases", outDim*numGroups)
	if err != nil {
		return nil, err
	}

	return &MLXQuantWeight{
		Weight:    weight,
		Scales:    scales,
		Biases:    biases,
		OutDim:    outDim,
		InDim:     inDim,
		Groups:    numGroups,
		GroupSize: groupSize,
		Bits:      bits,
	}, nil
}

// loadMLXFloat loads a float tensor, handling F16/BF16/F32.
func loadMLXFloat(f interface {
	GetFloat32(name string) ([]float32, []int, error)
	GetRaw(name string) ([]byte, string, []int, error)
}, name string, expectedN int) ([]float32, error) {
	// Try direct F32 first
	data, _, err := f.GetFloat32(name)
	if err == nil {
		return data, nil
	}

	// Try raw with dtype conversion
	raw, dtype, _, err := f.GetRaw(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}

	switch dtype {
	case "F16":
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return out, nil
	case "BF16":
		n := len(raw) / 2
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16
			out[i] = math.Float32frombits(bits)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported dtype %s for %s", dtype, name)
	}
}
