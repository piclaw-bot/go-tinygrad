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
	if err := ValidateMLXQuantWeight(qw); err != nil {
		return nil
	}
	outLen, ok := checkedMulInt(qw.OutDim, qw.InDim)
	if !ok {
		return nil
	}
	out := make([]float32, outLen)
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
	if err := ValidateMLXQuantWeight(qw); err != nil || len(out) < qw.OutDim || len(x) < qw.InDim {
		return
	}
	if qw.Bits == 4 && qw.GroupSize%8 == 0 {
		gemvMLQ4(out, x, qw)
		return
	}
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

func gemvMLQ4(out, x []float32, qw *MLXQuantWeight) {
	packedPerRow := qw.InDim / 8
	packsPerGroup := qw.GroupSize / 8
	for row := 0; row < qw.OutDim; row++ {
		packedOff := row * packedPerRow
		scaleOff := row * qw.Groups
		sum := float32(0)
		for g := 0; g < qw.Groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			packBase := packedOff + g*packsPerGroup
			xBase := g * qw.GroupSize
			gsum := float32(0)
			xsum := float32(0)
			for p := 0; p < packsPerGroup; p++ {
				packed := qw.Weight[packBase+p]
				xi := xBase + p*8
				x0 := x[xi]
				x1 := x[xi+1]
				x2 := x[xi+2]
				x3 := x[xi+3]
				x4 := x[xi+4]
				x5 := x[xi+5]
				x6 := x[xi+6]
				x7 := x[xi+7]
				gsum += float32(packed&0xF)*x0 + float32((packed>>4)&0xF)*x1 + float32((packed>>8)&0xF)*x2 + float32((packed>>12)&0xF)*x3 +
					float32((packed>>16)&0xF)*x4 + float32((packed>>20)&0xF)*x5 + float32((packed>>24)&0xF)*x6 + float32((packed>>28)&0xF)*x7
				xsum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
			}
			sum += gsum*scale + xsum*bias
		}
		out[row] = sum
	}
}

// ValidateMLXQuantWeight checks an in-memory MLX quantized weight before use.
func ValidateMLXQuantWeight(qw *MLXQuantWeight) error {
	if qw == nil {
		return fmt.Errorf("nil MLX quant weight")
	}
	if qw.Bits <= 0 || qw.Bits > 32 || 32%qw.Bits != 0 {
		return fmt.Errorf("invalid MLX bits=%d", qw.Bits)
	}
	if qw.OutDim <= 0 || qw.InDim <= 0 || qw.GroupSize <= 0 || qw.Groups <= 0 {
		return fmt.Errorf("invalid MLX dims out=%d in=%d groupSize=%d groups=%d", qw.OutDim, qw.InDim, qw.GroupSize, qw.Groups)
	}
	packFactor := 32 / qw.Bits
	if qw.InDim%packFactor != 0 {
		return fmt.Errorf("MLX inDim=%d is not divisible by packFactor=%d", qw.InDim, packFactor)
	}
	if qw.InDim%qw.GroupSize != 0 || qw.Groups != qw.InDim/qw.GroupSize {
		return fmt.Errorf("MLX group layout mismatch inDim=%d groupSize=%d groups=%d", qw.InDim, qw.GroupSize, qw.Groups)
	}
	wantWeight, ok := checkedMulInt(qw.OutDim, qw.InDim/packFactor)
	if !ok {
		return fmt.Errorf("MLX weight size overflows out=%d in=%d packFactor=%d", qw.OutDim, qw.InDim, packFactor)
	}
	wantScale, ok := checkedMulInt(qw.OutDim, qw.Groups)
	if !ok {
		return fmt.Errorf("MLX scale/bias size overflows out=%d groups=%d", qw.OutDim, qw.Groups)
	}
	if len(qw.Weight) < wantWeight {
		return fmt.Errorf("MLX weight length=%d, expected at least %d", len(qw.Weight), wantWeight)
	}
	if len(qw.Scales) < wantScale || len(qw.Biases) < wantScale {
		return fmt.Errorf("MLX scale/bias length=%d/%d, expected at least %d", len(qw.Scales), len(qw.Biases), wantScale)
	}
	return nil
}

// LoadMLXWeight loads an MLX affine quantized weight from safetensors.
// prefix is e.g. "model.layers.0.self_attn.q_proj"
func LoadMLXWeight(f interface {
	GetFloat32(name string) ([]float32, []int, error)
	GetRaw(name string) ([]byte, string, []int, error)
}, prefix string, outDim, inDim, groupSize, bits int) (*MLXQuantWeight, error) {
	if bits <= 0 || bits > 32 || 32%bits != 0 {
		return nil, fmt.Errorf("invalid MLX bits=%d", bits)
	}
	if groupSize <= 0 {
		return nil, fmt.Errorf("invalid MLX groupSize=%d", groupSize)
	}
	packFactor := 32 / bits
	// Load packed weight: [outDim, inDim/packFactor] as uint32. Prefer the
	// safetensors shape when available so callers cannot accidentally use a
	// matching element count with the wrong logical row/column dimensions.
	raw, dtype, shape, err := f.GetRaw(prefix + ".weight")
	if err != nil {
		return nil, fmt.Errorf("load %s.weight: %w", prefix, err)
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("MLX weight raw byte length %d is not divisible by 4", len(raw))
	}

	if len(shape) == 2 {
		shapeOut, shapePackedIn := shape[0], shape[1]
		if shapeOut <= 0 || shapePackedIn <= 0 {
			return nil, fmt.Errorf("MLX weight invalid shape %v", shape)
		}
		shapeIn, ok := checkedMulInt(shapePackedIn, packFactor)
		if !ok {
			return nil, fmt.Errorf("MLX weight shape %v overflows inDim with packFactor=%d", shape, packFactor)
		}
		if shapeIn%groupSize != 0 {
			return nil, fmt.Errorf("MLX weight shape %v implies inDim=%d not divisible by groupSize=%d", shape, shapeIn, groupSize)
		}
		outDim = shapeOut
		inDim = shapeIn
	} else if len(shape) != 0 {
		return nil, fmt.Errorf("MLX weight shape rank %d unsupported for %s.weight", len(shape), prefix)
	}
	if inDim <= 0 || outDim <= 0 {
		return nil, fmt.Errorf("invalid MLX dims outDim=%d inDim=%d", outDim, inDim)
	}
	if inDim%packFactor != 0 {
		return nil, fmt.Errorf("MLX inDim=%d is not divisible by packFactor=%d", inDim, packFactor)
	}
	if inDim%groupSize != 0 {
		return nil, fmt.Errorf("MLX inDim=%d is not divisible by groupSize=%d", inDim, groupSize)
	}
	numGroups := inDim / groupSize

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

	expectedN, ok := checkedMulInt(outDim, inDim/packFactor)
	if !ok {
		return nil, fmt.Errorf("MLX weight expected size overflows out=%d in=%d packFactor=%d", outDim, inDim, packFactor)
	}
	if len(weight) != expectedN {
		return nil, fmt.Errorf("MLX weight shape mismatch: got %d, expected %d (%dx%d)", len(weight), expectedN, outDim, inDim/packFactor)
	}

	expectedScaleN, ok := checkedMulInt(outDim, numGroups)
	if !ok {
		return nil, fmt.Errorf("MLX scale/bias expected size overflows out=%d groups=%d", outDim, numGroups)
	}

	// Load scales: [outDim, numGroups]
	scales, err := loadMLXFloat(f, prefix+".scales", expectedScaleN)
	if err != nil {
		return nil, err
	}

	// Load biases: [outDim, numGroups]
	biases, err := loadMLXFloat(f, prefix+".biases", expectedScaleN)
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

// loadMLXFloat loads a scale/bias tensor, accepting only F32/F16/BF16.
func loadMLXFloat(f interface {
	GetFloat32(name string) ([]float32, []int, error)
	GetRaw(name string) ([]byte, string, []int, error)
}, name string, expectedN int) ([]float32, error) {
	raw, dtype, shape, err := f.GetRaw(name)
	if err != nil {
		// Fallback for minimal test doubles that do not expose raw dtype.
		data, shape, f32Err := f.GetFloat32(name)
		if f32Err != nil {
			return nil, fmt.Errorf("load %s: %w", name, err)
		}
		if err := validateMLXFloatLen(name, len(data), shape, expectedN); err != nil {
			return nil, err
		}
		return data, nil
	}

	var out []float32
	switch dtype {
	case "F32":
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("%s F32 raw byte length %d is not divisible by 4", name, len(raw))
		}
		n := len(raw) / 4
		out = make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case "F16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("%s F16 raw byte length %d is not divisible by 2", name, len(raw))
		}
		n := len(raw) / 2
		out = make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "BF16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("%s BF16 raw byte length %d is not divisible by 2", name, len(raw))
		}
		n := len(raw) / 2
		out = make([]float32, n)
		for i := 0; i < n; i++ {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16
			out[i] = math.Float32frombits(bits)
		}
	default:
		return nil, fmt.Errorf("unsupported dtype %s for %s", dtype, name)
	}
	if err := validateMLXFloatLen(name, len(out), shape, expectedN); err != nil {
		return nil, err
	}
	return out, nil
}

func validateMLXFloatLen(name string, got int, shape []int, expectedN int) error {
	if expectedN < 0 {
		return fmt.Errorf("%s invalid expected length %d", name, expectedN)
	}
	if len(shape) > 0 {
		shapeN := 1
		for _, d := range shape {
			if d <= 0 {
				return fmt.Errorf("%s invalid shape %v", name, shape)
			}
			var ok bool
			shapeN, ok = checkedMulInt(shapeN, d)
			if !ok {
				return fmt.Errorf("%s shape %v element count overflows", name, shape)
			}
		}
		if shapeN != got {
			return fmt.Errorf("%s shape %v has %d elements, raw data has %d", name, shape, shapeN, got)
		}
	}
	if got != expectedN {
		return fmt.Errorf("%s length mismatch: got %d, expected %d", name, got, expectedN)
	}
	return nil
}
