package model

import (
	"encoding/binary"
	"fmt"
	"math"
)

// LoadSwitchMLXExperts loads a switch_mlp-style 3D packed tensor and
// slices it into per-expert MLXQuantWeight entries.
//
// The safetensors weight has shape [numExperts, outDim, packedInDim] (U32)
// with matching scales/biases [numExperts, outDim, numGroups] (BF16/F32).
func LoadSwitchMLXExperts(
	f interface {
		GetRaw(name string) ([]byte, string, []int, error)
	},
	baseName string,
	numExperts, outDim, inDim, groupSize, bits int,
) ([]*MLXQuantWeight, error) {
	// Load packed weight
	wRaw, wDtype, wShape, err := f.GetRaw(baseName + ".weight")
	if err != nil {
		return nil, fmt.Errorf("load %s.weight: %w", baseName, err)
	}
	if len(wShape) != 3 || wShape[0] != numExperts {
		return nil, fmt.Errorf("%s.weight: expected [%d, ?, ?], got %v", baseName, numExperts, wShape)
	}
	_ = wDtype // should be U32

	// Load scales
	sRaw, _, sShape, err := f.GetRaw(baseName + ".scales")
	if err != nil {
		return nil, fmt.Errorf("load %s.scales: %w", baseName, err)
	}
	if len(sShape) != 3 || sShape[0] != numExperts {
		return nil, fmt.Errorf("%s.scales: expected [%d, ?, ?], got %v", baseName, numExperts, sShape)
	}

	// Load biases
	bRaw, _, bShape, err := f.GetRaw(baseName + ".biases")
	if err != nil {
		return nil, fmt.Errorf("load %s.biases: %w", baseName, err)
	}
	if len(bShape) != 3 || bShape[0] != numExperts {
		return nil, fmt.Errorf("%s.biases: expected [%d, ?, ?], got %v", baseName, numExperts, bShape)
	}

	packFactor := 32 / bits
	numGroups := inDim / groupSize
	packedPerRow := inDim / packFactor

	// Verify shapes
	if wShape[1] != outDim || wShape[2] != packedPerRow {
		return nil, fmt.Errorf("%s.weight: expected [%d, %d, %d], got %v",
			baseName, numExperts, outDim, packedPerRow, wShape)
	}

	// Per-expert slicing
	wStride := outDim * packedPerRow * 4 // bytes per expert in weight
	sStride := outDim * numGroups * 2    // bytes per expert in scales (BF16)
	bStride := outDim * numGroups * 2    // bytes per expert in biases (BF16)

	experts := make([]*MLXQuantWeight, numExperts)
	for e := 0; e < numExperts; e++ {
		wSlice := wRaw[e*wStride : (e+1)*wStride]
		sSlice := sRaw[e*sStride : (e+1)*sStride]
		bSlice := bRaw[e*bStride : (e+1)*bStride]

		// Parse uint32 weight
		nW := len(wSlice) / 4
		weight := make([]uint32, nW)
		for i := 0; i < nW; i++ {
			weight[i] = binary.LittleEndian.Uint32(wSlice[i*4:])
		}

		// Parse BF16 scales → float32
		nS := len(sSlice) / 2
		scales := make([]float32, nS)
		for i := 0; i < nS; i++ {
			bits16 := binary.LittleEndian.Uint16(sSlice[i*2:])
			scales[i] = bf16ToF32(bits16)
		}

		// Parse BF16 biases → float32
		nB := len(bSlice) / 2
		biases := make([]float32, nB)
		for i := 0; i < nB; i++ {
			bits16 := binary.LittleEndian.Uint16(bSlice[i*2:])
			biases[i] = bf16ToF32(bits16)
		}

		experts[e] = &MLXQuantWeight{
			Weight:    weight,
			Scales:    scales,
			Biases:    biases,
			InDim:     inDim,
			OutDim:    outDim,
			Groups:    numGroups,
			GroupSize: groupSize,
			Bits:      bits,
		}
	}

	return experts, nil
}

func bf16ToF32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}
