package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"

	"github.com/rcarmo/go-pherence/runtime/quant"

	"github.com/rcarmo/go-pherence/backends/simd"
)

// LoadSwitchMLXExperts loads a switch_mlp-style 3D packed tensor and
// slices it into per-expert quant.MLXQuantWeight entries.
//
// The safetensors weight has shape [numExperts, outDim, packedInDim] (U32)
// with matching scales/biases [numExperts, outDim, numGroups] (BF16/F32).
func LoadSwitchMLXExperts(
	f interface {
		GetRaw(name string) ([]byte, string, []int, error)
	},
	baseName string,
	numExperts, outDim, inDim, groupSize, bits int,
) ([]*quant.MLXQuantWeight, error) {
	if f == nil {
		return nil, fmt.Errorf("nil safetensors source")
	}
	if numExperts <= 0 || outDim <= 0 || inDim <= 0 || groupSize <= 0 || bits <= 0 || bits > 32 || 32%bits != 0 {
		return nil, fmt.Errorf("invalid switch MLX dims experts=%d out=%d in=%d groupSize=%d bits=%d", numExperts, outDim, inDim, groupSize, bits)
	}
	if inDim%groupSize != 0 {
		return nil, fmt.Errorf("switch MLX inDim=%d is not divisible by groupSize=%d", inDim, groupSize)
	}
	packFactor := 32 / bits
	if inDim%packFactor != 0 {
		return nil, fmt.Errorf("switch MLX inDim=%d is not divisible by packFactor=%d", inDim, packFactor)
	}
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

	numGroups := inDim / groupSize
	packedPerRow := inDim / packFactor

	// Verify shapes
	if wShape[1] != outDim || wShape[2] != packedPerRow {
		return nil, fmt.Errorf("%s.weight: expected [%d, %d, %d], got %v",
			baseName, numExperts, outDim, packedPerRow, wShape)
	}

	// Per-expert slicing
	wElems, ok := checkedProduct(outDim, packedPerRow)
	if !ok {
		return nil, fmt.Errorf("%s.weight per-expert element count overflows", baseName)
	}
	sbElems, ok := checkedProduct(outDim, numGroups)
	if !ok {
		return nil, fmt.Errorf("%s scale/bias per-expert element count overflows", baseName)
	}
	wStride, ok := checkedProduct(wElems, 4) // bytes per expert in weight
	if !ok {
		return nil, fmt.Errorf("%s.weight byte stride overflows", baseName)
	}
	sStride, ok := checkedProduct(sbElems, 2) // bytes per expert in scales (BF16)
	if !ok {
		return nil, fmt.Errorf("%s.scales byte stride overflows", baseName)
	}
	bStride := sStride // bytes per expert in biases (BF16)
	wantW, ok := checkedProduct(wStride, numExperts)
	if !ok {
		return nil, fmt.Errorf("%s.weight total byte size overflows", baseName)
	}
	wantS, ok := checkedProduct(sStride, numExperts)
	if !ok {
		return nil, fmt.Errorf("%s.scales total byte size overflows", baseName)
	}
	wantB, ok := checkedProduct(bStride, numExperts)
	if !ok {
		return nil, fmt.Errorf("%s.biases total byte size overflows", baseName)
	}
	if len(wRaw) < wantW || len(sRaw) < wantS || len(bRaw) < wantB {
		return nil, fmt.Errorf("%s: raw tensor data shorter than expected expert strides", baseName)
	}

	experts := make([]*quant.MLXQuantWeight, numExperts)
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

		experts[e] = &quant.MLXQuantWeight{
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

// moeForward runs the MoE forward pass: router → top-k → expert MLPs → weighted sum.
func moeForward(x []float32, layer *LlamaLayer, cfg LlamaConfig) []float32 {
	if layer == nil || len(x) == 0 || cfg.NumExperts <= 0 || cfg.MoEIntermediate <= 0 {
		return nil
	}
	h := len(x)
	numExperts := cfg.NumExperts
	numActive := cfg.NumExpertsPerTok
	if numActive <= 0 {
		numActive = 8
	}
	if numActive > numExperts {
		numActive = numExperts
	}

	// Router: compute logits for each expert
	routerLogits := make([]float32, numExperts)
	if layer.RouterW != nil {
		quant.GemvMLQ(routerLogits, x, layer.RouterW)
	}

	// Softmax over router logits
	maxLogit := routerLogits[0]
	for _, v := range routerLogits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var expSum float32
	for i := range routerLogits {
		routerLogits[i] = float32(math.Exp(float64(routerLogits[i] - maxLogit)))
		expSum += routerLogits[i]
	}
	if expSum <= 0 || math.IsNaN(float64(expSum)) || math.IsInf(float64(expSum), 0) {
		return nil
	}
	for i := range routerLogits {
		routerLogits[i] /= expSum
	}

	// Top-k selection
	type expertScore struct {
		id    int
		score float32
	}
	selected := make([]expertScore, 0, numActive)
	for i := 0; i < numActive; i++ {
		bestID := -1
		bestScore := float32(-1)
		for j, s := range routerLogits {
			if s > bestScore {
				// Check not already selected
				alreadyPicked := false
				for _, sel := range selected {
					if sel.id == j {
						alreadyPicked = true
						break
					}
				}
				if !alreadyPicked {
					bestID = j
					bestScore = s
				}
			}
		}
		if bestID >= 0 {
			selected = append(selected, expertScore{id: bestID, score: bestScore})
		}
	}

	// Normalize selected weights (norm_topk_prob)
	if cfg.NormTopKProb {
		var sum float32
		for _, s := range selected {
			sum += s.score
		}
		if sum > 0 {
			for i := range selected {
				selected[i].score /= sum
			}
		}
	}

	// Run selected experts in parallel and accumulate weighted output
	moeInter := cfg.MoEIntermediate
	out := make([]float32, h)

	type expertResult struct {
		down   []float32
		weight float32
	}
	results := make([]expertResult, len(selected))
	var wg sync.WaitGroup
	for si, exp := range selected {
		eid := exp.id
		if eid < 0 || eid >= len(layer.ExpertGateW) || eid >= len(layer.ExpertUpW) || eid >= len(layer.ExpertDownW) || layer.ExpertGateW[eid] == nil || layer.ExpertUpW[eid] == nil || layer.ExpertDownW[eid] == nil {
			continue
		}
		wg.Add(1)
		go func(idx int, expertID int, w float32) {
			defer wg.Done()
			// Expert MLP: gate_proj → SiLU × up_proj → down_proj
			gate := make([]float32, moeInter)
			up := make([]float32, moeInter)
			quant.GemvMLQ(gate, x, layer.ExpertGateW[expertID])
			quant.GemvMLQ(up, x, layer.ExpertUpW[expertID])
			simd.VecSiLUMul(gate, gate, up)
			down := make([]float32, h)
			quant.GemvMLQ(down, gate, layer.ExpertDownW[expertID])
			results[idx] = expertResult{down: down, weight: w}
		}(si, eid, exp.score)
	}
	wg.Wait()

	for _, r := range results {
		if r.down == nil {
			continue
		}
		for i := range out {
			out[i] += r.weight * r.down[i]
		}
	}

	return out
}
