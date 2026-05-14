package placement

import (
	"fmt"
)

// Tier represents where a weight set lives.
type Tier int

const (
	TierGPU       Tier = iota // GPU VRAM
	TierPinnedCPU             // Pinned CPU RAM (fast DMA)
	TierMmap                  // mmap-backed (OS page cache)
)

func (t Tier) String() string {
	switch t {
	case TierGPU:
		return "GPU"
	case TierPinnedCPU:
		return "CPU-pinned"
	case TierMmap:
		return "mmap"
	default:
		return "unknown"
	}
}

// LayerPlacement describes where a single transformer layer's weights live.
type LayerPlacement struct {
	Layer    int
	Location Tier
	WeightMB float64 // estimated weight size in MB
}

// PlacementPlan describes the full model placement across tiers.
type PlacementPlan struct {
	Layers          []LayerPlacement
	ResidentMB      float64 // embeddings + LM head + norms + work buffers
	GPULayers       int
	CPULayers       int
	MmapLayers      int
	TotalGPUMB      float64
	TotalCPUMB      float64
	AvailGPUMB      float64
	NVFP4ResidentMB float64
	NVFP4LayerMB    float64
	NVFP4ExpertMB   float64
}

// ModelSizeInfo provides the dimensions needed for placement planning.
type ModelSizeInfo struct {
	NumLayers         int
	HiddenSize        int
	Intermediate      int
	NumHeads          int
	NumKVHeads        int
	HeadDim           int
	GlobalHeadDim     int
	VocabSize         int
	HiddenPerLayer    int
	QuantBits         int    // 0 = fp32/bf16, 4 = int4/fp4
	QuantFormat       string // "", "mlx", "gptq", "nvfp4"
	IsGemma4          bool
	HasDoubleWideMLP  bool // Gemma4 KV-shared layers
	NumKVSharedLayers int
	NumExperts        int
	MoEIntermediate   int
}

// EstimateLayerWeightBytes estimates GPU VRAM for one transformer layer.
func EstimateLayerWeightBytes(info ModelSizeInfo, layerIdx int) int64 {
	h := nonNegativeInt64(info.HiddenSize)
	inter := nonNegativeInt64(info.Intermediate)
	numKVHeads := nonNegativeInt64(info.NumKVHeads)
	numHeads := nonNegativeInt64(info.NumHeads)

	headDim := nonNegativeInt64(info.HeadDim)
	if info.GlobalHeadDim > 0 && info.IsGemma4 {
		// Full attention layers have different head dim
		// Simplified: use max for budget estimation
		if headDim < nonNegativeInt64(info.GlobalHeadDim) {
			headDim = nonNegativeInt64(info.GlobalHeadDim)
		}
	}

	qDim := saturatingMulInt64(numHeads, headDim)
	kvDim := saturatingMulInt64(numKVHeads, headDim)

	// Double-wide MLP for KV-shared layers
	layerInter := inter
	if info.HasDoubleWideMLP && info.NumKVSharedLayers > 0 {
		firstShared := info.NumLayers - info.NumKVSharedLayers
		if layerIdx >= firstShared {
			layerInter = saturatingMulInt64(inter, 2)
		}
	}

	var bytes int64
	add := func(v int64) { bytes = saturatingAddInt64(bytes, v) }

	if isNVFP4(info) {
		// NVFP4: two FP4 values per byte plus one F8 scale per 16 input values
		// and one scalar F32 secondary scale per tensor.
		pack := func(inD, outD int64) int64 { return estimateNVFP4MatrixBytes(inD, outD) }
		add(pack(h, qDim))       // Q proj
		add(pack(h, kvDim))      // K proj
		add(pack(h, kvDim))      // V proj
		add(pack(qDim, h))       // O proj
		add(pack(h, layerInter)) // Gate proj
		add(pack(h, layerInter)) // Up proj
		add(pack(layerInter, h)) // Down proj
	} else if info.QuantBits == 4 {
		// INT4 quantized: packed weights + scales + biases
		// Packed: inDim * outDim / 8 bytes (4-bit)
		// Scales: outDim * numGroups * 4 bytes
		// Biases: outDim * numGroups * 4 bytes
		groupSize := int64(64) // MLX default
		pack := func(inD, outD int64) int64 {
			if inD <= 0 || outD <= 0 {
				return 0
			}
			numGroups := divCeilInt64(inD, groupSize)
			packed := saturatingMulInt64(inD, outD) / 8
			scales := saturatingMulInt64(saturatingMulInt64(outD, numGroups), 4)
			biases := saturatingMulInt64(saturatingMulInt64(outD, numGroups), 4)
			return saturatingAddInt64(packed, saturatingAddInt64(scales, biases))
		}
		add(pack(h, qDim))       // Q proj
		add(pack(h, kvDim))      // K proj
		add(pack(h, kvDim))      // V proj
		add(pack(qDim, h))       // O proj
		add(pack(h, layerInter)) // Gate proj
		add(pack(h, layerInter)) // Up proj
		add(pack(layerInter, h)) // Down proj
	} else {
		// FP32/BF16: full weight matrices
		elemSize := int64(4) // f32; bf16 would be 2
		matBytes := func(inD, outD int64) int64 {
			return saturatingMulInt64(saturatingMulInt64(inD, outD), elemSize)
		}
		add(matBytes(h, qDim))
		add(matBytes(h, kvDim))
		add(matBytes(h, kvDim))
		add(matBytes(qDim, h))
		add(matBytes(h, layerInter))
		add(matBytes(h, layerInter))
		add(matBytes(layerInter, h))
	}

	// Norm weights (small)
	add(saturatingMulInt64(h, 4*4))       // InputNorm + PostNorm + PreFFNNorm + PostFFNNorm
	add(saturatingMulInt64(headDim, 4*2)) // QNorm + KNorm

	// PLI weights (Gemma4)
	if info.HiddenPerLayer > 0 {
		hpl := nonNegativeInt64(info.HiddenPerLayer)
		add(saturatingMulInt64(saturatingMulInt64(h, hpl), 4)) // PLIGate
		add(saturatingMulInt64(saturatingMulInt64(hpl, h), 4)) // PLIProj
		add(saturatingMulInt64(h, 4))                          // PLIPostNorm
	}

	// KV cache estimate (per token, assume 1024 tokens)
	kvPerToken := saturatingMulInt64(kvDim, 4*2) // K + V, float32
	add(saturatingMulInt64(kvPerToken, 1024))

	return bytes
}

// EstimateResidentBytes estimates GPU VRAM for permanent (non-layer) tensors.
func EstimateResidentBytes(info ModelSizeInfo) int64 {
	h := nonNegativeInt64(info.HiddenSize)
	vocab := nonNegativeInt64(info.VocabSize)

	var bytes int64
	add := func(v int64) { bytes = saturatingAddInt64(bytes, v) }
	matrixBytes := func(rows, cols, elemSize int64) int64 {
		return saturatingMulInt64(saturatingMulInt64(rows, cols), elemSize)
	}

	// Embedding table. Inspected NVIDIA NVFP4 Qwen/Gemma checkpoints keep
	// embeddings in BF16, so estimate them as 2-byte resident tensors.
	if isNVFP4(info) {
		add(matrixBytes(vocab, h, 2))
	} else if info.QuantBits == 4 {
		add(divCeilInt64(saturatingMulInt64(vocab, h), 2)) // packed INT4
	} else {
		add(matrixBytes(vocab, h, 4)) // F32
	}

	// LM head (often same size as embedding, or quantized). Inspected NVIDIA
	// NVFP4 Qwen checkpoints keep LM head in BF16; Gemma4 may omit/untie it.
	if isNVFP4(info) {
		add(matrixBytes(vocab, h, 2))
	} else if info.QuantBits == 4 {
		add(divCeilInt64(saturatingMulInt64(vocab, h), 2))
	} else {
		add(matrixBytes(vocab, h, 4))
	}

	// Final norm
	add(saturatingMulInt64(h, 4))

	// RoPE tables (small)
	add(saturatingMulInt64(saturatingMulInt64(2048, nonNegativeInt64(info.HeadDim)), 4))

	// Work buffers (hidden, residual, normed, q, k, v, attn, gate, up, down)
	maxHeadDim := nonNegativeInt64(info.HeadDim)
	if nonNegativeInt64(info.GlobalHeadDim) > maxHeadDim {
		maxHeadDim = nonNegativeInt64(info.GlobalHeadDim)
	}
	maxQDim := saturatingMulInt64(nonNegativeInt64(info.NumHeads), maxHeadDim)
	maxInter := nonNegativeInt64(info.Intermediate)
	if info.HasDoubleWideMLP {
		maxInter = saturatingMulInt64(maxInter, 2)
	}
	workElems := int64(0)
	for _, v := range []int64{h, h, h, saturatingMulInt64(maxQDim, 2), maxQDim, maxQDim, saturatingMulInt64(maxInter, 2), h} {
		workElems = saturatingAddInt64(workElems, v)
	}
	add(saturatingMulInt64(workElems, 4))

	// Gemma4 PLI model-level projection
	if info.HiddenPerLayer > 0 {
		hpl := nonNegativeInt64(info.HiddenPerLayer)
		totalPLI := saturatingMulInt64(nonNegativeInt64(info.NumLayers), hpl)
		add(matrixBytes(h, totalPLI, 4))                            // per_layer_model_projection
		add(saturatingMulInt64(hpl, 4))                             // per_layer_projection_norm
		add(saturatingMulInt64(saturatingMulInt64(totalPLI, 4), 2)) // perLayerProjBuf + perLayerEmbedBuf
	}

	return bytes
}

// PlanLayerPlacement decides where each layer goes based on available GPU VRAM.
// gpuLayers < 0 means auto-fit as many as possible. availGPUBytes is explicit
// so placement policy stays independent from CUDA/Vulkan device queries.
func PlanLayerPlacement(info ModelSizeInfo, gpuLayers int, availGPUBytes uint64) PlacementPlan {
	availGPU := uint64ToInt64(availGPUBytes)

	residentBytes := EstimateResidentBytes(info)
	remainingGPU := availGPU - residentBytes
	if remainingGPU < 0 {
		remainingGPU = 0
	}

	numLayers := info.NumLayers
	if numLayers < 0 {
		numLayers = 0
	}

	plan := PlacementPlan{
		Layers:     make([]LayerPlacement, numLayers),
		ResidentMB: bytesToMB(residentBytes),
		AvailGPUMB: bytesToMB(availGPU),
	}
	if isNVFP4(info) {
		plan.NVFP4ResidentMB = bytesToMB(residentBytes)
	}

	if gpuLayers < 0 {
		// Auto-fit: place layers on GPU until budget exhausted
		gpuLayers = 0
		used := int64(0)
		for i := 0; i < numLayers; i++ {
			layerBytes := EstimateLayerWeightBytes(info, i)
			if layerBytes <= remainingGPU-used {
				gpuLayers++
				used += layerBytes
			} else {
				break
			}
		}
	}

	var totalGPU, totalCPU float64
	for i := 0; i < numLayers; i++ {
		layerBytes := EstimateLayerWeightBytes(info, i)
		layerMB := bytesToMB(layerBytes)
		if i < gpuLayers {
			plan.Layers[i] = LayerPlacement{Layer: i, Location: TierGPU, WeightMB: layerMB}
			plan.GPULayers++
			totalGPU += layerMB
			if isNVFP4(info) {
				plan.NVFP4LayerMB += layerMB
			}
		} else {
			plan.Layers[i] = LayerPlacement{Layer: i, Location: TierMmap, WeightMB: layerMB}
			plan.MmapLayers++
			totalCPU += layerMB
		}
	}
	plan.TotalGPUMB = totalGPU + plan.ResidentMB
	plan.TotalCPUMB = totalCPU
	if isNVFP4(info) {
		plan.NVFP4ExpertMB = bytesToMB(EstimateNVFP4ExpertBytes(info))
	}

	return plan
}

// PrintPlan logs the placement decision.
func (p PlacementPlan) PrintPlan() {
	fmt.Printf("[placement] GPU: %d layers (%.0f MB weights + %.0f MB resident = %.0f MB total)\n",
		p.GPULayers, p.TotalGPUMB-p.ResidentMB, p.ResidentMB, p.TotalGPUMB)
	if p.MmapLayers > 0 {
		fmt.Printf("[placement] mmap: %d layers (%.0f MB)\n", p.MmapLayers, p.TotalCPUMB)
	}
	if p.NVFP4ResidentMB > 0 || p.NVFP4LayerMB > 0 || p.NVFP4ExpertMB > 0 {
		fmt.Printf("[placement] NVFP4: %.0f MB resident, %.0f MB GPU layers, %.0f MB experts\n",
			p.NVFP4ResidentMB, p.NVFP4LayerMB, p.NVFP4ExpertMB)
	}
	fmt.Printf("[placement] available GPU: %.0f MB\n", p.AvailGPUMB)
}

// EstimateNVFP4ExpertBytes estimates one layer's full MoE expert set in NVFP4.
// It is reported separately from dense layer bytes so expert-cache policies can
// size hot slots without conflating router/attention residency.
func EstimateNVFP4ExpertBytes(info ModelSizeInfo) int64 {
	if !isNVFP4(info) || info.NumExperts <= 0 {
		return 0
	}
	h := nonNegativeInt64(info.HiddenSize)
	inter := nonNegativeInt64(info.MoEIntermediate)
	if inter == 0 {
		inter = nonNegativeInt64(info.Intermediate)
	}
	experts := nonNegativeInt64(info.NumExperts)
	perExpert := int64(0)
	add := func(v int64) { perExpert = saturatingAddInt64(perExpert, v) }
	add(estimateNVFP4MatrixBytes(h, inter))
	add(estimateNVFP4MatrixBytes(h, inter))
	add(estimateNVFP4MatrixBytes(inter, h))
	return saturatingMulInt64(experts, perExpert)
}

func bytesToMB(v int64) float64 { return float64(v) / (1024 * 1024) }

func isNVFP4(info ModelSizeInfo) bool {
	return info.QuantFormat == "nvfp4"
}

func estimateNVFP4MatrixBytes(inD, outD int64) int64 {
	if inD <= 0 || outD <= 0 {
		return 0
	}
	params := saturatingMulInt64(inD, outD)
	packed := divCeilInt64(params, 2)
	groups := divCeilInt64(inD, 16)
	f8Scales := saturatingMulInt64(outD, groups)
	return saturatingAddInt64(saturatingAddInt64(packed, f8Scales), 4)
}

func nonNegativeInt64(v int) int64 {
	if v <= 0 {
		return 0
	}
	return int64(v)
}

func divCeilInt64(a, b int64) int64 {
	if a <= 0 || b <= 0 {
		return 0
	}
	return 1 + (a-1)/b
}

func saturatingAddInt64(a, b int64) int64 {
	if a < 0 || b < 0 {
		return 0
	}
	maxInt64 := int64(^uint64(0) >> 1)
	if a > maxInt64-b {
		return maxInt64
	}
	return a + b
}

func saturatingMulInt64(a, b int64) int64 {
	if a < 0 || b < 0 {
		return 0
	}
	maxInt64 := int64(^uint64(0) >> 1)
	if b != 0 && a > maxInt64/b {
		return maxInt64
	}
	return a * b
}
