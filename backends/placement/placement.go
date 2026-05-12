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
	Layers     []LayerPlacement
	ResidentMB float64 // embeddings + LM head + norms + work buffers
	GPULayers  int
	CPULayers  int
	MmapLayers int
	TotalGPUMB float64
	TotalCPUMB float64
	AvailGPUMB float64
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
	QuantBits         int // 0 = fp32/bf16, 4 = int4
	IsGemma4          bool
	HasDoubleWideMLP  bool // Gemma4 KV-shared layers
	NumKVSharedLayers int
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

	if info.QuantBits == 4 {
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

	// Embedding table
	if info.QuantBits == 4 {
		bytes += vocab * h / 2 // packed INT4
	} else {
		bytes += vocab * h * 4 // F32
	}

	// LM head (often same size as embedding, or quantized)
	if info.QuantBits == 4 {
		bytes += vocab * h / 2
	} else {
		bytes += vocab * h * 4
	}

	// Final norm
	bytes += h * 4

	// RoPE tables (small)
	bytes += 2048 * nonNegativeInt64(info.HeadDim) * 4

	// Work buffers (hidden, residual, normed, q, k, v, attn, gate, up, down)
	maxHeadDim := nonNegativeInt64(info.HeadDim)
	if nonNegativeInt64(info.GlobalHeadDim) > maxHeadDim {
		maxHeadDim = nonNegativeInt64(info.GlobalHeadDim)
	}
	maxQDim := nonNegativeInt64(info.NumHeads) * maxHeadDim
	maxInter := nonNegativeInt64(info.Intermediate)
	if info.HasDoubleWideMLP {
		maxInter *= 2
	}
	bytes += (h + h + h + maxQDim*2 + maxQDim + maxQDim + maxInter*2 + h) * 4

	// Gemma4 PLI model-level projection
	if info.HiddenPerLayer > 0 {
		totalPLI := nonNegativeInt64(info.NumLayers) * nonNegativeInt64(info.HiddenPerLayer)
		bytes += h * totalPLI * 4                          // per_layer_model_projection
		bytes += nonNegativeInt64(info.HiddenPerLayer) * 4 // per_layer_projection_norm
		bytes += totalPLI * 4 * 2                          // perLayerProjBuf + perLayerEmbedBuf
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
		ResidentMB: float64(residentBytes) / (1024 * 1024),
		AvailGPUMB: float64(availGPU) / (1024 * 1024),
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
		layerMB := float64(EstimateLayerWeightBytes(info, i)) / (1024 * 1024)
		if i < gpuLayers {
			plan.Layers[i] = LayerPlacement{Layer: i, Location: TierGPU, WeightMB: layerMB}
			plan.GPULayers++
			totalGPU += layerMB
		} else {
			plan.Layers[i] = LayerPlacement{Layer: i, Location: TierMmap, WeightMB: layerMB}
			plan.MmapLayers++
			totalCPU += layerMB
		}
	}
	plan.TotalGPUMB = totalGPU + plan.ResidentMB
	plan.TotalCPUMB = totalCPU

	return plan
}

// PrintPlan logs the placement decision.
func (p PlacementPlan) PrintPlan() {
	fmt.Printf("[placement] GPU: %d layers (%.0f MB weights + %.0f MB resident = %.0f MB total)\n",
		p.GPULayers, p.TotalGPUMB-p.ResidentMB, p.ResidentMB, p.TotalGPUMB)
	if p.MmapLayers > 0 {
		fmt.Printf("[placement] mmap: %d layers (%.0f MB)\n", p.MmapLayers, p.TotalCPUMB)
	}
	fmt.Printf("[placement] available GPU: %.0f MB\n", p.AvailGPUMB)
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
