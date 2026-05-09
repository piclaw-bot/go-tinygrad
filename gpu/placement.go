package gpu

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
	h := int64(info.HiddenSize)
	inter := int64(info.Intermediate)
	numKVHeads := int64(info.NumKVHeads)
	numHeads := int64(info.NumHeads)

	headDim := int64(info.HeadDim)
	if info.GlobalHeadDim > 0 && info.IsGemma4 {
		// Full attention layers have different head dim
		// Simplified: use max for budget estimation
		if headDim < int64(info.GlobalHeadDim) {
			headDim = int64(info.GlobalHeadDim)
		}
	}

	qDim := numHeads * headDim
	kvDim := numKVHeads * headDim

	// Double-wide MLP for KV-shared layers
	layerInter := inter
	if info.HasDoubleWideMLP && info.NumKVSharedLayers > 0 {
		firstShared := info.NumLayers - info.NumKVSharedLayers
		if layerIdx >= firstShared {
			layerInter = inter * 2
		}
	}

	var bytes int64

	if info.QuantBits == 4 {
		// INT4 quantized: packed weights + scales + biases
		// Packed: inDim * outDim / 8 bytes (4-bit)
		// Scales: outDim * numGroups * 4 bytes
		// Biases: outDim * numGroups * 4 bytes
		groupSize := int64(64) // MLX default
		pack := func(inD, outD int64) int64 {
			numGroups := inD / groupSize
			return inD*outD/8 + outD*numGroups*4 + outD*numGroups*4
		}
		bytes += pack(h, qDim)       // Q proj
		bytes += pack(h, kvDim)      // K proj
		bytes += pack(h, kvDim)      // V proj
		bytes += pack(qDim, h)       // O proj
		bytes += pack(h, layerInter) // Gate proj
		bytes += pack(h, layerInter) // Up proj
		bytes += pack(layerInter, h) // Down proj
	} else {
		// FP32/BF16: full weight matrices
		elemSize := int64(4) // f32; bf16 would be 2
		bytes += h * qDim * elemSize
		bytes += h * kvDim * elemSize
		bytes += h * kvDim * elemSize
		bytes += qDim * h * elemSize
		bytes += h * layerInter * elemSize
		bytes += h * layerInter * elemSize
		bytes += layerInter * h * elemSize
	}

	// Norm weights (small)
	bytes += h * 4 * 4       // InputNorm + PostNorm + PreFFNNorm + PostFFNNorm
	bytes += headDim * 4 * 2 // QNorm + KNorm

	// PLI weights (Gemma4)
	if info.HiddenPerLayer > 0 {
		hpl := int64(info.HiddenPerLayer)
		bytes += h * hpl * 4 // PLIGate
		bytes += hpl * h * 4 // PLIProj
		bytes += h * 4       // PLIPostNorm
	}

	// KV cache estimate (per token, assume 1024 tokens)
	kvPerToken := kvDim * 4 * 2 // K + V, float32
	bytes += kvPerToken * 1024

	return bytes
}

// EstimateResidentBytes estimates GPU VRAM for permanent (non-layer) tensors.
func EstimateResidentBytes(info ModelSizeInfo) int64 {
	h := int64(info.HiddenSize)
	vocab := int64(info.VocabSize)

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
	bytes += 2048 * int64(info.HeadDim) * 4

	// Work buffers (hidden, residual, normed, q, k, v, attn, gate, up, down)
	maxHeadDim := int64(info.HeadDim)
	if int64(info.GlobalHeadDim) > maxHeadDim {
		maxHeadDim = int64(info.GlobalHeadDim)
	}
	maxQDim := int64(info.NumHeads) * maxHeadDim
	maxInter := int64(info.Intermediate)
	if info.HasDoubleWideMLP {
		maxInter *= 2
	}
	bytes += (h + h + h + maxQDim*2 + maxQDim + maxQDim + maxInter*2 + h) * 4

	// Gemma4 PLI model-level projection
	if info.HiddenPerLayer > 0 {
		totalPLI := int64(info.NumLayers) * int64(info.HiddenPerLayer)
		bytes += h * totalPLI * 4               // per_layer_model_projection
		bytes += int64(info.HiddenPerLayer) * 4 // per_layer_projection_norm
		bytes += totalPLI * 4 * 2               // perLayerProjBuf + perLayerEmbedBuf
	}

	return bytes
}

// PlanLayerPlacement decides where each layer goes based on available GPU VRAM.
// gpuLayers < 0 means auto-fit as many as possible.
func PlanLayerPlacement(info ModelSizeInfo, gpuLayers int) PlacementPlan {
	free, _ := MemInfo()
	availGPU := int64(free)

	residentBytes := EstimateResidentBytes(info)
	remainingGPU := availGPU - residentBytes
	if remainingGPU < 0 {
		remainingGPU = 0
	}

	plan := PlacementPlan{
		Layers:     make([]LayerPlacement, info.NumLayers),
		ResidentMB: float64(residentBytes) / (1024 * 1024),
		AvailGPUMB: float64(availGPU) / (1024 * 1024),
	}

	if gpuLayers < 0 {
		// Auto-fit: place layers on GPU until budget exhausted
		gpuLayers = 0
		used := int64(0)
		for i := 0; i < info.NumLayers; i++ {
			layerBytes := EstimateLayerWeightBytes(info, i)
			if used+layerBytes <= remainingGPU {
				gpuLayers++
				used += layerBytes
			} else {
				break
			}
		}
	}

	var totalGPU, totalCPU float64
	for i := 0; i < info.NumLayers; i++ {
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
