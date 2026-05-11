package placement

import (
	"testing"
)

const testAvailGPUBytes = 12 * 1024 * 1024 * 1024

func TestPlanLayerPlacementSmall(t *testing.T) {
	// SmolLM2-135M: should fit entirely on GPU
	info := ModelSizeInfo{
		NumLayers:    30,
		HiddenSize:   576,
		Intermediate: 1536,
		NumHeads:     9,
		NumKVHeads:   3,
		HeadDim:      64,
		VocabSize:    49152,
		QuantBits:    0, // BF16
	}
	plan := PlanLayerPlacement(info, -1, testAvailGPUBytes)
	plan.PrintPlan()
	t.Logf("SmolLM2-135M: %d GPU layers, %d mmap, %.0f MB total GPU",
		plan.GPULayers, plan.MmapLayers, plan.TotalGPUMB)
}

func TestPlanLayerPlacementMedium(t *testing.T) {
	// Qwen2.5-7B MLX4: may not fit entirely
	info := ModelSizeInfo{
		NumLayers:    28,
		HiddenSize:   3584,
		Intermediate: 18944,
		NumHeads:     28,
		NumKVHeads:   4,
		HeadDim:      128,
		VocabSize:    152064,
		QuantBits:    4,
	}
	plan := PlanLayerPlacement(info, -1, testAvailGPUBytes)
	plan.PrintPlan()
	t.Logf("Qwen2.5-7B MLX4: %d GPU layers, %d mmap, %.0f MB total GPU",
		plan.GPULayers, plan.MmapLayers, plan.TotalGPUMB)
}

func TestPlanLayerPlacementGemma4(t *testing.T) {
	// Gemma4-E2B MLX4
	info := ModelSizeInfo{
		NumLayers:         35,
		HiddenSize:        1536,
		Intermediate:      6144,
		NumHeads:          8,
		NumKVHeads:        4,
		HeadDim:           256,
		GlobalHeadDim:     512,
		VocabSize:         262144,
		HiddenPerLayer:    256,
		QuantBits:         4,
		IsGemma4:          true,
		HasDoubleWideMLP:  true,
		NumKVSharedLayers: 20,
	}
	plan := PlanLayerPlacement(info, -1, testAvailGPUBytes)
	plan.PrintPlan()
	t.Logf("Gemma4-E2B MLX4: %d GPU layers, %d mmap, %.0f MB total GPU",
		plan.GPULayers, plan.MmapLayers, plan.TotalGPUMB)

	// Test explicit layer count
	plan10 := PlanLayerPlacement(info, 10, testAvailGPUBytes)
	t.Logf("Gemma4-E2B MLX4 (10 GPU layers): %d GPU, %d mmap, %.0f MB GPU",
		plan10.GPULayers, plan10.MmapLayers, plan10.TotalGPUMB)
}

func TestEstimateLayerWeightBytes(t *testing.T) {
	// Gemma4 layer size estimates
	info := ModelSizeInfo{
		NumLayers:         35,
		HiddenSize:        1536,
		Intermediate:      6144,
		NumHeads:          8,
		NumKVHeads:        4,
		HeadDim:           256,
		GlobalHeadDim:     512,
		VocabSize:         262144,
		HiddenPerLayer:    256,
		QuantBits:         4,
		IsGemma4:          true,
		HasDoubleWideMLP:  true,
		NumKVSharedLayers: 20,
	}
	for _, l := range []int{0, 14, 15, 34} {
		bytes := EstimateLayerWeightBytes(info, l)
		t.Logf("Gemma4 layer %d: %.1f MB", l, float64(bytes)/(1024*1024))
	}
	resBytes := EstimateResidentBytes(info)
	t.Logf("Gemma4 resident: %.1f MB", float64(resBytes)/(1024*1024))
}
