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

func TestEstimateNVFP4MatrixBytes(t *testing.T) {
	// Qwen3 dense q_proj observed layout: U8 [4096,2048] + F8 [4096,256] + F32 scalar.
	got := estimateNVFP4MatrixBytes(4096, 4096)
	want := int64(4096*2048 + 4096*256 + 4)
	if got != want {
		t.Fatalf("estimateNVFP4MatrixBytes=%d want %d", got, want)
	}
}

func TestPlanLayerPlacementReportsNVFP4Breakdown(t *testing.T) {
	info := ModelSizeInfo{NumLayers: 2, HiddenSize: 4096, Intermediate: 12288, NumHeads: 32, NumKVHeads: 8, HeadDim: 128, VocabSize: 151936, QuantBits: 4, QuantFormat: "nvfp4", NumExperts: 128, MoEIntermediate: 768}
	plan := PlanLayerPlacement(info, 1, testAvailGPUBytes)
	if plan.NVFP4ResidentMB <= 0 || plan.NVFP4LayerMB <= 0 || plan.NVFP4ExpertMB <= 0 {
		t.Fatalf("missing NVFP4 breakdown: %+v", plan)
	}
	if plan.NVFP4LayerMB >= plan.TotalGPUMB {
		t.Fatalf("NVFP4 layer breakdown=%f should be below total GPU=%f", plan.NVFP4LayerMB, plan.TotalGPUMB)
	}
}

func TestEstimateNVFP4ExpertBytes(t *testing.T) {
	info := ModelSizeInfo{HiddenSize: 2048, Intermediate: 768, NumExperts: 128, QuantBits: 4, QuantFormat: "nvfp4"}
	got := EstimateNVFP4ExpertBytes(info)
	wantOneExpert := 2*estimateNVFP4MatrixBytes(2048, 768) + estimateNVFP4MatrixBytes(768, 2048)
	want := int64(128) * wantOneExpert
	if got != want {
		t.Fatalf("EstimateNVFP4ExpertBytes=%d want %d", got, want)
	}
}

func TestEstimateLayerWeightBytesNVFP4(t *testing.T) {
	info := ModelSizeInfo{NumLayers: 1, HiddenSize: 4096, Intermediate: 12288, NumHeads: 32, NumKVHeads: 8, HeadDim: 128, QuantBits: 4, QuantFormat: "nvfp4"}
	got := EstimateLayerWeightBytes(info, 0)
	wantWeights := estimateNVFP4MatrixBytes(4096, 4096) + 2*estimateNVFP4MatrixBytes(4096, 1024) + estimateNVFP4MatrixBytes(4096, 4096) + 2*estimateNVFP4MatrixBytes(4096, 12288) + estimateNVFP4MatrixBytes(12288, 4096)
	if got <= wantWeights {
		t.Fatalf("NVFP4 layer estimate=%d want above raw weight estimate=%d due to norms/KV", got, wantWeights)
	}
}

func TestEstimateResidentBytesNVFP4UsesBF16Embeddings(t *testing.T) {
	info := ModelSizeInfo{HiddenSize: 4, VocabSize: 8, QuantBits: 4, QuantFormat: "nvfp4"}
	got := EstimateResidentBytes(info)
	// Embedding + LM head are BF16 for inspected NVIDIA NVFP4 checkpoints: 8*4*2*2.
	if got < 128 {
		t.Fatalf("resident estimate=%d want at least BF16 embedding+LM head bytes", got)
	}
	if got >= EstimateResidentBytes(ModelSizeInfo{HiddenSize: 4, VocabSize: 8, QuantBits: 0}) {
		t.Fatalf("NVFP4 resident estimate=%d should stay below F32 resident estimate", got)
	}
}

func TestEstimateLayerWeightBytes(t *testing.T) {
	oddGroups := ModelSizeInfo{NumLayers: 1, HiddenSize: 65, Intermediate: 65, NumHeads: 1, NumKVHeads: 1, HeadDim: 65, QuantBits: 4}
	if got := EstimateLayerWeightBytes(oddGroups, 0); got <= 0 {
		t.Fatalf("odd-group quantized estimate=%d, want positive", got)
	}

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

func TestPlacementHandlesInvalidAndHugeInputs(t *testing.T) {
	bad := ModelSizeInfo{
		NumLayers:    -5,
		HiddenSize:   -1024,
		Intermediate: -4096,
		NumHeads:     -8,
		NumKVHeads:   -4,
		HeadDim:      -128,
		VocabSize:    -32000,
		QuantBits:    4,
	}
	if got := EstimateLayerWeightBytes(bad, 0); got != 0 {
		t.Fatalf("negative layer estimate=%d, want 0", got)
	}
	if got := EstimateResidentBytes(bad); got != 0 {
		t.Fatalf("negative resident estimate=%d, want 0", got)
	}
	huge := ModelSizeInfo{NumLayers: 1, HiddenSize: int(^uint(0) >> 2), Intermediate: int(^uint(0) >> 2), NumHeads: int(^uint(0) >> 4), NumKVHeads: int(^uint(0) >> 4), HeadDim: int(^uint(0) >> 4), VocabSize: int(^uint(0) >> 4)}
	if got := EstimateLayerWeightBytes(huge, 0); got <= 0 {
		t.Fatalf("huge layer estimate=%d, want positive saturated value", got)
	}
	if got := EstimateResidentBytes(huge); got <= 0 {
		t.Fatalf("huge resident estimate=%d, want positive saturated value", got)
	}
	oddResident := ModelSizeInfo{HiddenSize: 3, VocabSize: 3, QuantBits: 4}
	if got := EstimateResidentBytes(oddResident); got < 9 {
		t.Fatalf("odd resident estimate=%d, want at least two ceil-packed 3x3 matrices", got)
	}
	plan := PlanLayerPlacement(bad, -1, ^uint64(0))
	if len(plan.Layers) != 0 || plan.GPULayers != 0 || plan.MmapLayers != 0 || plan.AvailGPUMB <= 0 {
		t.Fatalf("unexpected plan for invalid/huge inputs: %+v", plan)
	}
}

func TestPlanLayerPlacementManualLayerCountClampsToModel(t *testing.T) {
	info := ModelSizeInfo{
		NumLayers:    2,
		HiddenSize:   64,
		Intermediate: 128,
		NumHeads:     4,
		NumKVHeads:   2,
		HeadDim:      16,
		VocabSize:    256,
	}
	plan := PlanLayerPlacement(info, 999, testAvailGPUBytes)
	if len(plan.Layers) != 2 || plan.GPULayers != 2 || plan.MmapLayers != 0 {
		t.Fatalf("manual gpu layer count should clamp by model layers: %+v", plan)
	}
}
