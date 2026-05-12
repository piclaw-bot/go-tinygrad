package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func TestPrefillGPURejectsMalformedInputs(t *testing.T) {
	if (&GPUModel{}).prefillGPU([]int{1, 2}) != nil {
		t.Fatal("prefillGPU accepted nil CPU model")
	}
	g := &GPUModel{CPU: &LlamaModel{Config: LlamaConfig{HiddenSize: 4, NumHeads: 0, NumKVHeads: 1, Intermediate: 8, VocabSize: 2}, EmbedTokens: tensor.FromFloat32(make([]float32, 8), []int{2, 4})}}
	if g.prefillGPU([]int{0, 1}) != nil {
		t.Fatal("prefillGPU accepted zero NumHeads")
	}
	g.CPU.Config.NumHeads = 3
	if g.prefillGPU([]int{0, 1}) != nil {
		t.Fatal("prefillGPU accepted non-divisible head dims")
	}
	g.CPU.Config.NumHeads = 2
	if g.prefillGPU([]int{0, -1}) != nil {
		t.Fatal("prefillGPU accepted negative token")
	}
	if g.prefillGPU([]int{0, 2}) != nil {
		t.Fatal("prefillGPU accepted out-of-range token")
	}
	g.CPU.EmbedTokens = tensor.FromFloat32(make([]float32, 4), []int{1, 4})
	if g.prefillGPU([]int{0, 1}) != nil {
		t.Fatal("prefillGPU accepted short embedding backing data")
	}
}
