package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestMoEForwardGPUMalformedInputs(t *testing.T) {
	cfg := LlamaConfig{HiddenSize: 4, NumExperts: 2, NumExpertsPerTok: 4, MoEIntermediate: 8}
	if got := moeForwardGPU(nil, nil, &LlamaLayer{}, cfg, gpu.NewExpertPool(1, nil), 0, nil); got != nil {
		t.Fatalf("nil xDev returned %#v, want nil", got)
	}
	if got := moeForwardGPU(nil, gpu.NewDevBuf(2), &LlamaLayer{}, cfg, gpu.NewExpertPool(1, nil), 0, nil); got != nil {
		t.Fatalf("short xDev returned %#v, want nil", got)
	}
	badCfg := cfg
	badCfg.NumExperts = 0
	if got := moeForwardGPU(nil, gpu.NewDevBuf(4), &LlamaLayer{}, badCfg, gpu.NewExpertPool(1, nil), 0, nil); got != nil {
		t.Fatalf("zero experts returned %#v, want nil", got)
	}
}

func TestMoEForwardGPUClampsActiveExperts(t *testing.T) {
	cfg := LlamaConfig{HiddenSize: 4, NumExperts: 2, NumExpertsPerTok: 8, MoEIntermediate: 8}
	layer := &LlamaLayer{}
	got := moeForwardGPU(nil, gpu.NewDevBuf(4), layer, cfg, nil, 0, nil)
	if len(got) != cfg.HiddenSize {
		t.Fatalf("len=%d, want %d", len(got), cfg.HiddenSize)
	}
}
