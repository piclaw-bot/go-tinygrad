package model

import "testing"

func TestMoeForwardRejectsMalformedInputs(t *testing.T) {
	cfg := LlamaConfig{NumExperts: 0, NumExpertsPerTok: 1, MoEIntermediate: 4}
	if got := moeForward([]float32{1, 2}, &LlamaLayer{}, cfg); got != nil {
		t.Fatalf("zero experts output=%v, want nil", got)
	}
	cfg.NumExperts = 2
	if got := moeForward(nil, &LlamaLayer{}, cfg); got != nil {
		t.Fatalf("nil input output=%v, want nil", got)
	}
	if got := moeForward([]float32{1, 2}, nil, cfg); got != nil {
		t.Fatalf("nil layer output=%v, want nil", got)
	}
	cfg.MoEIntermediate = 0
	if got := moeForward([]float32{1, 2}, &LlamaLayer{}, cfg); got != nil {
		t.Fatalf("zero intermediate output=%v, want nil", got)
	}
}
