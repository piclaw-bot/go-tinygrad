package model

import (
	"errors"
	"testing"
)

func TestRunMTPVerifierForwardScaffoldValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 8, HiddenSize: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	plan, err := NewMTPVerifierPlan(m, 1, []int{2}, 5)
	if err != nil {
		t.Fatalf("NewMTPVerifierPlan: %v", err)
	}
	_, err = m.RunMTPVerifierForward(plan, make([][]float32, len(m.Layers)), make([][]float32, len(m.Layers)))
	if !errors.Is(err, errMTPVerifierForwardNotImplemented) {
		t.Fatalf("RunMTPVerifierForward err=%v, want not implemented", err)
	}
}

func TestRunMTPVerifierForwardScaffoldRejectsMalformedInputs(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 8, HiddenSize: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	plan, err := NewMTPVerifierPlan(m, 1, []int{2}, 5)
	if err != nil {
		t.Fatalf("NewMTPVerifierPlan: %v", err)
	}
	if _, err := (*LlamaModel)(nil).RunMTPVerifierForward(plan, nil, nil); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := m.RunMTPVerifierForward(MTPVerifierPlan{}, nil, nil); err == nil {
		t.Fatal("accepted empty plan")
	}
	bad := plan
	bad.Positions = bad.Positions[:1]
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted plan with mismatched positions")
	}
	bad = plan
	bad.VerifierTokens[0] = 3
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted plan with wrong input token")
	}
	bad = plan
	bad.DraftedTokens = append(bad.DraftedTokens, 4)
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted plan with wrong drafted token count")
	}
	bad = plan
	bad.Positions[1] = 99
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted non-contiguous verifier positions")
	}
	if _, err := m.RunMTPVerifierForward(plan, nil, nil); err == nil {
		t.Fatal("accepted nil/short KV caches")
	}
}
