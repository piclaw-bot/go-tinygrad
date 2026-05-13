package model

import (
	"errors"
	"testing"
)

func TestRunMTPVerifierForwardScaffoldValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 8, HiddenSize: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	plan := mustMTPVerifierPlan(t, m, 1, []int{2}, 5)
	_, err := m.RunMTPVerifierForward(plan, make([][]float32, len(m.Layers)), make([][]float32, len(m.Layers)))
	if !errors.Is(err, errMTPVerifierForwardNotImplemented) {
		t.Fatalf("RunMTPVerifierForward err=%v, want not implemented", err)
	}
}

func TestRunMTPVerifierForwardScaffoldRejectsMalformedInputs(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 8, HiddenSize: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	base := mustMTPVerifierPlan(t, m, 1, []int{2}, 5)
	if _, err := (*LlamaModel)(nil).RunMTPVerifierForward(base, nil, nil); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := m.RunMTPVerifierForward(MTPVerifierPlan{}, nil, nil); err == nil {
		t.Fatal("accepted empty plan")
	}
	bad := cloneMTPVerifierPlan(base)
	bad.Positions = bad.Positions[:1]
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted plan with mismatched positions")
	}
	bad = cloneMTPVerifierPlan(base)
	bad.VerifierTokens[0] = 3
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted plan with wrong input token")
	}
	bad = cloneMTPVerifierPlan(base)
	bad.DraftedTokens = append(bad.DraftedTokens, 4)
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted plan with wrong drafted token count")
	}
	bad = cloneMTPVerifierPlan(base)
	bad.DraftedTokens[0] = 4
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted drafted token mismatch with verifier suffix")
	}
	bad = cloneMTPVerifierPlan(base)
	bad.VerifierTokens[1] = 8
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted out-of-vocab verifier token")
	}
	bad = cloneMTPVerifierPlan(base)
	bad.Positions[1] = 99
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted non-contiguous verifier positions")
	}
	bad = cloneMTPVerifierPlan(base)
	bad.StartPos = int(^uint(0) >> 1)
	if _, err := m.RunMTPVerifierForward(bad, nil, nil); err == nil {
		t.Fatal("accepted overflowing verifier positions")
	}
	if _, err := m.RunMTPVerifierForward(base, nil, nil); err == nil {
		t.Fatal("accepted nil/short KV caches")
	}
}

func mustMTPVerifierPlan(t *testing.T, m *LlamaModel, inputToken int, drafted []int, startPos int) MTPVerifierPlan {
	t.Helper()
	plan, err := NewMTPVerifierPlan(m, inputToken, drafted, startPos)
	if err != nil {
		t.Fatalf("NewMTPVerifierPlan: %v", err)
	}
	return plan
}

func cloneMTPVerifierPlan(plan MTPVerifierPlan) MTPVerifierPlan {
	plan.DraftedTokens = append([]int(nil), plan.DraftedTokens...)
	plan.VerifierTokens = append([]int(nil), plan.VerifierTokens...)
	plan.Positions = append([]int(nil), plan.Positions...)
	return plan
}
