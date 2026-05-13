package model

import "testing"

func TestNewMTPVerifierPlan(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 128}}
	plan, err := NewMTPVerifierPlan(m, 7, []int{10, 11}, 42)
	if err != nil {
		t.Fatalf("NewMTPVerifierPlan: %v", err)
	}
	if plan.InputToken != 7 || plan.StartPos != 42 {
		t.Fatalf("unexpected plan metadata: %+v", plan)
	}
	if !sameInts(plan.DraftedTokens, []int{10, 11}) {
		t.Fatalf("DraftedTokens=%v want [10 11]", plan.DraftedTokens)
	}
	if !sameInts(plan.VerifierTokens, []int{7, 10, 11}) {
		t.Fatalf("VerifierTokens=%v want [7 10 11]", plan.VerifierTokens)
	}
	if !sameInts(plan.Positions, []int{42, 43, 44}) {
		t.Fatalf("Positions=%v want [42 43 44]", plan.Positions)
	}

	// Plan owns mutable inputs.
	drafted := []int{1, 2}
	plan, err = NewMTPVerifierPlan(m, 3, drafted, 0)
	if err != nil {
		t.Fatalf("NewMTPVerifierPlan: %v", err)
	}
	drafted[0] = 99
	if !sameInts(plan.DraftedTokens, []int{1, 2}) || !sameInts(plan.VerifierTokens, []int{3, 1, 2}) {
		t.Fatalf("plan aliases mutable draft slice: %+v", plan)
	}
}

func TestNewMTPVerifierPlanValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 4}}
	if _, err := NewMTPVerifierPlan(nil, 1, nil, 0); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := NewMTPVerifierPlan(&LlamaModel{}, 1, nil, 0); err == nil {
		t.Fatal("accepted invalid vocab")
	}
	if _, err := NewMTPVerifierPlan(m, 1, nil, -1); err == nil {
		t.Fatal("accepted negative start position")
	}
	if _, err := NewMTPVerifierPlan(m, -1, nil, 0); err == nil {
		t.Fatal("accepted negative input token")
	}
	if _, err := NewMTPVerifierPlan(m, 4, nil, 0); err == nil {
		t.Fatal("accepted input token outside vocab")
	}
	if _, err := NewMTPVerifierPlan(m, 1, []int{-2}, 0); err == nil {
		t.Fatal("accepted negative drafted token")
	}
	if _, err := NewMTPVerifierPlan(m, 1, []int{4}, 0); err == nil {
		t.Fatal("accepted drafted token outside vocab")
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := NewMTPVerifierPlan(m, 1, []int{2}, maxInt); err == nil {
		t.Fatal("accepted overflowing verifier positions")
	}
}
