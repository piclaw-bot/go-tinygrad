package model

import "testing"

func TestRunMTPDrafterStepsProjectionOnly(t *testing.T) {
	m := validDrafterStepBackboneModel()
	d := validProjectionOnlyDrafter()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	got, err := m.RunMTPDrafterSteps(d, state, nil, 3)
	if err != nil {
		t.Fatalf("RunMTPDrafterSteps: %v", err)
	}
	if !sameInts(got.Tokens, []int{1, 0, 0}) {
		t.Fatalf("Tokens=%v want [1 0 0]", got.Tokens)
	}
	if len(got.Logits) != 3 || len(got.Activations) != 3 {
		t.Fatalf("result rows logits/activations=%d/%d", len(got.Logits), len(got.Activations))
	}
	if got.FinalState.PreviousToken != 0 || len(got.FinalState.Activation) != d.BackboneHiddenSize {
		t.Fatalf("FinalState=%+v", got.FinalState)
	}
	got.Activations[0][0] = 99
	if got.FinalState.Activation[0] == 99 {
		t.Fatal("final state aliases stored activation rows")
	}
}

func TestRunMTPDrafterStepsQOnlySynthetic(t *testing.T) {
	m := validDrafterStepBackboneModel()
	d := validDrafterStepScaffold()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	externalKV := &MTPDrafterExternalKV{K: [][]float32{{1, 0}}, V: [][]float32{{0, 1}}, SourceLayers: []int{0}, SeqLen: 1}
	got, err := m.RunMTPDrafterSteps(d, state, externalKV, 2)
	if err != nil {
		t.Fatalf("RunMTPDrafterSteps: %v", err)
	}
	if len(got.Tokens) != 2 || len(got.Logits) != 2 || len(got.Activations) != 2 {
		t.Fatalf("result rows tokens/logits/activations=%d/%d/%d", len(got.Tokens), len(got.Logits), len(got.Activations))
	}
	for i := range got.Logits {
		if len(got.Logits[i]) != m.Config.VocabSize || len(got.Activations[i]) != d.BackboneHiddenSize {
			t.Fatalf("row %d shapes logits/activation=%d/%d", i, len(got.Logits[i]), len(got.Activations[i]))
		}
	}
}

func TestRunMTPDrafterStepsValidation(t *testing.T) {
	m := validDrafterStepBackboneModel()
	d := validProjectionOnlyDrafter()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	if _, err := (*LlamaModel)(nil).RunMTPDrafterSteps(d, state, nil, 1); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := m.RunMTPDrafterSteps(d, state, nil, -1); err == nil {
		t.Fatal("accepted negative draft count")
	}
	if _, err := m.RunMTPDrafterSteps(d, state, nil, maxMTPDraftCount+1); err == nil {
		t.Fatal("accepted oversized draft count")
	}
	got, err := m.RunMTPDrafterSteps(d, state, nil, 0)
	if err != nil || got.FinalState.PreviousToken != state.PreviousToken {
		t.Fatalf("zero-count result=%+v err=%v", got, err)
	}
	got.FinalState.Activation[0] = 99
	if state.Activation[0] == 99 {
		t.Fatal("zero-count final state aliases caller state")
	}
	bad := state
	bad.Activation = bad.Activation[:1]
	if _, err := m.RunMTPDrafterSteps(d, bad, nil, 0); err == nil {
		t.Fatal("accepted malformed zero-count state")
	}
}
