package model

import "testing"

func TestSyntheticQwenNativeMTPCorrectnessHarness(t *testing.T) {
	meta := testQwenNativeMTPMeta()
	head := syntheticQwenNativeMTPHead(meta)
	m := syntheticQwenMTPMainModel(meta)
	state := QwenNativeMTPDraftState{Hidden: []float32{0, 1, 0, 0}}
	plan, err := NewQwenNativeMTPPlan(0, state, 2, meta)
	if err != nil {
		t.Fatalf("NewQwenNativeMTPPlan: %v", err)
	}
	_, drafted, _, err := head.DraftSteps(m, plan.TokenID, plan.State, plan.MaxSteps, 1e-6, meta)
	if err != nil {
		t.Fatalf("DraftSteps: %v", err)
	}
	verifier := append([]int(nil), drafted...)
	verifier = append(verifier, 1)
	res, err := RunQwenNativeMTPPlan(head, m, plan, verifier, 1e-6, meta)
	if err != nil {
		t.Fatalf("RunQwenNativeMTPPlan: %v", err)
	}
	if !res.Acceptance.AllDraftsAccepted || res.Stats.AcceptanceRate() != 1 {
		t.Fatalf("acceptance=%+v stats=%+v", res.Acceptance, res.Stats)
	}
}
