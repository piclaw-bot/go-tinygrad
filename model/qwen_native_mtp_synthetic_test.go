package model

import "testing"

func TestNewSyntheticQwenNativeMTPFixture(t *testing.T) {
	m, head, meta, state := NewSyntheticQwenNativeMTPFixture()
	if m == nil || head == nil || !meta.HasNativeMTP {
		t.Fatalf("fixture model/head/meta invalid: m=%v head=%v meta=%+v", m, head, meta)
	}
	if len(state.Hidden) != meta.HiddenSize {
		t.Fatalf("state hidden len=%d want %d", len(state.Hidden), meta.HiddenSize)
	}
	if err := ValidateQwenNativeMTPHead(head, meta); err != nil {
		t.Fatalf("ValidateQwenNativeMTPHead: %v", err)
	}
	plan, err := NewQwenNativeMTPPlan(0, state, 1, meta)
	if err != nil {
		t.Fatalf("NewQwenNativeMTPPlan: %v", err)
	}
	_, drafted, _, err := head.DraftSteps(m, plan.TokenID, plan.State, plan.MaxSteps, 1e-6, meta)
	if err != nil || len(drafted) != 1 {
		t.Fatalf("DraftSteps drafted=%v err=%v", drafted, err)
	}
}
