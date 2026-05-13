package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func TestRunMTPVerifierForwardZeroDraftZeroLayer(t *testing.T) {
	m := newZeroLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 1, nil, 7)
	result, err := m.RunMTPVerifierForward(plan, nil, nil)
	if err != nil {
		t.Fatalf("RunMTPVerifierForward: %v", err)
	}
	if !sameInts(result.VerifierTokens, []int{1}) {
		t.Fatalf("VerifierTokens=%v want [1]", result.VerifierTokens)
	}
	if !sameInts(result.Acceptance.OutputTokens, []int{0}) {
		t.Fatalf("OutputTokens=%v want [0]", result.Acceptance.OutputTokens)
	}
	if len(result.Logits) != 1 || len(result.Logits[0]) != m.Config.VocabSize {
		t.Fatalf("logits shape=%d/%d", len(result.Logits), len(result.Logits[0]))
	}
	if len(result.FinalActivation) != 2 || result.FinalActivation[0] != 0 || result.FinalActivation[1] < 1.41 || result.FinalActivation[1] > 1.42 {
		t.Fatalf("FinalActivation=%v want approximately [0 sqrt(2)]", result.FinalActivation)
	}
}

func TestRunMTPVerifierForwardOneDraftZeroLayer(t *testing.T) {
	m := newZeroLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 4)
	result, err := m.RunMTPVerifierForward(plan, nil, nil)
	if err != nil {
		t.Fatalf("RunMTPVerifierForward: %v", err)
	}
	if !result.Acceptance.AllDraftsAccepted || result.Acceptance.AcceptedPrefixLen != 1 || result.Acceptance.BonusToken != 0 {
		t.Fatalf("acceptance=%+v, want all accepted prefix=1 bonus=0", result.Acceptance)
	}
	if !sameInts(result.Acceptance.OutputTokens, []int{1, 0}) {
		t.Fatalf("OutputTokens=%v want [1 0]", result.Acceptance.OutputTokens)
	}
}

func TestRunMTPVerifierForwardFirstTokenRejectionZeroLayer(t *testing.T) {
	m := newZeroLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{2}, 4)
	result, err := m.RunMTPVerifierForward(plan, nil, nil)
	if err != nil {
		t.Fatalf("RunMTPVerifierForward: %v", err)
	}
	if result.Acceptance.AllDraftsAccepted || result.Acceptance.AcceptedPrefixLen != 0 || result.Acceptance.BonusToken != 1 {
		t.Fatalf("acceptance=%+v, want first rejection bonus=1", result.Acceptance)
	}
	if !sameInts(result.Acceptance.OutputTokens, []int{1}) {
		t.Fatalf("OutputTokens=%v want [1]", result.Acceptance.OutputTokens)
	}
}

func TestRunMTPVerifierForwardScaffoldRejectsMalformedInputs(t *testing.T) {
	m := newZeroLayerVerifierModel()
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
	bad.VerifierTokens[1] = 3
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
	withLayer := newZeroLayerVerifierModel()
	withLayer.Config.NumLayers = 1
	withLayer.Layers = []LlamaLayer{{}}
	layerPlan := mustMTPVerifierPlan(t, withLayer, 1, nil, 0)
	if _, err := withLayer.RunMTPVerifierForward(layerPlan, nil, nil); err == nil {
		t.Fatal("accepted nil/short KV caches")
	}
}

func newZeroLayerVerifierModel() *LlamaModel {
	return &LlamaModel{
		Config: LlamaConfig{VocabSize: 3, HiddenSize: 2, NumLayers: 0, NumHeads: 1, NumKVHeads: 1, HeadDim: 2, RMSNormEps: 0},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 0,
			0, 1,
			1, 1,
		}, []int{3, 2}),
		Norm: tensor.Ones([]int{2}),
		LMHead: tensor.FromFloat32([]float32{
			0, 1,
			1, 1,
			1, 0,
		}, []int{3, 2}),
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
