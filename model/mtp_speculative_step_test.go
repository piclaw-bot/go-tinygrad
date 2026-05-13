package model

import "testing"

func TestRunMTPSpeculativeStepProjectionOnly(t *testing.T) {
	m := newSingleLayerVerifierModel()
	d := validProjectionOnlyDrafterForModel(m)
	state, err := NewMTPDrafterState(0, []float32{1, 0}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	result, err := m.RunMTPSpeculativeStep(d, state, nil, 0, kvCacheK, kvCacheV, MTPSpeculationStats{})
	if err != nil {
		t.Fatalf("RunMTPSpeculativeStep: %v", err)
	}
	if result.Draft.Token != 1 {
		t.Fatalf("draft token=%d want 1", result.Draft.Token)
	}
	if !sameInts(result.Plan.VerifierTokens, []int{0, 1}) {
		t.Fatalf("verifier tokens=%v want [0 1]", result.Plan.VerifierTokens)
	}
	if result.Verifier.Acceptance.AllDraftsAccepted || result.Verifier.Acceptance.AcceptedPrefixLen != 0 {
		t.Fatalf("acceptance=%+v, want first-token rejection", result.Verifier.Acceptance)
	}
	if result.Stats.Steps != 1 || result.Stats.DraftedTokens != 1 || result.Stats.VerifiedTokens != 0 || result.Stats.BonusTokens != 1 {
		t.Fatalf("stats=%+v", result.Stats)
	}
	kvDim, err := m.LayerKVDim(0)
	if err != nil {
		t.Fatalf("LayerKVDim: %v", err)
	}
	if got, want := len(kvCacheK[0]), len(result.Plan.VerifierTokens)*kvDim; got != want {
		t.Fatalf("staged K len=%d want %d", got, want)
	}
}

func TestRunMTPSpeculativeStepValidation(t *testing.T) {
	m := newSingleLayerVerifierModel()
	d := validProjectionOnlyDrafterForModel(m)
	state, err := NewMTPDrafterState(0, []float32{1, 0}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	if _, err := (*LlamaModel)(nil).RunMTPSpeculativeStep(d, state, nil, 0, nil, nil, MTPSpeculationStats{}); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := m.RunMTPSpeculativeStep(d, state, nil, 0, nil, nil, MTPSpeculationStats{}); err == nil {
		t.Fatal("accepted missing verifier KV caches")
	}
	badStats := MTPSpeculationStats{Steps: int(^uint(0) >> 1)}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	if _, err := m.RunMTPSpeculativeStep(d, state, nil, 0, kvCacheK, kvCacheV, badStats); err == nil {
		t.Fatal("accepted overflowing stats")
	}
}

func validProjectionOnlyDrafterForModel(m *LlamaModel) *Gemma4MTPDrafter {
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	return d
}
