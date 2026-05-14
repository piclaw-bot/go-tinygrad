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

func TestRunMTPMultiDraftSpeculativeStepProjectionOnlyFirstRejection(t *testing.T) {
	m := newSingleLayerVerifierModel()
	d := validProjectionOnlyDrafterForModel(m)
	state, err := NewMTPDrafterState(0, []float32{1, 0}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	result, err := m.RunMTPMultiDraftSpeculativeStep(d, state, nil, 0, 2, kvCacheK, kvCacheV, MTPSpeculationStats{})
	if err != nil {
		t.Fatalf("RunMTPMultiDraftSpeculativeStep: %v", err)
	}
	if !sameInts(result.Drafts.Tokens, []int{1, 0}) {
		t.Fatalf("draft tokens=%v want [1 0]", result.Drafts.Tokens)
	}
	if !sameInts(result.Plan.VerifierTokens, []int{0, 1, 0}) {
		t.Fatalf("verifier tokens=%v want [0 1 0]", result.Plan.VerifierTokens)
	}
	if result.Verifier.Acceptance.AllDraftsAccepted || result.Verifier.Acceptance.AcceptedPrefixLen != 0 || result.Verifier.Acceptance.DraftedCount != 2 {
		t.Fatalf("acceptance=%+v, want first rejection for two drafts", result.Verifier.Acceptance)
	}
	if result.Stats.Steps != 1 || result.Stats.DraftedTokens != 2 || result.Stats.VerifiedTokens != 0 || result.Stats.BonusTokens != 1 {
		t.Fatalf("stats=%+v", result.Stats)
	}
}

func TestRunMTPMultiDraftSpeculativeStepProjectionOnlyAllAccepted(t *testing.T) {
	m := newZeroLayerVerifierModel()
	d := validProjectionOnlyDrafterForModel(m)
	state, err := NewMTPDrafterState(0, []float32{1, 0}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	result, err := m.RunMTPMultiDraftSpeculativeStep(d, state, nil, 0, 2, nil, nil, MTPSpeculationStats{})
	if err != nil {
		t.Fatalf("RunMTPMultiDraftSpeculativeStep: %v", err)
	}
	if !sameInts(result.Drafts.Tokens, []int{1, 0}) {
		t.Fatalf("draft tokens=%v want [1 0]", result.Drafts.Tokens)
	}
	if !result.Verifier.Acceptance.AllDraftsAccepted || result.Verifier.Acceptance.AcceptedPrefixLen != 2 {
		t.Fatalf("acceptance=%+v, want all accepted two-token draft", result.Verifier.Acceptance)
	}
	if !sameInts(result.Verifier.Acceptance.OutputTokens, []int{1, 0, 1}) {
		t.Fatalf("OutputTokens=%v want [1 0 1]", result.Verifier.Acceptance.OutputTokens)
	}
	if result.Stats.Steps != 1 || result.Stats.DraftedTokens != 2 || result.Stats.VerifiedTokens != 2 || result.Stats.BonusTokens != 1 || result.Stats.OutputTokens != 3 {
		t.Fatalf("stats=%+v", result.Stats)
	}
}

func TestRunMTPMultiDraftSpeculativeStepPreflightsStatsBeforeKV(t *testing.T) {
	m := newSingleLayerVerifierModel()
	d := validProjectionOnlyDrafterForModel(m)
	state, err := NewMTPDrafterState(0, []float32{1, 0}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	stats := MTPSpeculationStats{DraftedTokens: int(^uint(0)>>1) - 1}
	if _, err := m.RunMTPMultiDraftSpeculativeStep(d, state, nil, 0, 2, kvCacheK, kvCacheV, stats); err == nil {
		t.Fatal("accepted preflight stats overflow")
	}
	if len(kvCacheK[0]) != 0 || len(kvCacheV[0]) != 0 {
		t.Fatalf("preflight stats failure mutated verifier KV K/V=%d/%d", len(kvCacheK[0]), len(kvCacheV[0]))
	}
}

func TestRunMTPMultiDraftSpeculativeStepValidation(t *testing.T) {
	m := newSingleLayerVerifierModel()
	d := validProjectionOnlyDrafterForModel(m)
	state, err := NewMTPDrafterState(0, []float32{1, 0}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	if _, err := m.RunMTPMultiDraftSpeculativeStep(d, state, nil, 0, 0, nil, nil, MTPSpeculationStats{}); err == nil {
		t.Fatal("accepted zero draft count")
	}
	if _, err := m.RunMTPMultiDraftSpeculativeStep(d, state, nil, 0, -1, nil, nil, MTPSpeculationStats{}); err == nil {
		t.Fatal("accepted negative draft count")
	}
	if _, err := m.RunMTPMultiDraftSpeculativeStep(d, state, nil, 0, maxMTPDraftCount+1, nil, nil, MTPSpeculationStats{}); err == nil {
		t.Fatal("accepted oversized draft count")
	}
}

func TestRunMTPSpeculativeStepRestoresKVOnPostVerifierStatsFailure(t *testing.T) {
	m := newSingleLayerVerifierModel()
	d := validProjectionOnlyDrafterForModel(m)
	d.PostProjection = []float32{0, 10, 10, 0}
	state, err := NewMTPDrafterState(0, []float32{0, 0}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	stats := MTPSpeculationStats{VerifiedTokens: int(^uint(0) >> 1)}
	if _, err := m.RunMTPSpeculativeStep(d, state, nil, 0, kvCacheK, kvCacheV, stats); err == nil {
		t.Fatal("accepted post-verifier stats overflow")
	}
	if len(kvCacheK[0]) != 0 || len(kvCacheV[0]) != 0 {
		t.Fatalf("post-verifier stats failure left staged KV K/V=%d/%d", len(kvCacheK[0]), len(kvCacheV[0]))
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
	if len(kvCacheK[0]) != 0 || len(kvCacheV[0]) != 0 {
		t.Fatalf("overflowing stats mutated verifier KV K/V=%d/%d", len(kvCacheK[0]), len(kvCacheV[0]))
	}
}

func validProjectionOnlyDrafterForModel(m *LlamaModel) *Gemma4MTPDrafter {
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	return d
}
