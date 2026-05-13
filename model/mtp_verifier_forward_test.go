package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/kv"
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

func TestRunMTPVerifierForwardOneLayerDeterministicAcceptance(t *testing.T) {
	m := newSingleLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{0}, 0)
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	result, err := m.RunMTPVerifierForward(plan, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatalf("RunMTPVerifierForward: %v", err)
	}
	if !result.Acceptance.AllDraftsAccepted || result.Acceptance.AcceptedPrefixLen != 1 {
		t.Fatalf("acceptance=%+v, want deterministic all-accepted one-token draft", result.Acceptance)
	}
	if !sameInts(result.Acceptance.OutputTokens, []int{0, 0}) {
		t.Fatalf("OutputTokens=%v want [0 0]", result.Acceptance.OutputTokens)
	}
	kvDim, err := m.LayerKVDim(0)
	if err != nil {
		t.Fatalf("LayerKVDim: %v", err)
	}
	if got, want := len(kvCacheK[0]), len(plan.VerifierTokens)*kvDim; got != want {
		t.Fatalf("staged K len=%d want %d", got, want)
	}
	if len(result.FinalActivation) != m.Config.HiddenSize {
		t.Fatalf("FinalActivation len=%d want %d", len(result.FinalActivation), m.Config.HiddenSize)
	}
}

func TestRunMTPVerifierForwardFloatKVCommitKeepsAcceptedPrefix(t *testing.T) {
	m := newSingleLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{2}, 0)
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	cp := kv.CheckpointFloatKV(kvCacheK, kvCacheV)
	result, err := m.RunMTPVerifierForward(plan, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatalf("RunMTPVerifierForward: %v", err)
	}
	kvDim, err := m.LayerKVDim(0)
	if err != nil {
		t.Fatalf("LayerKVDim: %v", err)
	}
	if got, want := len(kvCacheK[0]), len(plan.VerifierTokens)*kvDim; got != want {
		t.Fatalf("staged K len=%d want %d", got, want)
	}
	keep := result.Acceptance.KVKeepTokens()
	if err := result.CommitFloatKV(m, kvCacheK, kvCacheV, cp); err != nil {
		t.Fatalf("CommitFloatKV: %v", err)
	}
	if got, want := len(kvCacheK[0]), keep*kvDim; got != want {
		t.Fatalf("committed K len=%d want %d acceptance=%+v", got, want, result.Acceptance)
	}
	if got, want := len(kvCacheV[0]), keep*kvDim; got != want {
		t.Fatalf("committed V len=%d want %d acceptance=%+v", got, want, result.Acceptance)
	}
}

func TestRunMTPVerifierForwardRequiresPromptHistoryKV(t *testing.T) {
	m := newSingleLayerVerifierModel()
	kvDim, err := m.LayerKVDim(0)
	if err != nil {
		t.Fatalf("LayerKVDim: %v", err)
	}
	plan := mustMTPVerifierPlan(t, m, 0, []int{2}, 1)
	if _, err := m.RunMTPVerifierForward(plan, make([][]float32, len(m.Layers)), make([][]float32, len(m.Layers))); err == nil {
		t.Fatal("accepted missing prompt/history KV for non-zero start position")
	}
	kvCacheK := [][]float32{make([]float32, kvDim)}
	kvCacheV := [][]float32{make([]float32, kvDim)}
	result, err := m.RunMTPVerifierForward(plan, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatalf("RunMTPVerifierForward with history: %v", err)
	}
	if got, want := len(kvCacheK[0]), (plan.StartPos+len(plan.VerifierTokens))*kvDim; got != want {
		t.Fatalf("staged history K len=%d want %d", got, want)
	}
	if result.InputToken != 0 {
		t.Fatalf("result input token=%d want 0", result.InputToken)
	}
}

func TestRunMTPVerifierForwardRejectsGemma4PLIUntilSupported(t *testing.T) {
	m := newZeroLayerVerifierModel()
	m.Config.HiddenPerLayer = 2
	plan := mustMTPVerifierPlan(t, m, 1, nil, 0)
	if _, err := m.RunMTPVerifierForward(plan, nil, nil); err == nil {
		t.Fatal("accepted Gemma4 per-layer input gating")
	}
}

func TestRunMTPVerifierForwardRejectsMalformedSharedKVLayers(t *testing.T) {
	m := newSingleLayerVerifierModel()
	m.Config.NumLayers = 2
	m.Layers = append(m.Layers, LlamaLayer{HasKV: false, KVSourceLayer: -1})
	plan := mustMTPVerifierPlan(t, m, 0, nil, 0)
	if _, err := m.RunMTPVerifierForward(plan, make([][]float32, len(m.Layers)), make([][]float32, len(m.Layers))); err == nil {
		t.Fatal("accepted shared-KV layer with invalid source")
	}
	m.Layers[1].KVSourceLayer = 1
	if _, err := m.RunMTPVerifierForward(plan, make([][]float32, len(m.Layers)), make([][]float32, len(m.Layers))); err == nil {
		t.Fatal("accepted shared-KV layer whose source is also shared")
	}
	m.Layers[1].KVSourceLayer = 0
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	kvCacheK[1] = []float32{1}
	if _, err := m.RunMTPVerifierForward(plan, kvCacheK, kvCacheV); err == nil {
		t.Fatal("accepted shared-KV layer with owned K cache entries")
	}
}

func TestRunMTPVerifierForwardCompressedKVCommitKeepsAcceptedPrefix(t *testing.T) {
	m := newSingleLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{2}, 0)
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	result, err := m.RunMTPVerifierForward(plan, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatalf("RunMTPVerifierForward: %v", err)
	}
	cache := kv.NewCompressedKVCache(2, 1, 2, nil, true)
	cp := kv.CheckpointCompressedKV([]*kv.CompressedKVCache{cache})
	for i := 0; i < len(plan.VerifierTokens); i++ {
		base := float32(i*10 + 1)
		cache.Append([]float32{base, base + 1}, []float32{base + 100, base + 101})
	}
	if got, want := cache.SeqLen(), len(plan.VerifierTokens); got != want {
		t.Fatalf("staged compressed seq len=%d want %d", got, want)
	}
	keep := result.Acceptance.KVKeepTokens()
	if err := result.CommitCompressedKV([]*kv.CompressedKVCache{cache}, cp); err != nil {
		t.Fatalf("CommitCompressedKV: %v", err)
	}
	if got, want := cache.SeqLen(), keep; got != want {
		t.Fatalf("committed compressed seq len=%d want %d acceptance=%+v", got, want, result.Acceptance)
	}
	if got, want := len(cache.GetK()), keep*2; got != want {
		t.Fatalf("committed compressed K len=%d want %d", got, want)
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

func newSingleLayerVerifierModel() *LlamaModel {
	m := newZeroLayerVerifierModel()
	m.Config.NumLayers = 1
	m.Config.Intermediate = 2
	identity := []float32{1, 0, 0, 1}
	m.Layers = []LlamaLayer{{
		InputNorm: tensor.Ones([]int{2}),
		PostNorm:  tensor.Ones([]int{2}),
		HasKV:     true,
		QW:        tensor.FromFloat32(append([]float32(nil), identity...), []int{2, 2}),
		KW:        tensor.FromFloat32(append([]float32(nil), identity...), []int{2, 2}),
		VW:        tensor.FromFloat32(append([]float32(nil), identity...), []int{2, 2}),
		OW:        tensor.FromFloat32(append([]float32(nil), identity...), []int{2, 2}),
		GateW:     tensor.FromFloat32(append([]float32(nil), identity...), []int{2, 2}),
		UpW:       tensor.FromFloat32(append([]float32(nil), identity...), []int{2, 2}),
		DownW:     tensor.FromFloat32(append([]float32(nil), identity...), []int{2, 2}),
	}}
	return m
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
