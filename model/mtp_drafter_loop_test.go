package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func TestNewMTPDrafterState(t *testing.T) {
	state, err := NewMTPDrafterState(7, []float32{1, 2, 3}, 3)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	if state.PreviousToken != 7 || !sameFloat32s(state.Activation, []float32{1, 2, 3}) {
		t.Fatalf("state=%+v", state)
	}
	state.Activation[0] = 99
	orig := []float32{4, 5}
	state, err = NewMTPDrafterState(1, orig, 2)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	orig[0] = 99
	if state.Activation[0] == 99 {
		t.Fatal("state activation aliases caller slice")
	}
}

func TestNewMTPDrafterStateValidation(t *testing.T) {
	if _, err := NewMTPDrafterState(-1, nil, 1); err == nil {
		t.Fatal("accepted negative previous token")
	}
	if _, err := NewMTPDrafterState(1, nil, 0); err == nil {
		t.Fatal("accepted invalid backbone width")
	}
	if _, err := NewMTPDrafterState(1, []float32{1}, 2); err == nil {
		t.Fatal("accepted wrong activation width")
	}
}

func TestRunMTPDrafterStepProjectionOnly(t *testing.T) {
	m := validDrafterStepBackboneModel()
	d := validProjectionOnlyDrafter()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	got, err := m.RunMTPDrafterStep(d, state)
	if err != nil {
		t.Fatalf("RunMTPDrafterStep: %v", err)
	}
	if got.Token != 1 {
		t.Fatalf("Token=%d want 1 logits=%v", got.Token, got.Logits)
	}
	if !sameFloat32s(got.NextActivation, []float32{0, 0.5}) {
		t.Fatalf("NextActivation=%v want [0 0.5]", got.NextActivation)
	}
	got.NextActivation[0] = 99
	if got.NextState.Activation[0] == 99 {
		t.Fatal("NextState activation aliases result activation")
	}
}

func TestRunMTPDrafterStepContractValidation(t *testing.T) {
	m := validDrafterStepBackboneModel()
	if _, err := (*LlamaModel)(nil).RunMTPDrafterStep(validProjectionOnlyDrafter(), MTPDrafterState{}); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := m.RunMTPDrafterStep(nil, MTPDrafterState{}); err == nil {
		t.Fatal("accepted nil drafter")
	}
	d := validDrafterStepScaffold()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	_, err = m.RunMTPDrafterStep(d, state)
	if err == nil || !strings.Contains(err.Error(), "external KV is required") {
		t.Fatalf("RunMTPDrafterStep err=%v, want missing external KV", err)
	}
	externalKV := &MTPDrafterExternalKV{K: [][]float32{{1, 0}}, V: [][]float32{{0, 1}}, SourceLayers: []int{0}, SeqLen: 1}
	got, err := m.RunMTPDrafterStepWithExternalKV(d, state, externalKV)
	if err != nil {
		t.Fatalf("RunMTPDrafterStepWithExternalKV: %v", err)
	}
	if len(got.Logits) != m.Config.VocabSize || len(got.NextActivation) != d.BackboneHiddenSize {
		t.Fatalf("result logits/activation len=%d/%d", len(got.Logits), len(got.NextActivation))
	}

	bad := *d
	bad.Config.VocabSize = 0
	if _, err := m.RunMTPDrafterStep(&bad, state); err == nil {
		t.Fatal("accepted invalid drafter dims")
	}
	bad = *d
	bad.BackboneHiddenSize = 3
	if _, err := m.RunMTPDrafterStep(&bad, MTPDrafterState{PreviousToken: 1, Activation: []float32{1, 2, 3}}); err == nil {
		t.Fatal("accepted model/drafter dimension mismatch")
	}
	if _, err := m.RunMTPDrafterStep(d, MTPDrafterState{PreviousToken: 99, Activation: []float32{1, 2}}); err == nil {
		t.Fatal("accepted previous token outside vocab")
	}
	if _, err := m.RunMTPDrafterStep(d, MTPDrafterState{PreviousToken: 1, Activation: []float32{1}}); err == nil {
		t.Fatal("accepted wrong state activation width")
	}
	bad = *d
	bad.PreProjection = nil
	if _, err := m.RunMTPDrafterStep(&bad, state); err == nil {
		t.Fatal("accepted missing projection weights")
	}
}

func TestMTPDrafterGQAAttentionUsesGemmaScale(t *testing.T) {
	d := validDrafterStepScaffold()
	d.Config.ModelType = "gemma4_text"
	q := []float32{1, 0}
	k := []float32{0, 1, 2, 0}
	v := []float32{1, 10, 100, 1000}
	got := drafterGQAAttention(d, q, k, v, 2, 1, 1, 2)
	want := gqaAttentionScale(q, k, v, 2, 1, 1, 2, 1.0)
	if !sameFloat32s(got, want) {
		t.Fatalf("drafter attention=%v want Gemma scale %v", got, want)
	}
	defaultScaled := gqaAttention(q, k, v, 2, 1, 1, 2)
	if sameFloat32s(got, defaultScaled) {
		t.Fatalf("Gemma drafter attention unexpectedly matched default scaled path: %v", got)
	}
}

func TestMTPDrafterRMSNormUsesGemmaBF16Path(t *testing.T) {
	d := validDrafterStepScaffold()
	d.Config.ModelType = "gemma4_text"
	x := []float32{1.0001, 2.0001}
	want := append([]float32(nil), x...)
	rmsNormBF16(want, []float32{1, 1}, float32(d.Config.RMSNormEps))
	drafterRMSNormInPlace(d, x, []float32{1, 1})
	if !sameFloat32s(x, want) {
		t.Fatalf("drafter norm=%v want BF16 path %v", x, want)
	}
}

func TestRunMTPDrafterStepAppliesFinalNormBeforePostProjection(t *testing.T) {
	m := validDrafterStepBackboneModel()
	d := validDrafterStepScaffold()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	externalKV := &MTPDrafterExternalKV{K: [][]float32{{1, 0}}, V: [][]float32{{0, 1}}, SourceLayers: []int{0}, SeqLen: 1}
	got, err := m.RunMTPDrafterStepWithExternalKV(d, state, externalKV)
	if err != nil {
		t.Fatalf("RunMTPDrafterStepWithExternalKV: %v", err)
	}
	d.Norm = tensor.FromFloat32([]float32{1, 2}, []int{2})
	withNorm, err := m.RunMTPDrafterStepWithExternalKV(d, state, externalKV)
	if err != nil {
		t.Fatalf("RunMTPDrafterStepWithExternalKV with norm: %v", err)
	}
	if sameFloat32s(got.NextActivation, withNorm.NextActivation) {
		t.Fatalf("final norm did not affect next activation: %v", got.NextActivation)
	}
}

func TestNewMTPDrafterExternalKVDefaultMapping(t *testing.T) {
	d := validDrafterStepScaffold()
	externalKV, err := NewMTPDrafterExternalKV(d, [][]float32{{1, 0}}, [][]float32{{0, 1}}, 1)
	if err != nil {
		t.Fatalf("NewMTPDrafterExternalKV: %v", err)
	}
	if !sameInts(externalKV.SourceLayers, []int{0}) {
		t.Fatalf("SourceLayers=%v want [0]", externalKV.SourceLayers)
	}
	if _, err := NewMTPDrafterExternalKV(nil, nil, nil, 0); err == nil {
		t.Fatal("accepted nil drafter")
	}
}

func TestRunMTPDrafterStepRealAssetContract(t *testing.T) {
	drafterDir := filepath.Join("..", "models", "gemma4-e2b-mtp-drafter")
	mainDir := filepath.Join("..", "models", "gemma4-e2b-mlx4")
	if _, errSingle := os.Stat(filepath.Join(drafterDir, "model.safetensors")); errSingle != nil {
		if _, errSharded := os.Stat(filepath.Join(drafterDir, "model.safetensors.index.json")); errSharded != nil {
			t.Skipf("local Gemma4 MTP drafter asset not available: single=%v sharded=%v", errSingle, errSharded)
		}
	}
	if _, errSingle := os.Stat(filepath.Join(mainDir, "model.safetensors")); errSingle != nil {
		if _, errSharded := os.Stat(filepath.Join(mainDir, "model.safetensors.index.json")); errSharded != nil {
			t.Skipf("local Gemma4 main asset not available: single=%v sharded=%v", errSingle, errSharded)
		}
	}
	d, err := LoadGemma4MTPDrafter(drafterDir)
	if err != nil {
		t.Fatalf("LoadGemma4MTPDrafter: %v", err)
	}
	m, err := LoadLlama(mainDir)
	if err != nil {
		t.Fatalf("LoadLlama: %v", err)
	}
	if m.Config.HiddenSize != d.BackboneHiddenSize || m.Config.VocabSize != d.Config.VocabSize {
		t.Fatalf("model/drafter mismatch h/vocab=%d/%d backbone/vocab=%d/%d", m.Config.HiddenSize, m.Config.VocabSize, d.BackboneHiddenSize, d.Config.VocabSize)
	}
	k := make([][]float32, d.Config.NumLayers)
	v := make([][]float32, d.Config.NumLayers)
	for i := range d.Layers {
		headDim := d.Config.HeadDim
		if d.Layers[i].HeadDimLocal > 0 {
			headDim = d.Layers[i].HeadDimLocal
		}
		kvDim := d.Config.NumKVHeads * headDim
		k[i] = make([]float32, kvDim)
		v[i] = make([]float32, kvDim)
	}
	externalKV, err := NewMTPDrafterExternalKV(d, k, v, 1)
	if err != nil {
		t.Fatalf("NewMTPDrafterExternalKV: %v", err)
	}
	state, err := NewMTPDrafterState(0, make([]float32, d.BackboneHiddenSize), d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	result, err := m.RunMTPDrafterStepWithExternalKV(d, state, externalKV)
	if err != nil {
		t.Fatalf("RunMTPDrafterStepWithExternalKV: %v", err)
	}
	if result.Token < 0 || result.Token >= m.Config.VocabSize {
		t.Fatalf("draft token=%d out of range [0,%d)", result.Token, m.Config.VocabSize)
	}
	if len(result.Logits) != m.Config.VocabSize || len(result.NextActivation) != d.BackboneHiddenSize || len(result.NextState.Activation) != d.BackboneHiddenSize {
		t.Fatalf("result shapes logits/activation/state=%d/%d/%d", len(result.Logits), len(result.NextActivation), len(result.NextState.Activation))
	}
}

func TestRunMTPDrafterStepExternalKVValidation(t *testing.T) {
	m := validDrafterStepBackboneModel()
	d := validDrafterStepScaffold()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	validKV := &MTPDrafterExternalKV{K: [][]float32{{1, 0}}, V: [][]float32{{0, 1}}, SourceLayers: []int{0}, SeqLen: 1}
	badKV := *validKV
	badKV.SeqLen = 0
	if _, err := m.RunMTPDrafterStepWithExternalKV(d, state, &badKV); err == nil {
		t.Fatal("accepted invalid external KV seq len")
	}
	badKV = *validKV
	badKV.SourceLayers = nil
	if _, err := m.RunMTPDrafterStepWithExternalKV(d, state, &badKV); err == nil {
		t.Fatal("accepted missing external KV source mapping")
	}
	badKV = *validKV
	badKV.SourceLayers = []int{1}
	if _, err := m.RunMTPDrafterStepWithExternalKV(d, state, &badKV); err == nil {
		t.Fatal("accepted out-of-range external KV source")
	}
	badKV = *validKV
	badKV.K = [][]float32{{1}}
	if _, err := m.RunMTPDrafterStepWithExternalKV(d, state, &badKV); err == nil {
		t.Fatal("accepted wrong external KV width")
	}
	bad := *d
	bad.Layers = append([]Gemma4MTPDrafterLayer(nil), d.Layers...)
	bad.Layers[0].KVSourceLayer = 0
	if _, err := m.RunMTPDrafterStepWithExternalKV(&bad, state, validKV); err == nil {
		t.Fatal("accepted non-q-only drafter KV source")
	}
	bad = *d
	bad.Norm = nil
	if _, err := m.RunMTPDrafterStepWithExternalKV(&bad, state, validKV); err == nil {
		t.Fatal("accepted missing drafter final norm")
	}
	bad = *d
	bad.Layers = append([]Gemma4MTPDrafterLayer(nil), d.Layers...)
	bad.Layers[0].QW = bad.Layers[0].QW[:1]
	if _, err := m.RunMTPDrafterStepWithExternalKV(&bad, state, validKV); err == nil {
		t.Fatal("accepted invalid q-only projection dims")
	}
}

func validDrafterStepBackboneModel() *LlamaModel {
	return &LlamaModel{
		Config: LlamaConfig{VocabSize: 4, HiddenSize: 2},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 0,
			0, 1,
			1, 1,
			-1, 0,
		}, []int{4, 2}),
		LMHead: tensor.FromFloat32([]float32{
			1, 0,
			0, 1,
			1, 1,
			-1, 0,
		}, []int{4, 2}),
	}
}

func validProjectionOnlyDrafter() *Gemma4MTPDrafter {
	return &Gemma4MTPDrafter{
		Config:             LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 0},
		BackboneHiddenSize: 2,
		PreProjection: []float32{
			1, 0, 0, 0,
			0, 0, 1, 0,
		},
		PostProjection: []float32{
			1, 0,
			0, 1,
		},
	}
}

func validDrafterStepScaffold() *Gemma4MTPDrafter {
	d := validProjectionOnlyDrafter()
	d.Config.NumLayers = 1
	d.Config.NumHeads = 1
	d.Config.NumKVHeads = 1
	d.Config.HeadDim = 2
	d.Config.Intermediate = 2
	d.Norm = tensor.Ones([]int{2})
	d.Layers = []Gemma4MTPDrafterLayer{{
		InputNorm:     tensor.Ones([]int{2}),
		PostNorm:      tensor.Ones([]int{2}),
		PreFFNNorm:    tensor.Ones([]int{2}),
		PostFFNNorm:   tensor.Ones([]int{2}),
		QNorm:         tensor.Ones([]int{2}),
		LayerScalar:   1,
		KVSourceLayer: -1,
		QW:            []float32{1, 0, 0, 1},
		OW:            []float32{1, 0, 0, 1},
		GateW:         []float32{1, 0, 0, 1},
		UpW:           []float32{1, 0, 0, 1},
		DownW:         []float32{1, 0, 0, 1},
	}}
	return d
}
