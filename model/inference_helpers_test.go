package model

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func TestTokenEmbeddingHelpers(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 3, HiddenSize: 4, ModelType: "gemma4_text"},
		EmbedTokens: tensor.FromFloat32([]float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
		}, []int{3, 4}),
	}

	raw := make([]float32, 4)
	if err := m.TokenEmbeddingInto(raw, 1); err != nil {
		t.Fatalf("TokenEmbeddingInto: %v", err)
	}
	if want := []float32{5, 6, 7, 8}; !sameFloat32s(raw, want) {
		t.Fatalf("raw embedding = %v, want %v", raw, want)
	}

	scaled := make([]float32, 4)
	if err := m.ScaledTokenEmbeddingInto(scaled, 1); err != nil {
		t.Fatalf("ScaledTokenEmbeddingInto: %v", err)
	}
	if want := []float32{10, 12, 14, 16}; !sameFloat32s(scaled, want) {
		t.Fatalf("scaled embedding = %v, want %v", scaled, want)
	}

	if err := m.TokenEmbeddingInto(make([]float32, 3), 1); err == nil {
		t.Fatal("TokenEmbeddingInto accepted short destination")
	}
	if err := m.TokenEmbeddingInto(make([]float32, 4), 3); err == nil {
		t.Fatal("TokenEmbeddingInto accepted out-of-range token")
	}
}

func TestGemma4PerLayerInputs(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{
			HiddenSize:     2,
			NumLayers:      2,
			HiddenPerLayer: 2,
			VocabPerLayer:  2,
			RMSNormEps:     0,
		},
		PerLayerModelProj: []float32{
			1, 0,
			1, 0,
			2, 0,
			2, 0,
		},
		PerLayerProjNorm:   []float32{1, 1},
		PerLayerProjScale:  1,
		PerLayerInputScale: 0.5,
		EmbedPerLayerScale: 1,
		EmbedPerLayer: []float32{
			0, 0, 0, 0,
			10, 20, 30, 40,
		},
	}
	inputs, err := m.Gemma4PerLayerInputs([]float32{1, 99}, 1)
	if err != nil {
		t.Fatalf("Gemma4PerLayerInputs: %v", err)
	}
	if len(inputs) != 2 {
		t.Fatalf("len(inputs)=%d want 2", len(inputs))
	}
	if want := []float32{5.5, 10.5}; !sameFloat32s(inputs[0], want) {
		t.Fatalf("inputs[0]=%v want %v", inputs[0], want)
	}
	if want := []float32{15.5, 20.5}; !sameFloat32s(inputs[1], want) {
		t.Fatalf("inputs[1]=%v want %v", inputs[1], want)
	}

	m.PerLayerModelProj = nil
	inputs, err = m.Gemma4PerLayerInputs([]float32{1, 2}, 1)
	if err != nil || inputs != nil {
		t.Fatalf("disabled per-layer inputs = %v, %v; want nil, nil", inputs, err)
	}
}

func TestGemma4PerLayerInputsValidation(t *testing.T) {
	m := &LlamaModel{
		Config:             LlamaConfig{HiddenSize: 2, NumLayers: 1, HiddenPerLayer: 2},
		PerLayerModelProj:  []float32{1, 2, 3},
		PerLayerProjNorm:   []float32{1, 1},
		PerLayerProjScale:  1,
		PerLayerInputScale: 1,
		EmbedPerLayerScale: 1,
	}
	if _, err := m.Gemma4PerLayerInputs([]float32{1}, 0); err == nil {
		t.Fatal("Gemma4PerLayerInputs accepted short hidden")
	}
	if _, err := m.Gemma4PerLayerInputs([]float32{1, 2}, 0); err == nil {
		t.Fatal("Gemma4PerLayerInputs accepted short projection")
	}
}

func TestLMHeadLogitsAndArgmax(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 3, HiddenSize: 2},
		LMHead: tensor.FromFloat32([]float32{
			1, 0,
			0, 1,
			1, 1,
		}, []int{3, 2}),
	}
	logits := make([]float32, 3)
	if err := m.LMHeadLogitsInto(logits, []float32{2, 3}); err != nil {
		t.Fatalf("LMHeadLogitsInto: %v", err)
	}
	if want := []float32{2, 3, 5}; !sameFloat32s(logits, want) {
		t.Fatalf("logits = %v, want %v", logits, want)
	}
	idx, val, err := ArgmaxLogits(logits)
	if err != nil {
		t.Fatalf("ArgmaxLogits: %v", err)
	}
	if idx != 2 || val != 5 {
		t.Fatalf("ArgmaxLogits = %d, %v; want 2, 5", idx, val)
	}
	if _, _, err := ArgmaxLogits(nil); err == nil {
		t.Fatal("ArgmaxLogits accepted empty logits")
	}
	if err := m.LMHeadLogitsInto(make([]float32, 2), []float32{2, 3}); err == nil {
		t.Fatal("LMHeadLogitsInto accepted short logits")
	}
}

func sameFloat32s(a, b []float32) bool {
	return closeFloat32s(a, b, 0)
}

func closeFloat32s(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if float32(math.Abs(float64(a[i]-b[i]))) > tol {
			return false
		}
	}
	return true
}
