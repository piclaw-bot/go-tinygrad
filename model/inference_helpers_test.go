package model

import (
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
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
