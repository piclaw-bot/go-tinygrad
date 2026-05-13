package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestMTPVerifierTokens(t *testing.T) {
	tokens, err := MTPVerifierTokens(7, []int{10, 11})
	if err != nil {
		t.Fatalf("MTPVerifierTokens: %v", err)
	}
	if !sameInts(tokens, []int{7, 10, 11}) {
		t.Fatalf("tokens=%v want [7 10 11]", tokens)
	}
	if _, err := MTPVerifierTokens(-1, nil); err == nil {
		t.Fatal("MTPVerifierTokens accepted negative input token")
	}
	if _, err := MTPVerifierTokens(1, []int{2, -3}); err == nil {
		t.Fatal("MTPVerifierTokens accepted negative drafted token")
	}
}

func TestNewMTPVerifierResult(t *testing.T) {
	logits := [][]float32{
		{0, 9, 1, 0},
		{0, 1, 8, 0},
		{0, 0, 1, 7},
	}
	got, err := NewMTPVerifierResult(99, []int{1, 2}, logits, []float32{3, 4})
	if err != nil {
		t.Fatalf("NewMTPVerifierResult: %v", err)
	}
	if !sameInts(got.VerifierTokens, []int{99, 1, 2}) {
		t.Fatalf("VerifierTokens=%v want [99 1 2]", got.VerifierTokens)
	}
	if !sameInts(got.Acceptance.OutputTokens, []int{1, 2, 3}) {
		t.Fatalf("OutputTokens=%v want [1 2 3]", got.Acceptance.OutputTokens)
	}
	if !sameFloat32s(got.FinalActivation, []float32{3, 4}) {
		t.Fatalf("FinalActivation=%v want [3 4]", got.FinalActivation)
	}

	// Result owns copies of mutable inputs.
	logits[0][1] = -1
	if got.Logits[0][1] != 9 {
		t.Fatalf("result logits were aliased: %v", got.Logits[0])
	}
}

func TestNewMTPVerifierResultValidation(t *testing.T) {
	if _, err := NewMTPVerifierResult(1, []int{2}, [][]float32{{0, 1}}, nil); err == nil {
		t.Fatal("NewMTPVerifierResult accepted missing verifier bonus row")
	}
	if _, err := NewMTPVerifierResult(1, nil, [][]float32{nil}, nil); err == nil {
		t.Fatal("NewMTPVerifierResult accepted empty logits row")
	}
}

func TestNewMTPVerifierResultForModelValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 4, HiddenSize: 2}}
	if _, err := NewMTPVerifierResultForModel(nil, 1, nil, [][]float32{{0, 1, 2, 3}}, []float32{1, 2}); err == nil {
		t.Fatal("NewMTPVerifierResultForModel accepted nil model")
	}
	if _, err := NewMTPVerifierResultForModel(&LlamaModel{}, 1, nil, [][]float32{{0}}, nil); err == nil {
		t.Fatal("NewMTPVerifierResultForModel accepted invalid model dims")
	}
	if _, err := NewMTPVerifierResultForModel(m, 4, nil, [][]float32{{0, 1, 2, 3}}, []float32{1, 2}); err == nil {
		t.Fatal("NewMTPVerifierResultForModel accepted input token outside vocab")
	}
	if _, err := NewMTPVerifierResultForModel(m, 1, []int{4}, [][]float32{{0, 1, 2, 3}, {0, 1, 2, 3}}, []float32{1, 2}); err == nil {
		t.Fatal("NewMTPVerifierResultForModel accepted drafted token outside vocab")
	}
	if _, err := NewMTPVerifierResultForModel(m, 1, nil, [][]float32{{0, 1, 2}}, []float32{1, 2}); err == nil {
		t.Fatal("NewMTPVerifierResultForModel accepted short logits row")
	}
	if _, err := NewMTPVerifierResultForModel(m, 1, nil, [][]float32{{0, 1, 2, 3}}, []float32{1}); err == nil {
		t.Fatal("NewMTPVerifierResultForModel accepted wrong final activation width")
	}
	got, err := NewMTPVerifierResultForModel(m, 1, []int{2}, [][]float32{{0, 0, 9, 0}, {0, 0, 0, 8}}, []float32{3, 4})
	if err != nil {
		t.Fatalf("NewMTPVerifierResultForModel: %v", err)
	}
	if !sameInts(got.Acceptance.OutputTokens, []int{2, 3}) {
		t.Fatalf("OutputTokens=%v want [2 3]", got.Acceptance.OutputTokens)
	}
}

func TestMTPVerifierResultCommitFloatKV(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	result, err := NewMTPVerifierResult(9, []int{1, 2}, [][]float32{{0, 9, 0}, {0, 8, 0}, {0, 0, 7}}, nil)
	if err != nil {
		t.Fatalf("NewMTPVerifierResult: %v", err)
	}
	k := [][]float32{{1, 2, 10, 11, 12, 13, 14, 15}}
	v := [][]float32{{3, 4, 20, 21, 22, 23, 24, 25}}
	cp := kv.FloatKVCheckpoint{KLen: []int{2}, VLen: []int{2}}
	if err := result.CommitFloatKV(m, k, v, cp); err != nil {
		t.Fatalf("CommitFloatKV: %v", err)
	}
	if want := []float32{1, 2, 10, 11, 12, 13}; !sameFloat32s(k[0], want) {
		t.Fatalf("K=%v want %v", k[0], want)
	}
}

func TestMTPVerifierResultCommitCompressedKV(t *testing.T) {
	cache := kv.NewCompressedKVCache(2, 1, 2, nil, true)
	cache.Append([]float32{1, 2}, []float32{10, 20})
	cp := kv.CheckpointCompressedKV([]*kv.CompressedKVCache{cache})
	cache.Append([]float32{3, 4}, []float32{30, 40})
	cache.Append([]float32{5, 6}, []float32{50, 60})
	cache.Append([]float32{7, 8}, []float32{70, 80})
	result, err := NewMTPVerifierResult(9, []int{1, 2}, [][]float32{{0, 9, 0}, {0, 8, 0}, {0, 0, 7}}, nil)
	if err != nil {
		t.Fatalf("NewMTPVerifierResult: %v", err)
	}
	if err := result.CommitCompressedKV([]*kv.CompressedKVCache{cache}, cp); err != nil {
		t.Fatalf("CommitCompressedKV: %v", err)
	}
	if got, want := cache.SeqLen(), 3; got != want {
		t.Fatalf("seq len=%d want %d", got, want)
	}
}
