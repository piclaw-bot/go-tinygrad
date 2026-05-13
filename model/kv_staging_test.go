package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestLayerKVDimsAndModelCommitAcceptedFloatKV(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{NumKVHeads: 2, HeadDim: 4},
		Layers: []LlamaLayer{
			{HasKV: true},
			{HasKV: true, HeadDimLocal: 8},
			{HasKV: false, KVSourceLayer: 0},
		},
	}
	dims, err := m.LayerKVDims()
	if err != nil {
		t.Fatalf("LayerKVDims: %v", err)
	}
	wantDims := []int{8, 16, 0}
	for i := range wantDims {
		if dims[i] != wantDims[i] {
			t.Fatalf("dims=%v want %v", dims, wantDims)
		}
	}

	k := [][]float32{make([]float32, 8+3*8), make([]float32, 3*16), nil}
	v := [][]float32{make([]float32, 8+3*8), make([]float32, 3*16), nil}
	for i := range k[0] {
		k[0][i] = float32(i)
		v[0][i] = float32(i + 100)
	}
	for i := range k[1] {
		k[1][i] = float32(i + 200)
		v[1][i] = float32(i + 300)
	}
	cp := kv.FloatKVCheckpoint{KLen: []int{8, 0, 0}, VLen: []int{8, 0, 0}}
	acceptance, err := AcceptMTPDraft([]int{10, 11}, []int{10, 12, 13}) // keep two verifier positions
	if err != nil {
		t.Fatalf("AcceptMTPDraft: %v", err)
	}
	if err := m.CommitAcceptedFloatKV(k, v, cp, acceptance); err != nil {
		t.Fatalf("CommitAcceptedFloatKV: %v", err)
	}
	if got, want := len(k[0]), 8+2*8; got != want {
		t.Fatalf("len(k[0])=%d want %d", got, want)
	}
	if got, want := len(k[1]), 2*16; got != want {
		t.Fatalf("len(k[1])=%d want %d", got, want)
	}
	if got, want := len(k[2]), 0; got != want {
		t.Fatalf("len(k[2])=%d want %d", got, want)
	}
}

func TestLayerKVDimValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{NumKVHeads: 0, HeadDim: 4}, Layers: []LlamaLayer{{HasKV: true}}}
	if _, err := m.LayerKVDim(0); err == nil {
		t.Fatal("LayerKVDim accepted num_key_value_heads=0")
	}
	m.Config.NumKVHeads = 1
	if _, err := m.LayerKVDim(1); err == nil {
		t.Fatal("LayerKVDim accepted out-of-range layer")
	}
}

func TestLayerKVDimRejectsOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	m := &LlamaModel{Config: LlamaConfig{NumKVHeads: maxInt/2 + 1, HeadDim: 3}, Layers: []LlamaLayer{{HasKV: true}}}
	if _, err := m.LayerKVDim(0); err == nil {
		t.Fatal("LayerKVDim accepted overflowing KV dimension")
	}
	m = &LlamaModel{Config: LlamaConfig{NumKVHeads: 2, HeadDim: maxInt/2 + 1}, Layers: []LlamaLayer{{HasKV: true}}}
	if _, err := m.LayerKVDim(0); err == nil {
		t.Fatal("LayerKVDim accepted overflowing local head dimension")
	}
}
