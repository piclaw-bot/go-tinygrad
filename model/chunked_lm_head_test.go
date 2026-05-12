package model

import "testing"

func TestChunkedGPULMHeadRejectsMalformedInputs(t *testing.T) {
	if (&GPUModel{}).chunkedGPULMHead(nil, nil, 1, 1) {
		t.Fatal("accepted missing logits/hidden")
	}
	g := &GPUModel{lmHead: []float32{1, 2, 3, 4}}
	if g.chunkedGPULMHead(make([]float32, 1), []float32{1, 2}, 2, 2) {
		t.Fatal("accepted short logits")
	}
	if g.chunkedGPULMHead(make([]float32, 2), []float32{1}, 2, 2) {
		t.Fatal("accepted short hidden")
	}
	if g.chunkedGPULMHead(make([]float32, 2), []float32{1, 2}, 2, 0) {
		t.Fatal("accepted zero hidden size")
	}
	if (&GPUModel{lmHead: []float32{1}}).chunkedGPULMHead(make([]float32, 2), []float32{1, 2}, 2, 2) {
		t.Fatal("accepted short LM head backing data")
	}
}
