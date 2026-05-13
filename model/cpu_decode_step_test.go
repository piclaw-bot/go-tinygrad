package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
)

func TestFinishCPUDecodeStep(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 3, HiddenSize: 2, RMSNormEps: 0},
		Norm:   tensor.Ones([]int{2}),
		LMHead: tensor.FromFloat32([]float32{
			1, 0,
			0, 1,
			1, 1,
		}, []int{3, 2}),
	}
	hidden := []float32{3, 4}
	activation, logits, tok, err := m.finishCPUDecodeStep(hidden)
	if err != nil {
		t.Fatalf("finishCPUDecodeStep: %v", err)
	}
	if tok != 2 {
		t.Fatalf("token=%d want 2 logits=%v", tok, logits)
	}
	if len(activation) != 2 || len(logits) != 3 {
		t.Fatalf("activation/logits lengths=%d/%d", len(activation), len(logits))
	}
	if !sameFloat32s(activation, hidden) {
		t.Fatalf("activation=%v want mutated hidden=%v", activation, hidden)
	}
	hidden[0] = 99
	if activation[0] == 99 {
		t.Fatal("final activation aliases hidden scratch")
	}
}

func TestFinishCPUDecodeStepValidation(t *testing.T) {
	if _, _, _, err := (*LlamaModel)(nil).finishCPUDecodeStep(nil); err == nil {
		t.Fatal("accepted nil model")
	}
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 1, HiddenSize: 2}}
	if _, _, _, err := m.finishCPUDecodeStep([]float32{1, 2}); err == nil {
		t.Fatal("accepted missing final norm")
	}
	m.Norm = tensor.Ones([]int{2})
	if _, _, _, err := m.finishCPUDecodeStep([]float32{1}); err == nil {
		t.Fatal("accepted short hidden")
	}
	m.Norm = tensor.FromFloat32([]float32{1}, []int{1})
	hidden := []float32{1, 2}
	if _, _, _, err := m.finishCPUDecodeStep(hidden); err == nil {
		t.Fatal("accepted short final norm")
	}
	if !sameFloat32s(hidden, []float32{1, 2}) {
		t.Fatalf("short final norm mutated hidden=%v", hidden)
	}
	m.Norm = tensor.Ones([]int{2})
	m.LMHead = tensor.FromFloat32([]float32{1}, []int{1})
	if _, _, _, err := m.finishCPUDecodeStep([]float32{1, 2}); err == nil {
		t.Fatal("accepted malformed LM head")
	}
}
