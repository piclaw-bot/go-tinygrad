package model

import (
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

func TestRunMTPDrafterStepContractValidation(t *testing.T) {
	if _, _, err := (*Gemma4MTPDrafter)(nil).RunMTPDrafterStep(MTPDrafterState{}, nil); err == nil {
		t.Fatal("accepted nil drafter")
	}
	d := validDrafterStepScaffold()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatalf("NewMTPDrafterState: %v", err)
	}
	_, _, err = d.RunMTPDrafterStep(state, []float32{1, 0})
	if err == nil || !strings.Contains(err.Error(), "not implemented") {
		t.Fatalf("RunMTPDrafterStep err=%v, want not implemented", err)
	}

	bad := *d
	bad.Config.VocabSize = 0
	if _, _, err := bad.RunMTPDrafterStep(state, []float32{1, 0}); err == nil {
		t.Fatal("accepted invalid drafter dims")
	}
	if _, _, err := d.RunMTPDrafterStep(MTPDrafterState{PreviousToken: 99, Activation: []float32{1, 2}}, []float32{1, 0}); err == nil {
		t.Fatal("accepted previous token outside vocab")
	}
	if _, _, err := d.RunMTPDrafterStep(MTPDrafterState{PreviousToken: 1, Activation: []float32{1}}, []float32{1, 0}); err == nil {
		t.Fatal("accepted wrong state activation width")
	}
	if _, _, err := d.RunMTPDrafterStep(state, []float32{1}); err == nil {
		t.Fatal("accepted wrong backbone embedding width")
	}
	bad = *d
	bad.PreProjection = nil
	if _, _, err := bad.RunMTPDrafterStep(state, []float32{1, 0}); err == nil {
		t.Fatal("accepted missing projection weights")
	}
	bad = *d
	bad.Norm = nil
	if _, _, err := bad.RunMTPDrafterStep(state, []float32{1, 0}); err == nil {
		t.Fatal("accepted missing norm")
	}
	bad = *d
	bad.Layers = nil
	if _, _, err := bad.RunMTPDrafterStep(state, []float32{1, 0}); err == nil {
		t.Fatal("accepted missing layers")
	}
}

func validDrafterStepScaffold() *Gemma4MTPDrafter {
	return &Gemma4MTPDrafter{
		Config:             LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 1},
		BackboneHiddenSize: 2,
		PreProjection:      make([]float32, 2*4),
		PostProjection:     make([]float32, 2*2),
		Norm:               tensor.Ones([]int{2}),
		Layers:             []Gemma4MTPDrafterLayer{{KVSourceLayer: -1}},
	}
}
