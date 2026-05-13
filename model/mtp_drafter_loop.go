package model

import "fmt"

// MTPDrafterState carries the hidden-state-conditioned drafter inputs between
// speculative iterations. PreviousToken is the last token emitted/accepted by
// the verifier path; Activation is the main-model-width verifier activation
// used together with that token's embedding for the next drafter step.
type MTPDrafterState struct {
	PreviousToken int
	Activation    []float32
}

// MTPDrafterStepResult is one drafter iteration: the drafted token plus the
// projected main-model-width activation to carry into the next drafter step.
type MTPDrafterStepResult struct {
	Token          int
	Logits         []float32
	NextActivation []float32
	NextState      MTPDrafterState
}

// NewMTPDrafterState validates and copies the activation carry for one drafter
// loop. The activation width is the main/backbone model hidden size, not the
// assistant hidden size.
func NewMTPDrafterState(previousToken int, activation []float32, backboneHiddenSize int) (MTPDrafterState, error) {
	if previousToken < 0 {
		return MTPDrafterState{}, fmt.Errorf("previous token %d out of range", previousToken)
	}
	if backboneHiddenSize <= 0 {
		return MTPDrafterState{}, fmt.Errorf("invalid backbone hidden size %d", backboneHiddenSize)
	}
	if len(activation) != backboneHiddenSize {
		return MTPDrafterState{}, fmt.Errorf("activation len=%d, want %d", len(activation), backboneHiddenSize)
	}
	return MTPDrafterState{PreviousToken: previousToken, Activation: append([]float32(nil), activation...)}, nil
}

// RunMTPDrafterStep is one hidden-state-conditioned drafter iteration. The
// complete q-only attention loop is still pending; this function implements the
// projection/LM-head shell and supports zero-layer synthetic drafter fixtures for
// runtime validation. Real drafter assets with q-only layers still return an
// explicit not-implemented error until external-KV attention is wired.
func (m *LlamaModel) RunMTPDrafterStep(d *Gemma4MTPDrafter, state MTPDrafterState) (MTPDrafterStepResult, error) {
	if err := m.validateMTPDrafterStepModel(d, state); err != nil {
		return MTPDrafterStepResult{}, err
	}
	backboneEmbedding := make([]float32, d.BackboneHiddenSize)
	if err := m.TokenEmbeddingInto(backboneEmbedding, state.PreviousToken); err != nil {
		return MTPDrafterStepResult{}, fmt.Errorf("drafter backbone embedding: %w", err)
	}
	assistantHidden := make([]float32, d.Config.HiddenSize)
	if err := d.PreProjectInto(assistantHidden, backboneEmbedding, state.Activation); err != nil {
		return MTPDrafterStepResult{}, err
	}
	if d.Config.NumLayers != 0 || len(d.Layers) != 0 {
		return MTPDrafterStepResult{}, fmt.Errorf("MTP drafter q-only layer forward not implemented")
	}
	nextActivation := make([]float32, d.BackboneHiddenSize)
	if err := d.PostProjectInto(nextActivation, assistantHidden); err != nil {
		return MTPDrafterStepResult{}, err
	}
	logits := make([]float32, m.Config.VocabSize)
	if err := m.LMHeadLogitsInto(logits, nextActivation); err != nil {
		return MTPDrafterStepResult{}, err
	}
	tok, _, err := ArgmaxLogits(logits)
	if err != nil {
		return MTPDrafterStepResult{}, err
	}
	nextState, err := NewMTPDrafterState(tok, nextActivation, d.BackboneHiddenSize)
	if err != nil {
		return MTPDrafterStepResult{}, err
	}
	return MTPDrafterStepResult{Token: tok, Logits: logits, NextActivation: append([]float32(nil), nextActivation...), NextState: nextState}, nil
}

func (m *LlamaModel) validateMTPDrafterStepModel(d *Gemma4MTPDrafter, state MTPDrafterState) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}
	if d == nil {
		return fmt.Errorf("nil drafter")
	}
	if d.Config.HiddenSize <= 0 || d.BackboneHiddenSize <= 0 || d.Config.VocabSize <= 0 {
		return fmt.Errorf("invalid drafter dims hidden=%d backbone=%d vocab=%d", d.Config.HiddenSize, d.BackboneHiddenSize, d.Config.VocabSize)
	}
	if m.Config.HiddenSize != d.BackboneHiddenSize || m.Config.VocabSize != d.Config.VocabSize {
		return fmt.Errorf("model/drafter dims mismatch model h/vocab=%d/%d drafter backbone/vocab=%d/%d", m.Config.HiddenSize, m.Config.VocabSize, d.BackboneHiddenSize, d.Config.VocabSize)
	}
	if state.PreviousToken < 0 || state.PreviousToken >= d.Config.VocabSize {
		return fmt.Errorf("previous token %d out of range [0,%d)", state.PreviousToken, d.Config.VocabSize)
	}
	if len(state.Activation) != d.BackboneHiddenSize {
		return fmt.Errorf("state activation len=%d, want %d", len(state.Activation), d.BackboneHiddenSize)
	}
	if len(d.PreProjection) == 0 || len(d.PostProjection) == 0 {
		return fmt.Errorf("drafter projection weights are not loaded")
	}
	return nil
}
