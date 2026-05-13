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

// RunMTPDrafterStep is the future q-only assistant forward entrypoint. It will
// consume external/main-model KV state, use PreviousToken plus verifier
// activation carry, and return one drafted token plus the next projected
// activation. For now it validates the public contract and returns an explicit
// not-implemented error so speculative generation cannot silently enable before
// q-only attention is wired.
func (d *Gemma4MTPDrafter) RunMTPDrafterStep(state MTPDrafterState, backboneTokenEmbedding []float32) (token int, nextActivation []float32, err error) {
	if d == nil {
		return 0, nil, fmt.Errorf("nil drafter")
	}
	if d.Config.HiddenSize <= 0 || d.BackboneHiddenSize <= 0 || d.Config.VocabSize <= 0 {
		return 0, nil, fmt.Errorf("invalid drafter dims hidden=%d backbone=%d vocab=%d", d.Config.HiddenSize, d.BackboneHiddenSize, d.Config.VocabSize)
	}
	if state.PreviousToken < 0 || state.PreviousToken >= d.Config.VocabSize {
		return 0, nil, fmt.Errorf("previous token %d out of range [0,%d)", state.PreviousToken, d.Config.VocabSize)
	}
	if len(state.Activation) != d.BackboneHiddenSize {
		return 0, nil, fmt.Errorf("state activation len=%d, want %d", len(state.Activation), d.BackboneHiddenSize)
	}
	if len(backboneTokenEmbedding) != d.BackboneHiddenSize {
		return 0, nil, fmt.Errorf("backbone token embedding len=%d, want %d", len(backboneTokenEmbedding), d.BackboneHiddenSize)
	}
	if len(d.PreProjection) == 0 || len(d.PostProjection) == 0 {
		return 0, nil, fmt.Errorf("drafter projection weights are not loaded")
	}
	if d.Norm == nil || len(d.Layers) != d.Config.NumLayers {
		return 0, nil, fmt.Errorf("drafter layers/norm are not loaded")
	}
	return 0, nil, fmt.Errorf("MTP drafter forward not implemented")
}
