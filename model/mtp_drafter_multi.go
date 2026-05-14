package model

import "fmt"

const maxMTPDraftCount = 64

// MTPDrafterRunResult is a bounded internal drafter-only loop result. It is not
// a public generation API; verifier integration remains explicit and separate.
type MTPDrafterRunResult struct {
	Tokens      []int
	Logits      [][]float32
	Activations [][]float32
	FinalState  MTPDrafterState
}

// RunMTPDrafterSteps runs count one-step drafter iterations, carrying the
// hidden-state-conditioned drafter state between iterations.
func (m *LlamaModel) RunMTPDrafterSteps(d *Gemma4MTPDrafter, state MTPDrafterState, externalKV *MTPDrafterExternalKV, count int) (MTPDrafterRunResult, error) {
	if m == nil {
		return MTPDrafterRunResult{}, fmt.Errorf("nil model")
	}
	if count < 0 || count > maxMTPDraftCount {
		return MTPDrafterRunResult{}, fmt.Errorf("draft count %d out of range [0,%d]", count, maxMTPDraftCount)
	}
	if count == 0 {
		if err := m.validateMTPDrafterStepModel(d, state, externalKV); err != nil {
			return MTPDrafterRunResult{}, err
		}
		finalState, err := NewMTPDrafterState(state.PreviousToken, state.Activation, d.BackboneHiddenSize)
		if err != nil {
			return MTPDrafterRunResult{}, err
		}
		return MTPDrafterRunResult{FinalState: finalState}, nil
	}
	result := MTPDrafterRunResult{
		Tokens:      make([]int, 0, count),
		Logits:      make([][]float32, 0, count),
		Activations: make([][]float32, 0, count),
	}
	cur := state
	for i := 0; i < count; i++ {
		step, err := m.RunMTPDrafterStepWithExternalKV(d, cur, externalKV)
		if err != nil {
			return MTPDrafterRunResult{}, fmt.Errorf("MTP drafter step %d: %w", i, err)
		}
		result.Tokens = append(result.Tokens, step.Token)
		result.Logits = append(result.Logits, append([]float32(nil), step.Logits...))
		result.Activations = append(result.Activations, append([]float32(nil), step.NextActivation...))
		cur = step.NextState
	}
	result.FinalState = cur
	return result, nil
}
