package model

import "fmt"

// RunMTPVerifierForward runs a CPU verifier pass over plan.VerifierTokens at
// plan.Positions, writing candidate K/V into the provided staged float caches
// and returning per-position logits plus the final verifier activation.
func (m *LlamaModel) RunMTPVerifierForward(plan MTPVerifierPlan, kvCacheK, kvCacheV [][]float32) (MTPVerifierResult, error) {
	if err := m.validateMTPVerifierForwardInputs(plan, kvCacheK, kvCacheV); err != nil {
		return MTPVerifierResult{}, err
	}
	logitsRows := make([][]float32, len(plan.VerifierTokens))
	var finalActivation []float32
	for i, tok := range plan.VerifierTokens {
		hidden := make([]float32, m.Config.HiddenSize)
		if err := m.ScaledTokenEmbeddingInto(hidden, tok); err != nil {
			return MTPVerifierResult{}, fmt.Errorf("verifier token %d embedding: %w", i, err)
		}
		pos := plan.Positions[i]
		for l := 0; l < m.Config.NumLayers; l++ {
			hidden = m.ForwardLayer(hidden, l, pos, pos, kvCacheK, kvCacheV)
			if hidden == nil {
				return MTPVerifierResult{}, fmt.Errorf("verifier forward layer %d at position %d failed", l, pos)
			}
		}
		activation, logits, _, err := m.finishCPUDecodeStep(hidden)
		if err != nil {
			return MTPVerifierResult{}, fmt.Errorf("verifier decode finish at position %d: %w", pos, err)
		}
		logitsRows[i] = logits
		finalActivation = activation
	}
	return NewMTPVerifierResultForModel(m, plan.InputToken, plan.DraftedTokens, logitsRows, finalActivation)
}

func (m *LlamaModel) validateMTPVerifierForwardInputs(plan MTPVerifierPlan, kvCacheK, kvCacheV [][]float32) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}
	if len(plan.VerifierTokens) == 0 {
		return fmt.Errorf("empty verifier plan")
	}
	if len(plan.VerifierTokens) != len(plan.Positions) {
		return fmt.Errorf("verifier plan tokens=%d positions=%d", len(plan.VerifierTokens), len(plan.Positions))
	}
	if plan.InputToken != plan.VerifierTokens[0] {
		return fmt.Errorf("verifier plan input token=%d does not match first verifier token=%d", plan.InputToken, plan.VerifierTokens[0])
	}
	if len(plan.DraftedTokens)+1 != len(plan.VerifierTokens) {
		return fmt.Errorf("verifier plan drafted=%d tokens=%d", len(plan.DraftedTokens), len(plan.VerifierTokens))
	}
	vocab := m.Config.VocabSize
	if vocab <= 0 || m.Config.HiddenSize <= 0 || m.Config.NumLayers < 0 || len(m.Layers) < m.Config.NumLayers {
		return fmt.Errorf("invalid verifier model dims vocab=%d hidden=%d layers=%d/%d", vocab, m.Config.HiddenSize, m.Config.NumLayers, len(m.Layers))
	}
	for i, tok := range plan.VerifierTokens {
		if tok < 0 || tok >= vocab {
			return fmt.Errorf("verifier token %d at index %d out of range [0,%d)", tok, i, vocab)
		}
	}
	for i, tok := range plan.DraftedTokens {
		if tok != plan.VerifierTokens[i+1] {
			return fmt.Errorf("drafted token %d=%d does not match verifier token %d", i, tok, plan.VerifierTokens[i+1])
		}
	}
	wantPositions, err := mtpVerifierPositions(plan.StartPos, len(plan.VerifierTokens))
	if err != nil {
		return err
	}
	for i, pos := range plan.Positions {
		if pos != wantPositions[i] {
			return fmt.Errorf("verifier plan position %d=%d, want %d", i, pos, wantPositions[i])
		}
	}
	if len(kvCacheK) != m.Config.NumLayers || len(kvCacheV) != m.Config.NumLayers {
		return fmt.Errorf("KV cache layers K/V=%d/%d, want %d", len(kvCacheK), len(kvCacheV), m.Config.NumLayers)
	}
	return nil
}
