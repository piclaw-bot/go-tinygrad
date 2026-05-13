package model

import "fmt"

var errMTPVerifierForwardNotImplemented = fmt.Errorf("MTP verifier forward not implemented")

// RunMTPVerifierForward is the future main-model verifier entrypoint. It will
// run a short batched forward over plan.VerifierTokens at plan.Positions,
// writing candidate KV into the provided staged caches and returning verifier
// logits plus the final activation. For now it validates the runtime contract
// and returns an explicit not-implemented error so generation cannot silently
// enable speculative decoding before the verifier path exists.
func (m *LlamaModel) RunMTPVerifierForward(plan MTPVerifierPlan, kvCacheK, kvCacheV [][]float32) (MTPVerifierResult, error) {
	if m == nil {
		return MTPVerifierResult{}, fmt.Errorf("nil model")
	}
	if len(plan.VerifierTokens) == 0 {
		return MTPVerifierResult{}, fmt.Errorf("empty verifier plan")
	}
	if len(plan.VerifierTokens) != len(plan.Positions) {
		return MTPVerifierResult{}, fmt.Errorf("verifier plan tokens=%d positions=%d", len(plan.VerifierTokens), len(plan.Positions))
	}
	if plan.InputToken != plan.VerifierTokens[0] {
		return MTPVerifierResult{}, fmt.Errorf("verifier plan input token=%d does not match first verifier token=%d", plan.InputToken, plan.VerifierTokens[0])
	}
	if len(plan.DraftedTokens)+1 != len(plan.VerifierTokens) {
		return MTPVerifierResult{}, fmt.Errorf("verifier plan drafted=%d tokens=%d", len(plan.DraftedTokens), len(plan.VerifierTokens))
	}
	vocab := m.Config.VocabSize
	if vocab <= 0 {
		return MTPVerifierResult{}, fmt.Errorf("invalid verifier vocab size %d", vocab)
	}
	for i, tok := range plan.VerifierTokens {
		if tok < 0 || tok >= vocab {
			return MTPVerifierResult{}, fmt.Errorf("verifier token %d at index %d out of range [0,%d)", tok, i, vocab)
		}
	}
	for i, tok := range plan.DraftedTokens {
		if tok != plan.VerifierTokens[i+1] {
			return MTPVerifierResult{}, fmt.Errorf("drafted token %d=%d does not match verifier token %d", i, tok, plan.VerifierTokens[i+1])
		}
	}
	wantPositions, err := mtpVerifierPositions(plan.StartPos, len(plan.VerifierTokens))
	if err != nil {
		return MTPVerifierResult{}, err
	}
	for i, pos := range plan.Positions {
		if pos != wantPositions[i] {
			return MTPVerifierResult{}, fmt.Errorf("verifier plan position %d=%d, want %d", i, pos, wantPositions[i])
		}
	}
	if len(kvCacheK) != len(m.Layers) || len(kvCacheV) != len(m.Layers) {
		return MTPVerifierResult{}, fmt.Errorf("KV cache layers K/V=%d/%d, want %d", len(kvCacheK), len(kvCacheV), len(m.Layers))
	}
	return MTPVerifierResult{}, errMTPVerifierForwardNotImplemented
}
