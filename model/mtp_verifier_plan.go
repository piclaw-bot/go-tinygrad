package model

import "fmt"

// MTPVerifierPlan describes the token/position inputs for one main-model
// verifier pass. Tokens are always [inputToken] + draftedTokens and Positions
// are the absolute decode positions the verifier should write into staged KV.
type MTPVerifierPlan struct {
	InputToken     int
	DraftedTokens  []int
	VerifierTokens []int
	StartPos       int
	Positions      []int
}

// NewMTPVerifierPlan validates model/token bounds and prepares the verifier
// token and input_pos vectors for the future batched verifier forward path.
func NewMTPVerifierPlan(m *LlamaModel, inputToken int, drafted []int, startPos int) (MTPVerifierPlan, error) {
	if m == nil {
		return MTPVerifierPlan{}, fmt.Errorf("nil model")
	}
	if startPos < 0 {
		return MTPVerifierPlan{}, fmt.Errorf("start position %d out of range", startPos)
	}
	vocab := m.Config.VocabSize
	if vocab <= 0 {
		return MTPVerifierPlan{}, fmt.Errorf("invalid verifier vocab size %d", vocab)
	}
	verifierTokens, err := MTPVerifierTokens(inputToken, drafted)
	if err != nil {
		return MTPVerifierPlan{}, err
	}
	for i, tok := range verifierTokens {
		if tok >= vocab {
			return MTPVerifierPlan{}, fmt.Errorf("verifier token %d at index %d out of range [0,%d)", tok, i, vocab)
		}
	}
	positions, err := mtpVerifierPositions(startPos, len(verifierTokens))
	if err != nil {
		return MTPVerifierPlan{}, err
	}
	return MTPVerifierPlan{
		InputToken:     inputToken,
		DraftedTokens:  append([]int(nil), drafted...),
		VerifierTokens: verifierTokens,
		StartPos:       startPos,
		Positions:      positions,
	}, nil
}

// NewMTPVerifierPlanForModel is a compatibility alias for call sites that
// name the model-aware constructor explicitly.
func NewMTPVerifierPlanForModel(m *LlamaModel, inputToken int, drafted []int, startPos int) (MTPVerifierPlan, error) {
	return NewMTPVerifierPlan(m, inputToken, drafted, startPos)
}

func mtpVerifierPositions(startPos, count int) ([]int, error) {
	if startPos < 0 {
		return nil, fmt.Errorf("start position %d out of range", startPos)
	}
	if count < 0 {
		return nil, fmt.Errorf("verifier position count %d out of range", count)
	}
	maxInt := int(^uint(0) >> 1)
	if count > 0 && startPos > maxInt-(count-1) {
		return nil, fmt.Errorf("verifier positions overflow: start=%d count=%d", startPos, count)
	}
	positions := make([]int, count)
	for i := range positions {
		positions[i] = startPos + i
	}
	return positions, nil
}
