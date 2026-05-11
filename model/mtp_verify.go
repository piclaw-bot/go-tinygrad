package model

import "fmt"

// MTPVerifierTokens returns the token sequence the main verifier must process:
// the previous/input token followed by G drafter candidates.
func MTPVerifierTokens(inputToken int, drafted []int) ([]int, error) {
	if inputToken < 0 {
		return nil, fmt.Errorf("input token %d out of range", inputToken)
	}
	for i, tok := range drafted {
		if tok < 0 {
			return nil, fmt.Errorf("drafted token %d at index %d out of range", tok, i)
		}
	}
	tokens := make([]int, 0, len(drafted)+1)
	tokens = append(tokens, inputToken)
	tokens = append(tokens, drafted...)
	return tokens, nil
}

// MTPVerifierResult is the contract filled by a main-model verifier forward.
// Logits must contain one row per verifier position: G+1 rows for G drafted
// tokens. FinalActivation is the main-model activation that seeds the next
// drafter call.
type MTPVerifierResult struct {
	InputToken      int
	DraftedTokens   []int
	VerifierTokens  []int // [inputToken] + draftedTokens
	Logits          [][]float32
	FinalActivation []float32
	Acceptance      MTPAcceptance
}

// NewMTPVerifierResult validates verifier outputs, derives greedy acceptance,
// and copies slice headers/data that callers commonly mutate after verification.
func NewMTPVerifierResult(inputToken int, drafted []int, logits [][]float32, finalActivation []float32) (MTPVerifierResult, error) {
	verifierTokens, err := MTPVerifierTokens(inputToken, drafted)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	if len(logits) != len(drafted)+1 {
		return MTPVerifierResult{}, fmt.Errorf("verifier logits rows=%d, want drafted+1=%d", len(logits), len(drafted)+1)
	}
	copiedLogits := make([][]float32, len(logits))
	for i, row := range logits {
		if len(row) == 0 {
			return MTPVerifierResult{}, fmt.Errorf("verifier logits row %d is empty", i)
		}
		copiedLogits[i] = append([]float32(nil), row...)
	}
	acceptance, err := AcceptMTPDraftFromLogits(drafted, copiedLogits)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	return MTPVerifierResult{
		InputToken:      inputToken,
		DraftedTokens:   append([]int(nil), drafted...),
		VerifierTokens:  verifierTokens,
		Logits:          copiedLogits,
		FinalActivation: append([]float32(nil), finalActivation...),
		Acceptance:      acceptance,
	}, nil
}

// CommitFloatKV applies the verifier result's acceptance to staged uncompressed
// KV caches. The checkpoint must be from immediately before the verifier pass.
func (r MTPVerifierResult) CommitFloatKV(m *LlamaModel, kvCacheK, kvCacheV [][]float32, cp FloatKVCheckpoint) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}
	return m.CommitAcceptedFloatKV(kvCacheK, kvCacheV, cp, r.Acceptance)
}

// CommitCompressedKV applies the verifier result's acceptance to staged
// compressed/TurboQuant KV caches. The checkpoints must be from immediately
// before the verifier pass.
func (r MTPVerifierResult) CommitCompressedKV(caches []*CompressedKVCache, cp []CompressedKVCheckpoint) error {
	return CommitAcceptedCompressedKV(caches, cp, r.Acceptance)
}
