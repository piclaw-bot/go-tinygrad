package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

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
	return newMTPVerifierResult(inputToken, drafted, logits, finalActivation, 0, 0)
}

// NewMTPVerifierResultForModel validates verifier outputs against model-owned
// dimensions. It is intended for the real verifier path; tests and low-level
// helpers may keep using NewMTPVerifierResult when no model is available.
func NewMTPVerifierResultForModel(m *LlamaModel, inputToken int, drafted []int, logits [][]float32, finalActivation []float32) (MTPVerifierResult, error) {
	if m == nil {
		return MTPVerifierResult{}, fmt.Errorf("nil model")
	}
	vocab := m.Config.VocabSize
	hidden := m.Config.HiddenSize
	if vocab <= 0 || hidden <= 0 {
		return MTPVerifierResult{}, fmt.Errorf("invalid verifier model dims vocab=%d hidden=%d", vocab, hidden)
	}
	return newMTPVerifierResult(inputToken, drafted, logits, finalActivation, vocab, hidden)
}

func newMTPVerifierResult(inputToken int, drafted []int, logits [][]float32, finalActivation []float32, vocab, hidden int) (MTPVerifierResult, error) {
	verifierTokens, err := MTPVerifierTokens(inputToken, drafted)
	if err != nil {
		return MTPVerifierResult{}, err
	}
	if vocab > 0 {
		for i, tok := range verifierTokens {
			if tok >= vocab {
				return MTPVerifierResult{}, fmt.Errorf("verifier token %d at index %d out of range [0,%d)", tok, i, vocab)
			}
		}
	}
	if hidden > 0 && len(finalActivation) != hidden {
		return MTPVerifierResult{}, fmt.Errorf("final activation len=%d, want %d", len(finalActivation), hidden)
	}
	if len(logits) != len(drafted)+1 {
		return MTPVerifierResult{}, fmt.Errorf("verifier logits rows=%d, want drafted+1=%d", len(logits), len(drafted)+1)
	}
	copiedLogits := make([][]float32, len(logits))
	for i, row := range logits {
		if len(row) == 0 {
			return MTPVerifierResult{}, fmt.Errorf("verifier logits row %d is empty", i)
		}
		if vocab > 0 && len(row) != vocab {
			return MTPVerifierResult{}, fmt.Errorf("verifier logits row %d len=%d, want vocab=%d", i, len(row), vocab)
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
func (r MTPVerifierResult) CommitFloatKV(m *LlamaModel, kvCacheK, kvCacheV [][]float32, cp kv.FloatKVCheckpoint) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}
	return m.CommitAcceptedFloatKV(kvCacheK, kvCacheV, cp, r.Acceptance)
}

// CommitCompressedKV applies the verifier result's acceptance to staged
// compressed/TurboQuant KV caches. The checkpoints must be from immediately
// before the verifier pass.
func (r MTPVerifierResult) CommitCompressedKV(caches []*kv.CompressedKVCache, cp []kv.CompressedKVCheckpoint) error {
	return CommitAcceptedCompressedKV(caches, cp, r.Acceptance)
}
