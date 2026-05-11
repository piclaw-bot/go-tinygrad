package model

import "fmt"

// MTPAcceptance describes the deterministic accept/reject result for one
// speculative verification pass. VerifiedCount intentionally excludes the bonus
// token, matching LiteRT-LM's acceptance-rate accounting.
type MTPAcceptance struct {
	DraftedCount       int
	VerifiedCount      int
	AcceptedPrefixLen  int
	AcceptedTokens     []int
	BonusToken         int
	OutputTokens       []int
	AllDraftsAccepted  bool
	FirstRejectedIndex int // -1 when all drafts were accepted
}

// KVKeepTokens returns how many staged verifier positions must be retained in
// KV cache: accepted draft prefix plus the verifier bonus token. For zero-draft
// verifier passes this still returns 1, the ordinary verifier token.
func (a MTPAcceptance) KVKeepTokens() int {
	return a.AcceptedPrefixLen + 1
}

// CommitAcceptedFloatKV keeps the accepted verifier KV prefix plus bonus token.
// The checkpoint must have been taken immediately before the staged verifier
// pass whose logits produced acceptance.
func CommitAcceptedFloatKV(kvCacheK, kvCacheV [][]float32, cp FloatKVCheckpoint, kvDims []int, acceptance MTPAcceptance) error {
	return cp.KeepAppended(kvCacheK, kvCacheV, kvDims, acceptance.KVKeepTokens())
}

// CommitAcceptedCompressedKV keeps the accepted verifier KV prefix plus bonus
// token for compressed/TurboQuant-backed caches. The checkpoints must have been
// taken immediately before the staged verifier pass whose logits produced
// acceptance.
func CommitAcceptedCompressedKV(caches []*CompressedKVCache, cp []CompressedKVCheckpoint, acceptance MTPAcceptance) error {
	return KeepCompressedKVAppended(caches, cp, acceptance.KVKeepTokens())
}

// AcceptMTPDraftFromLogits greedily samples verifier logits and applies
// AcceptMTPDraft. The verifier must provide G+1 logit rows for G drafted IDs.
func AcceptMTPDraftFromLogits(drafted []int, verifierLogits [][]float32) (MTPAcceptance, error) {
	verifier := make([]int, len(verifierLogits))
	for i, logits := range verifierLogits {
		id, _, err := ArgmaxLogits(logits)
		if err != nil {
			return MTPAcceptance{}, fmt.Errorf("verifier logits row %d: %w", i, err)
		}
		verifier[i] = id
	}
	return AcceptMTPDraft(drafted, verifier)
}

// AcceptMTPDraft compares drafted token IDs with verifier greedy token IDs.
//
// The verifier must provide G+1 IDs for G drafted IDs: verifier[0:G] checks the
// drafted tokens, and verifier[G] is the bonus token when all drafts match. On
// the first mismatch at i, verifier[i] is emitted as the bonus token and the
// rejected draft suffix is discarded. With zero drafts, AllDraftsAccepted is
// vacuously true and OutputTokens contains the single verifier token.
func AcceptMTPDraft(drafted, verifier []int) (MTPAcceptance, error) {
	g := len(drafted)
	if len(verifier) != g+1 {
		return MTPAcceptance{}, fmt.Errorf("verifier token count=%d, want drafted+1=%d", len(verifier), g+1)
	}

	accepted := 0
	for accepted < g && drafted[accepted] == verifier[accepted] {
		accepted++
	}

	bonus := verifier[accepted]
	acceptedTokens := append([]int(nil), drafted[:accepted]...)
	output := make([]int, 0, accepted+1)
	output = append(output, acceptedTokens...)
	output = append(output, bonus)

	firstRejected := accepted
	allAccepted := accepted == g
	if allAccepted {
		firstRejected = -1
	}

	return MTPAcceptance{
		DraftedCount:       g,
		VerifiedCount:      accepted,
		AcceptedPrefixLen:  accepted,
		AcceptedTokens:     acceptedTokens,
		BonusToken:         bonus,
		OutputTokens:       output,
		AllDraftsAccepted:  allAccepted,
		FirstRejectedIndex: firstRejected,
	}, nil
}
