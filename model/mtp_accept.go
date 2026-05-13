package model

import (
	"fmt"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

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
	if a.AcceptedPrefixLen < 0 {
		return 0
	}
	maxInt := int(^uint(0) >> 1)
	if a.AcceptedPrefixLen >= maxInt {
		return 0
	}
	return a.AcceptedPrefixLen + 1
}

// Validate checks that a caller-supplied acceptance result is internally
// consistent before it is used to commit staged verifier KV. Constructors in
// this file already produce valid values, but commit helpers can be called with
// manually assembled structs in tests or future generation code.
func (a MTPAcceptance) Validate() error {
	if a.DraftedCount < 0 || a.VerifiedCount < 0 || a.AcceptedPrefixLen < 0 {
		return fmt.Errorf("invalid MTP acceptance counts drafted=%d verified=%d accepted=%d", a.DraftedCount, a.VerifiedCount, a.AcceptedPrefixLen)
	}
	if a.AcceptedPrefixLen > a.DraftedCount {
		return fmt.Errorf("accepted prefix len=%d exceeds drafted count=%d", a.AcceptedPrefixLen, a.DraftedCount)
	}
	if a.VerifiedCount != a.AcceptedPrefixLen {
		return fmt.Errorf("verified count=%d differs from accepted prefix len=%d", a.VerifiedCount, a.AcceptedPrefixLen)
	}
	if len(a.AcceptedTokens) != a.AcceptedPrefixLen {
		return fmt.Errorf("accepted tokens len=%d, want accepted prefix len=%d", len(a.AcceptedTokens), a.AcceptedPrefixLen)
	}
	keep := a.KVKeepTokens()
	if keep <= 0 {
		return fmt.Errorf("accepted prefix len=%d cannot be converted to KV keep count", a.AcceptedPrefixLen)
	}
	if len(a.OutputTokens) != keep {
		return fmt.Errorf("output tokens len=%d, want accepted prefix plus bonus=%d", len(a.OutputTokens), keep)
	}
	for i, tok := range a.AcceptedTokens {
		if tok < 0 {
			return fmt.Errorf("accepted token %d at index %d out of range", tok, i)
		}
		if a.OutputTokens[i] != tok {
			return fmt.Errorf("output token %d=%d does not match accepted token %d", i, a.OutputTokens[i], tok)
		}
	}
	if a.BonusToken < 0 {
		return fmt.Errorf("bonus token %d out of range", a.BonusToken)
	}
	if a.OutputTokens[a.AcceptedPrefixLen] != a.BonusToken {
		return fmt.Errorf("output bonus token=%d does not match bonus=%d", a.OutputTokens[a.AcceptedPrefixLen], a.BonusToken)
	}
	if a.AllDraftsAccepted {
		if a.AcceptedPrefixLen != a.DraftedCount || a.FirstRejectedIndex != -1 {
			return fmt.Errorf("all-accepted state inconsistent: accepted=%d drafted=%d firstRejected=%d", a.AcceptedPrefixLen, a.DraftedCount, a.FirstRejectedIndex)
		}
	} else {
		if a.AcceptedPrefixLen >= a.DraftedCount {
			return fmt.Errorf("rejected state has accepted=%d drafted=%d", a.AcceptedPrefixLen, a.DraftedCount)
		}
		if a.FirstRejectedIndex != a.AcceptedPrefixLen {
			return fmt.Errorf("first rejected index=%d, want accepted prefix len=%d", a.FirstRejectedIndex, a.AcceptedPrefixLen)
		}
	}
	return nil
}

// CommitAcceptedFloatKV keeps the accepted verifier KV prefix plus bonus token.
// The checkpoint must have been taken immediately before the staged verifier
// pass whose logits produced acceptance.
func CommitAcceptedFloatKV(kvCacheK, kvCacheV [][]float32, cp kv.FloatKVCheckpoint, kvDims []int, acceptance MTPAcceptance) error {
	if err := acceptance.Validate(); err != nil {
		return err
	}
	keep := acceptance.KVKeepTokens()
	if keep <= 0 {
		return fmt.Errorf("invalid MTP KV keep token count from accepted prefix %d", acceptance.AcceptedPrefixLen)
	}
	return cp.KeepAppended(kvCacheK, kvCacheV, kvDims, keep)
}

// CommitAcceptedCompressedKV keeps the accepted verifier KV prefix plus bonus
// token for compressed/TurboQuant-backed caches. The checkpoints must have been
// taken immediately before the staged verifier pass whose logits produced
// acceptance.
func CommitAcceptedCompressedKV(caches []*kv.CompressedKVCache, cp []kv.CompressedKVCheckpoint, acceptance MTPAcceptance) error {
	if err := acceptance.Validate(); err != nil {
		return err
	}
	keep := acceptance.KVKeepTokens()
	if keep <= 0 {
		return fmt.Errorf("invalid MTP KV keep token count from accepted prefix %d", acceptance.AcceptedPrefixLen)
	}
	return kv.KeepCompressedKVAppended(caches, cp, keep)
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
	for i, tok := range drafted {
		if tok < 0 {
			return MTPAcceptance{}, fmt.Errorf("drafted token %d at index %d out of range", tok, i)
		}
	}
	for i, tok := range verifier {
		if tok < 0 {
			return MTPAcceptance{}, fmt.Errorf("verifier token %d at index %d out of range", tok, i)
		}
	}
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
