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

// AcceptMTPDraft compares drafted token IDs with verifier greedy token IDs.
//
// The verifier must provide G+1 IDs for G drafted IDs: verifier[0:G] checks the
// drafted tokens, and verifier[G] is the bonus token when all drafts match. On
// the first mismatch at i, verifier[i] is emitted as the bonus token and the
// rejected draft suffix is discarded.
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
