package model

import "fmt"

// MTPSpeculationStats accumulates LiteRT-style speculative decoding accounting.
// DraftedTokens counts proposed draft tokens; VerifiedTokens counts accepted
// draft-prefix tokens and deliberately excludes verifier bonus tokens.
type MTPSpeculationStats struct {
	Steps          int
	DraftedTokens  int
	VerifiedTokens int
	BonusTokens    int
	OutputTokens   int
}

// ValidateOneStepCapacity checks counters for the compatibility one-draft
// wrapper. Prefer ValidateStepCapacity when the draft count is known.
func (s MTPSpeculationStats) ValidateOneStepCapacity() error {
	return s.ValidateStepCapacity(1)
}

// ValidateStepCapacity checks counters that would make a speculative step fail
// regardless of verifier output. It lets callers reject obviously bad stats
// before mutating staged verifier KV. VerifiedTokens is deliberately not
// preflighted against draftCount because the accepted prefix is only known after
// verifier forward; post-verifier accounting failures restore staged KV.
func (s MTPSpeculationStats) ValidateStepCapacity(draftCount int) error {
	if draftCount <= 0 {
		return fmt.Errorf("draft count %d out of range", draftCount)
	}
	if s.Steps < 0 || s.DraftedTokens < 0 || s.VerifiedTokens < 0 || s.BonusTokens < 0 || s.OutputTokens < 0 {
		return fmt.Errorf("invalid MTP stats counters: %+v", s)
	}
	if _, ok := checkedAddNonNegative(s.Steps, 1); !ok {
		return fmt.Errorf("MTP stats step count cannot record another step: %+v", s)
	}
	if _, ok := checkedAddNonNegative(s.DraftedTokens, draftCount); !ok {
		return fmt.Errorf("MTP stats drafted token count cannot record %d drafts: %+v", draftCount, s)
	}
	if _, ok := checkedAddNonNegative(s.BonusTokens, 1); !ok {
		return fmt.Errorf("MTP stats bonus token count cannot record another step: %+v", s)
	}
	if _, ok := checkedAddNonNegative(s.OutputTokens, draftCount+1); !ok {
		return fmt.Errorf("MTP stats output token count cannot record at most %d outputs: %+v", draftCount+1, s)
	}
	return nil
}

// Record adds one verifier acceptance result to the accounting totals.
func (s *MTPSpeculationStats) Record(a MTPAcceptance) error {
	if s == nil {
		return fmt.Errorf("nil MTP stats")
	}
	if err := a.Validate(); err != nil {
		return err
	}
	steps, ok := checkedAddNonNegative(s.Steps, 1)
	if !ok {
		return fmt.Errorf("MTP stats step count overflows")
	}
	drafted, ok := checkedAddNonNegative(s.DraftedTokens, a.DraftedCount)
	if !ok {
		return fmt.Errorf("MTP stats drafted token count overflows")
	}
	verified, ok := checkedAddNonNegative(s.VerifiedTokens, a.VerifiedCount)
	if !ok {
		return fmt.Errorf("MTP stats verified token count overflows")
	}
	bonus, ok := checkedAddNonNegative(s.BonusTokens, 1)
	if !ok {
		return fmt.Errorf("MTP stats bonus token count overflows")
	}
	output, ok := checkedAddNonNegative(s.OutputTokens, len(a.OutputTokens))
	if !ok {
		return fmt.Errorf("MTP stats output token count overflows")
	}
	s.Steps = steps
	s.DraftedTokens = drafted
	s.VerifiedTokens = verified
	s.BonusTokens = bonus
	s.OutputTokens = output
	return nil
}

// AcceptanceRate returns accepted draft tokens / drafted tokens. Bonus tokens
// are excluded to match LiteRT-LM accounting.
func (s MTPSpeculationStats) AcceptanceRate() float64 {
	if s.DraftedTokens <= 0 {
		return 0
	}
	return float64(s.VerifiedTokens) / float64(s.DraftedTokens)
}

func checkedAddNonNegative(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if a > maxInt-b {
		return 0, false
	}
	return a + b, true
}
