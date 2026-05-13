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

// ValidateOneStepCapacity checks counters that would make a one-step speculative
// wrapper fail regardless of verifier output. It lets callers reject obviously
// bad stats before mutating staged verifier KV.
func (s MTPSpeculationStats) ValidateOneStepCapacity() error {
	maxInt := int(^uint(0) >> 1)
	if s.Steps < 0 || s.DraftedTokens < 0 || s.VerifiedTokens < 0 || s.BonusTokens < 0 || s.OutputTokens < 0 {
		return fmt.Errorf("invalid MTP stats counters: %+v", s)
	}
	if s.Steps == maxInt || s.DraftedTokens == maxInt || s.VerifiedTokens == maxInt || s.BonusTokens == maxInt || s.OutputTokens == maxInt {
		return fmt.Errorf("MTP stats counters cannot record another step: %+v", s)
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
