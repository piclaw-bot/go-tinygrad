package model

import "testing"

func TestMTPSpeculationStatsRecord(t *testing.T) {
	accepted, err := AcceptMTPDraft([]int{1, 2, 3}, []int{1, 2, 9, 4})
	if err != nil {
		t.Fatalf("AcceptMTPDraft: %v", err)
	}
	all, err := AcceptMTPDraft([]int{5}, []int{5, 6})
	if err != nil {
		t.Fatalf("AcceptMTPDraft: %v", err)
	}
	var stats MTPSpeculationStats
	if err := stats.Record(accepted); err != nil {
		t.Fatalf("Record accepted: %v", err)
	}
	if err := stats.Record(all); err != nil {
		t.Fatalf("Record all: %v", err)
	}
	if stats.Steps != 2 || stats.DraftedTokens != 4 || stats.VerifiedTokens != 3 || stats.BonusTokens != 2 || stats.OutputTokens != 5 {
		t.Fatalf("stats=%+v", stats)
	}
	if got, want := stats.AcceptanceRate(), 0.75; got != want {
		t.Fatalf("AcceptanceRate=%f want %f", got, want)
	}
}

func TestMTPSpeculationStatsValidateStepCapacity(t *testing.T) {
	if err := (MTPSpeculationStats{}).ValidateOneStepCapacity(); err != nil {
		t.Fatalf("ValidateOneStepCapacity empty: %v", err)
	}
	if err := (MTPSpeculationStats{}).ValidateStepCapacity(2); err != nil {
		t.Fatalf("ValidateStepCapacity empty: %v", err)
	}
	if err := (MTPSpeculationStats{}).ValidateStepCapacity(0); err == nil {
		t.Fatal("accepted zero draft count")
	}
	if err := (MTPSpeculationStats{}).ValidateStepCapacity(maxMTPDraftCount + 1); err == nil {
		t.Fatal("accepted oversized draft count")
	}
	if err := (MTPSpeculationStats{Steps: -1}).ValidateStepCapacity(1); err == nil {
		t.Fatal("accepted negative stats counter")
	}
	if err := (MTPSpeculationStats{Steps: int(^uint(0) >> 1)}).ValidateStepCapacity(1); err == nil {
		t.Fatal("accepted saturated stats counter")
	}
	maxInt := int(^uint(0) >> 1)
	if err := (MTPSpeculationStats{DraftedTokens: maxInt - 1}).ValidateStepCapacity(2); err == nil {
		t.Fatal("accepted multi-draft counter overflow")
	}
	if err := (MTPSpeculationStats{OutputTokens: maxInt - 2}).ValidateStepCapacity(2); err == nil {
		t.Fatal("accepted multi-output counter overflow")
	}
	if err := (MTPSpeculationStats{VerifiedTokens: maxInt}).ValidateStepCapacity(2); err != nil {
		t.Fatalf("rejected saturated verified counter before acceptance is known: %v", err)
	}
}

func TestMTPSpeculationStatsValidation(t *testing.T) {
	if err := (*MTPSpeculationStats)(nil).Record(MTPAcceptance{}); err == nil {
		t.Fatal("accepted nil stats")
	}
	var stats MTPSpeculationStats
	if err := stats.Record(MTPAcceptance{AcceptedPrefixLen: -1}); err == nil {
		t.Fatal("accepted malformed acceptance")
	}
	if got := stats.AcceptanceRate(); got != 0 {
		t.Fatalf("empty AcceptanceRate=%f want 0", got)
	}
	maxInt := int(^uint(0) >> 1)
	stats = MTPSpeculationStats{Steps: maxInt}
	accepted, err := AcceptMTPDraft(nil, []int{1})
	if err != nil {
		t.Fatalf("AcceptMTPDraft: %v", err)
	}
	if err := stats.Record(accepted); err == nil {
		t.Fatal("accepted overflowing stats")
	}
}
