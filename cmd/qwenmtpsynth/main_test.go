package main

import (
	"encoding/json"
	"testing"

	"github.com/rcarmo/go-pherence/model"
)

func TestQwenMTPSynthReportJSON(t *testing.T) {
	report := Report{
		Passed:  true,
		Drafted: []int{1, 2},
		Stats:   model.QwenNativeMTPStats{DraftedTokens: 2, AcceptedTokens: 2, BonusTokens: 1, OutputTokens: 3},
	}
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var decoded Report
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !decoded.Passed || len(decoded.Drafted) != 2 || decoded.Stats.AcceptedTokens != 2 {
		t.Fatalf("decoded=%+v", decoded)
	}
}
