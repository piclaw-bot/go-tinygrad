package main

import (
	"encoding/json"
	"testing"
)

func TestReportJSON(t *testing.T) {
	report := Report{ModelDir: "x", HiddenSize: 4, MTPLayers: 1, OutputLen: 4, OutputAbsSum: 1.5, Passed: true}
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var decoded Report
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !decoded.Passed || decoded.HiddenSize != 4 || decoded.OutputAbsSum != 1.5 {
		t.Fatalf("decoded=%+v", decoded)
	}
}
