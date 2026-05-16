package main

import (
	"encoding/json"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

func TestShouldFailStrict(t *testing.T) {
	meta := loaderconfig.QwenNativeMTPMetadata{HasNativeMTP: true}
	if !shouldFailStrict(true, meta, Report{}) {
		t.Fatal("strict incomplete native MTP did not fail")
	}
	if shouldFailStrict(false, meta, Report{}) {
		t.Fatal("non-strict failed")
	}
	if shouldFailStrict(true, meta, Report{MTPTensorComplete: true}) {
		t.Fatal("complete tensor set failed")
	}
	if shouldFailStrict(true, loaderconfig.QwenNativeMTPMetadata{}, Report{}) {
		t.Fatal("non-MTP metadata failed")
	}
}

func TestReportCanLoadSharedHeadJSON(t *testing.T) {
	report := Report{Config: loaderconfig.QwenNativeMTPMetadata{HiddenSize: 4, VocabSize: 2}, OptionalSharedHeadTensors: []string{"mtp.shared_head_head.weight"}, CanLoadSharedHead: true, MTPTensorCount: 3, OptionalSharedHeadCount: 1, MissingMTPTensorCount: 2, MTPTensorComplete: false}
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var decoded Report
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !decoded.CanLoadSharedHead || decoded.Config.VocabSize != 2 || len(decoded.OptionalSharedHeadTensors) != 1 || decoded.MTPTensorCount != 3 || decoded.OptionalSharedHeadCount != 1 || decoded.MissingMTPTensorCount != 2 || decoded.MTPTensorComplete {
		t.Fatalf("decoded=%+v", decoded)
	}
}
