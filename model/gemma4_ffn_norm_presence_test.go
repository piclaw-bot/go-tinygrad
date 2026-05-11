//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"
)

func TestGemma4FFNNormPresence(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 FFN norm presence")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4: %v", err)
	}
	for _, l := range []int{0, 14, 15, 34} {
		ly := m.Layers[l]
		t.Logf("layer %d: PreFFNNorm=%v PostFFNNorm=%v", l, ly.PreFFNNorm != nil, ly.PostFFNNorm != nil)
	}
}
