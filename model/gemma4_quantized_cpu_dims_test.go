//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"
)

func TestGemma4QuantizedCPUMLXDims(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized MLX dims")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized model: %v", err)
	}
	for _, l := range []int{0, 14, 15} {
		ly := m.Layers[l]
		if ly.QWm != nil {
			t.Logf("layer %d q dims: in=%d out=%d groups=%d", l, ly.QWm.InDim, ly.QWm.OutDim, ly.QWm.Groups)
		}
		if ly.GateWm != nil {
			t.Logf("layer %d gate dims: in=%d out=%d groups=%d", l, ly.GateWm.InDim, ly.GateWm.OutDim, ly.GateWm.Groups)
			t.Logf("layer %d up dims: in=%d out=%d groups=%d", l, ly.UpWm.InDim, ly.UpWm.OutDim, ly.UpWm.Groups)
			t.Logf("layer %d down dims: in=%d out=%d groups=%d", l, ly.DownWm.InDim, ly.DownWm.OutDim, ly.DownWm.Groups)
		}
	}
}
