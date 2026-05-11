//go:build diagnostic
// +build diagnostic

package model

import (
	"fmt"
	"math"
	"os"
	"testing"
)

func TestGemma4NormWeightValues(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	// Load WITHOUT the +1 offset (current code)
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	// Check layer 0 norms
	for _, name := range []string{"InputNorm", "PostNorm", "PreFFNNorm", "PostFFNNorm", "QNorm", "KNorm"} {
		var d []float32
		switch name {
		case "InputNorm":
			d = m.Layers[0].InputNorm.Data()
		case "PostNorm":
			d = m.Layers[0].PostNorm.Data()
		case "PreFFNNorm":
			if m.Layers[0].PreFFNNorm != nil {
				d = m.Layers[0].PreFFNNorm.Data()
			}
		case "PostFFNNorm":
			if m.Layers[0].PostFFNNorm != nil {
				d = m.Layers[0].PostFFNNorm.Data()
			}
		case "QNorm":
			if m.Layers[0].QNorm != nil {
				d = m.Layers[0].QNorm.Data()
			}
		case "KNorm":
			if m.Layers[0].KNorm != nil {
				d = m.Layers[0].KNorm.Data()
			}
		}
		if d == nil {
			continue
		}
		min, max, mean := float32(math.MaxFloat32), float32(-math.MaxFloat32), float32(0)
		for _, v := range d {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
			mean += v
		}
		mean /= float32(len(d))
		t.Logf("L0 %s: len=%d min=%.4f max=%.4f mean=%.4f", name, len(d), min, max, mean)
		fmt.Printf("L0 %s: min=%.4f max=%.4f mean=%.4f\n", name, min, max, mean)
	}
}
