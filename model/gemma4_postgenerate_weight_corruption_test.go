//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4PostGenerateWeightParity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 post-generate weight parity")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !gpu.Available() {
		t.Skip("GPU not available")
	}
	t.Cleanup(gpu.Shutdown)

	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 gpu model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	t.Cleanup(g.Close)
	g.CPU.Tok = tok

	beforeIn := append([]float32(nil), g.Layers[0].InputNorm.Data()...)
	beforeQN := append([]float32(nil), g.Layers[0].QNorm.Data()...)
	beforeKN := append([]float32(nil), g.Layers[0].KNorm.Data()...)
	beforePLIN := append([]float32(nil), g.perLayerProjNorm.Data()...)

	_ = g.Generate(tok.Encode("Hello"), 1)

	afterIn := append([]float32(nil), g.Layers[0].InputNorm.Data()...)
	afterQN := append([]float32(nil), g.Layers[0].QNorm.Data()...)
	afterKN := append([]float32(nil), g.Layers[0].KNorm.Data()...)
	afterPLIN := append([]float32(nil), g.perLayerProjNorm.Data()...)

	for _, tc := range []struct {
		name string
		a, b []float32
	}{
		{"InputNorm", beforeIn, afterIn},
		{"QNorm", beforeQN, afterQN},
		{"KNorm", beforeKN, afterKN},
		{"PerLayerProjNorm", beforePLIN, afterPLIN},
	} {
		maxAbs, meanAbs := diffStats(tc.a, tc.b)
		t.Logf("post-generate %s parity: maxAbs=%.6g meanAbs=%.6g", tc.name, maxAbs, meanAbs)
	}
}
