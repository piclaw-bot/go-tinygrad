//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4GPUInputNormWeightParity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 inputnorm weight parity")
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
	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	t.Cleanup(g.Close)

	cpuW := append([]float32(nil), m.Layers[0].InputNorm.Data()...)
	gpuW := append([]float32(nil), g.Layers[0].InputNorm.Data()...)
	maxAbs, meanAbs := diffStats(cpuW, gpuW)
	t.Logf("layer0 InputNorm weight parity: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
