package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestMoECPUvsGPUExpert(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1")
	}
	dir := "../../models/qwen3-30b-a3b-mlx4"
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		dir = "../models/qwen3-30b-a3b-mlx4"
	}
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skip("model not found")
	}
	if !gpu.Available() {
		t.Skip("no GPU")
	}
	t.Cleanup(gpu.Shutdown)

	ForceOnTheFly = true
	defer func() { ForceOnTheFly = false }()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatal(err)
	}

	layer := &m.Layers[0]
	cfg := m.Config
	h := cfg.HiddenSize
	x := make([]float32, h)
	for i := range x {
		x[i] = float32(i%7-3) * 0.01
	}

	cpuOut := moeForward(x, layer, cfg)

	pool := gpu.NewExpertPool(200, nil)
	gpuOut := moeForwardGPU(x, layer, cfg, pool, 0, nil)

	maxAbs := float64(0)
	for i := range cpuOut {
		d := float64(cpuOut[i] - gpuOut[i])
		if d < 0 {
			d = -d
		}
		if d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("moeForward CPU vs GPU: maxAbs=%.6g", maxAbs)
	t.Logf("  CPU[0:5]: %v", cpuOut[:5])
	t.Logf("  GPU[0:5]: %v", gpuOut[:5])
	if maxAbs > 0.01 {
		t.Errorf("too much divergence: maxAbs=%.6g", maxAbs)
	}
}
