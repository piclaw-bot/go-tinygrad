package model

import (
	"fmt"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4QuantizedCPUvsGPULayerWalk(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized CPU vs GPU layer walk")
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
		t.Fatalf("load gemma4 quantized cpu model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1
	cpuLayers := map[int][]float32{}
	gpuLayers := map[int][]float32{}

	debugLayerHook = func(backend string, step, layer int, hidden []float32) {
		if step != traceStep {
			return
		}
		cp := append([]float32(nil), hidden...)
		if backend == "cpu" {
			cpuLayers[layer] = cp
		} else if backend == "gpu" {
			gpuLayers[layer] = cp
		}
	}
	defer func() { debugLayerHook = nil }()

	_ = m.Generate(tok.Encode("Hello"), 1)

	mgpu, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized gpu model: %v", err)
	}
	mgpu.Tok = tok
	g, err := LoadGPUModel(mgpu)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	firstOver1 := -1
	firstOver4 := -1
	for l := 0; l < len(m.Layers); l++ {
		c, ok1 := cpuLayers[l]
		g2, ok2 := gpuLayers[l]
		if !ok1 || !ok2 {
			t.Fatalf("missing layer trace %d (cpu=%v gpu=%v)", l, ok1, ok2)
		}
		maxAbs, meanAbs := diffStats(c, g2)
		if firstOver1 < 0 && maxAbs > 1.0 {
			firstOver1 = l
		}
		if firstOver4 < 0 && maxAbs > 4.0 {
			firstOver4 = l
		}
		t.Logf("layer %02d: maxAbs=%.6g meanAbs=%.6g", l, maxAbs, meanAbs)
	}
	t.Logf("first layer with maxAbs>1: %d", firstOver1)
	t.Logf("first layer with maxAbs>4: %d", firstOver4)
	fmt.Printf("[gemma4-layerwalk] first>1=%d first>4=%d\n", firstOver1, firstOver4)
}
