//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4QuantizedCPUvsGPUOpTraceEarly(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized CPU vs GPU early op trace")
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
	targetLayers := map[int]bool{7: true, 8: true}
	cpuOps := map[opTraceKey][]float32{}
	gpuOps := map[opTraceKey][]float32{}

	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if step != traceStep || !targetLayers[layer] {
			return
		}
		cp := append([]float32(nil), vec...)
		k := opTraceKey{layer: layer, op: op}
		if backend == "cpu" {
			cpuOps[k] = cp
		} else if backend == "gpu" {
			gpuOps[k] = cp
		}
	}
	defer func() { debugOpHook = nil }()

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

	opOrder := []string{
		"hidden_in",
		"normed",
		"q",
		"q_qknorm",
		"q_attn",
		"k_attn",
		"v_attn",
		"attn",
		"o",
		"mlp_input",
		"gate_pre",
		"up",
		"gate_act",
		"down",
	}
	for _, layer := range []int{7, 8} {
		for _, op := range opOrder {
			k := opTraceKey{layer: layer, op: op}
			c, ok1 := cpuOps[k]
			g2, ok2 := gpuOps[k]
			if !ok1 || !ok2 {
				continue
			}
			maxAbs, meanAbs := diffStats(c, g2)
			t.Logf("L%d %-10s maxAbs=%.6g meanAbs=%.6g", layer, op, maxAbs, meanAbs)
		}
	}
}
