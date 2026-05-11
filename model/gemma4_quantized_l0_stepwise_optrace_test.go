//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4QuantizedLayer0StepwiseOpTrace(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized layer0 stepwise op trace")
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

	cpuOps := map[frontStepOpKey][]float32{}
	gpuOps := map[frontStepOpKey][]float32{}
	targetOps := map[string]bool{
		"normed":    true,
		"q":         true,
		"q_qknorm":  true,
		"q_attn":    true,
		"attn":      true,
		"o":         true,
		"mlp_input": true,
	}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if layer != 0 || !targetOps[op] {
			return
		}
		k := frontStepOpKey{step: step, op: op}
		cp := append([]float32(nil), vec...)
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

	for _, op := range []string{"normed", "q", "q_qknorm", "q_attn", "attn", "o", "mlp_input"} {
		worstStep := -1
		worstMax := -1.0
		worstMean := 0.0
		for k, gv := range gpuOps {
			if k.op != op {
				continue
			}
			cv, ok := cpuOps[k]
			if !ok {
				continue
			}
			maxAbs, meanAbs := diffStats(cv, gv)
			if maxAbs > worstMax {
				worstMax = maxAbs
				worstMean = meanAbs
				worstStep = k.step
			}
		}
		t.Logf("L0 worst %-8s step=%d maxAbs=%.6g meanAbs=%.6g", op, worstStep, worstMax, worstMean)
	}
}
