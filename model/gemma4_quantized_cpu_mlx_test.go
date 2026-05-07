package model

import (
	"os"
	"testing"
)

func TestGemma4StandaloneQuantizedCPUMLXVsCPU(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized CPU MLX trace")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 cpu model: %v", err)
	}
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	cpuOps := map[opTraceKey][]float32{}
	targetLayers := map[int]bool{0: true, 14: true, 15: true}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || step != traceStep || !targetLayers[layer] {
			return
		}
		cpuOps[opTraceKey{layer: layer, op: op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()
	_ = m.Generate(tok.Encode("Hello"), 1)

	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()
	mq, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized model: %v", err)
	}
	mq.Tok = tok

	check := func(layerIdx int, op string, in []float32, qw *MLXQuantWeight, want []float32, bf16Out bool) {
		got := make([]float32, len(want))
		GemvMLQ(got, in, qw)
		if bf16Out {
			bf16Slice(got)
		}
		maxAbs, meanAbs := diffStats(want, got)
		t.Logf("quantized cpu layer %d op %-4s: maxAbs=%.6g meanAbs=%.6g", layerIdx, op, maxAbs, meanAbs)
	}

	check(0, "q", cpuOps[opTraceKey{0, "normed"}], mq.Layers[0].QWm, cpuOps[opTraceKey{0, "q"}], true)
	check(14, "q", cpuOps[opTraceKey{14, "normed"}], mq.Layers[14].QWm, cpuOps[opTraceKey{14, "q"}], true)
	check(14, "o", cpuOps[opTraceKey{14, "attn"}], mq.Layers[14].OWm, cpuOps[opTraceKey{14, "o"}], false)
	check(15, "gate", cpuOps[opTraceKey{15, "mlp_input"}], mq.Layers[15].GateWm, cpuOps[opTraceKey{15, "gate_pre"}], true)
	check(15, "up", cpuOps[opTraceKey{15, "mlp_input"}], mq.Layers[15].UpWm, cpuOps[opTraceKey{15, "up"}], true)
	check(15, "down", cpuOps[opTraceKey{15, "gate_act"}], mq.Layers[15].DownWm, cpuOps[opTraceKey{15, "down"}], true)
}
