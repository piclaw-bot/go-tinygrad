package model

import (
	"os"
	"testing"
)

func TestGemma4Layer15MVVsScalar(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 mv parity trace")
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
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || step != traceStep || layer != 15 {
			return
		}
		cpuOps[opTraceKey{layer: layer, op: op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()
	_ = m.Generate(tok.Encode("Hello"), 1)

	check := func(op string, in []float32, w []float32, outDim int, bf16Out bool) {
		mvOut := make([]float32, outDim)
		m.mv(mvOut, in, w, len(in), outDim)
		scalarOut := make([]float32, outDim)
		gemvNNScalar(scalarOut, in, w, len(in), outDim)
		if bf16Out {
			bf16Slice(mvOut)
			bf16Slice(scalarOut)
		}
		maxAbs, meanAbs := diffStats(mvOut, scalarOut)
		t.Logf("layer15 %s m.mv vs scalar: maxAbs=%.6g meanAbs=%.6g", op, maxAbs, meanAbs)
	}

	layer := m.Layers[15]
	check("gate", cpuOps[opTraceKey{15, "mlp_input"}], layer.GateW.Data(), len(cpuOps[opTraceKey{15, "gate_pre"}]), true)
	check("up", cpuOps[opTraceKey{15, "mlp_input"}], layer.UpW.Data(), len(cpuOps[opTraceKey{15, "up"}]), true)
	check("down", cpuOps[opTraceKey{15, "gate_act"}], layer.DownW.Data(), len(cpuOps[opTraceKey{15, "down"}]), true)
}
