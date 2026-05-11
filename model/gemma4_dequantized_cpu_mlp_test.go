//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func gemvNNScalar(out, x, w []float32, inDim, outDim int) {
	for j := 0; j < outDim; j++ {
		sum := float32(0)
		for p := 0; p < inDim; p++ {
			sum += x[p] * w[p*outDim+j]
		}
		out[j] = sum
	}
}

func TestGemma4Layer15DequantizedCPUMLPParity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 dequantized CPU MLP trace")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 cpu model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
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

	check := func(op string, in []float32, w []float32, outDim int, want []float32, bf16Out bool) {
		got := make([]float32, outDim)
		gemvNNScalar(got, in, w, len(in), outDim)
		if bf16Out {
			bf16Slice(got)
		}
		maxAbs, meanAbs := diffStats(want, got)
		t.Logf("dequantized cpu layer15 %-4s scalar-vs-trace: maxAbs=%.6g meanAbs=%.6g", op, maxAbs, meanAbs)
	}

	layer := m.Layers[15]
	check("gate", cpuOps[opTraceKey{15, "mlp_input"}], layer.GateW.Data(), len(cpuOps[opTraceKey{15, "gate_pre"}]), cpuOps[opTraceKey{15, "gate_pre"}], true)
	check("up", cpuOps[opTraceKey{15, "mlp_input"}], layer.UpW.Data(), len(cpuOps[opTraceKey{15, "up"}]), cpuOps[opTraceKey{15, "up"}], true)
	check("down", cpuOps[opTraceKey{15, "gate_act"}], layer.DownW.Data(), len(cpuOps[opTraceKey{15, "down"}]), cpuOps[opTraceKey{15, "down"}], true)
}
