//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/quant"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma4QuantizedLayer0MLPSensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized layer0 MLP sensitivity")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

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

	gpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "gpu" || step != traceStep || layer != 0 {
			return
		}
		k := opTraceKey{layer: layer, op: op}
		gpuOps[k] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()

	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	mlpIn := gpuOps[opTraceKey{0, "mlp_input"}]
	gpuGate := gpuOps[opTraceKey{0, "gate_pre"}]
	gpuUp := gpuOps[opTraceKey{0, "up"}]
	gpuGateAct := gpuOps[opTraceKey{0, "gate_act"}]
	gpuDown := gpuOps[opTraceKey{0, "down"}]
	if len(mlpIn) == 0 || len(gpuGate) == 0 || len(gpuUp) == 0 || len(gpuGateAct) == 0 || len(gpuDown) == 0 {
		t.Fatal("missing layer0 mlp traces")
	}

	layer := m.Layers[0]
	gate := make([]float32, len(gpuGate))
	up := make([]float32, len(gpuUp))
	if layer.GateWm != nil {
		quant.GemvMLQ(gate, mlpIn, layer.GateWm)
		quant.GemvMLQ(up, mlpIn, layer.UpWm)
	} else {
		t.Fatal("expected quantized MLX weights for layer0 gate/up")
	}
	bf16Slice(gate)
	bf16Slice(up)
	maxAbs, meanAbs := diffStats(gpuGate, gate)
	t.Logf("L0 cpu(gate from gpu mlp_input) vs gpu gate_pre: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	maxAbs, meanAbs = diffStats(gpuUp, up)
	t.Logf("L0 cpu(up from gpu mlp_input) vs gpu up: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	gateAct := append([]float32(nil), gate...)
	for i := range gateAct {
		gateAct[i] = geluTanh(gateAct[i]) * up[i]
	}
	bf16Slice(gateAct)
	maxAbs, meanAbs = diffStats(gpuGateAct, gateAct)
	t.Logf("L0 cpu(gate_act from gpu mlp_input) vs gpu gate_act: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	down := make([]float32, len(gpuDown))
	if layer.DownWm != nil {
		quant.GemvMLQ(down, gateAct, layer.DownWm)
	} else {
		t.Fatal("expected quantized MLX weights for layer0 down")
	}
	bf16Slice(down)
	maxAbs, meanAbs = diffStats(gpuDown, down)
	t.Logf("L0 cpu(down from gpu mlp_input) vs gpu down: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
