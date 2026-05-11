package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma4Layer15MLPSensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 layer15 MLP sensitivity")
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
	gpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if step != traceStep || layer != 15 {
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

	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()
	mgpu, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 gpu model: %v", err)
	}
	mgpu.Tok = tok
	g, err := LoadGPUModel(mgpu)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	mlpIn := gpuOps[opTraceKey{15, "mlp_input"}]
	gpuGate := gpuOps[opTraceKey{15, "gate_pre"}]
	gpuUp := gpuOps[opTraceKey{15, "up"}]
	gpuGateAct := gpuOps[opTraceKey{15, "gate_act"}]
	gpuDown := gpuOps[opTraceKey{15, "down"}]
	if len(mlpIn) == 0 || len(gpuGate) == 0 || len(gpuUp) == 0 || len(gpuGateAct) == 0 || len(gpuDown) == 0 {
		t.Fatal("missing gpu layer15 mlp traces")
	}

	layer := m.Layers[15]
	layerInter := len(gpuGate)
	gate := make([]float32, layerInter)
	up := make([]float32, layerInter)
	m.mv(gate, mlpIn, layer.GateW.Data(), len(mlpIn), layerInter)
	m.mv(up, mlpIn, layer.UpW.Data(), len(mlpIn), layerInter)
	bf16Slice(gate)
	bf16Slice(up)
	maxAbs, meanAbs := diffStats(gpuGate, gate)
	t.Logf("layer15 cpu(gate from gpu mlp_input) vs gpu gate_pre: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	maxAbs, meanAbs = diffStats(gpuUp, up)
	t.Logf("layer15 cpu(up from gpu mlp_input) vs gpu up: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	for i := range gate {
		gate[i] = geluTanh(gate[i]) * up[i]
	}
	bf16Slice(gate)
	maxAbs, meanAbs = diffStats(gpuGateAct, gate)
	t.Logf("layer15 cpu(gate_act from gpu mlp_input) vs gpu gate_act: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	down := make([]float32, len(gpuDown))
	m.mv(down, gate, layer.DownW.Data(), layerInter, len(down))
	bf16Slice(down)
	maxAbs, meanAbs = diffStats(gpuDown, down)
	t.Logf("layer15 cpu(down from gpu mlp_input) vs gpu down: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	// Also report how far this reconstructed CPU path is from the original CPU trace.
	maxAbs, meanAbs = diffStats(cpuOps[opTraceKey{15, "down"}], down)
	t.Logf("layer15 cpu(down from gpu mlp_input) vs cpu down: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
