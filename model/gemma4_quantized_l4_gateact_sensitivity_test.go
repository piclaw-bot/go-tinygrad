//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4QuantizedLayer4GateActSensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized layer4 gate_act sensitivity")
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
		t.Fatalf("load gemma4 quantized model: %v", err)
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
		if backend != "gpu" || step != traceStep || layer != 4 {
			return
		}
		if op != "gate_pre" && op != "up" && op != "gate_act" {
			return
		}
		gpuOps[opTraceKey{layer, op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()

	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	gpuGate := gpuOps[opTraceKey{4, "gate_pre"}]
	gpuUp := gpuOps[opTraceKey{4, "up"}]
	gpuGateAct := gpuOps[opTraceKey{4, "gate_act"}]
	if len(gpuGate) == 0 || len(gpuUp) == 0 || len(gpuGateAct) == 0 {
		t.Fatal("missing layer4 gate_pre/up/gate_act traces")
	}

	cpuAct := append([]float32(nil), gpuGate...)
	for i := range cpuAct {
		cpuAct[i] = geluTanh(cpuAct[i]) * gpuUp[i]
	}
	bf16Slice(cpuAct)
	maxAbs, meanAbs := diffStats(gpuGateAct, cpuAct)
	t.Logf("L4 cpu(gate_act from gpu gate_pre/up) vs gpu gate_act: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	gateBuf := gpu.NewDevBufFrom(append([]float32(nil), gpuGate...))
	upBuf := gpu.NewDevBufFrom(append([]float32(nil), gpuUp...))
	defer gateBuf.Free()
	defer upBuf.Free()
	if err := gateBuf.ToGPU(); err != nil {
		t.Fatalf("gateBuf.ToGPU: %v", err)
	}
	if err := upBuf.ToGPU(); err != nil {
		t.Fatalf("upBuf.ToGPU: %v", err)
	}
	gpu.DevGELUTanhMul(gateBuf, upBuf, len(gpuGate))
	gpu.DevToBF16(gateBuf, len(gpuGate))
	gpu.Sync()
	freshGPU := append([]float32(nil), gateBuf.Data()[:len(gpuGateAct)]...)
	maxAbs, meanAbs = diffStats(gpuGateAct, freshGPU)
	t.Logf("L4 fresh gpu(gate_act from gpu gate_pre/up) vs gpu gate_act: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
