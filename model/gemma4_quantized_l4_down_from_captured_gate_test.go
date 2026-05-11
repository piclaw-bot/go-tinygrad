//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/quant"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma4QuantizedLayer4DownFromCapturedGate(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized layer4 down sensitivity")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	// Dequantized CPU baseline.
	oldForce := ForceOnTheFly
	ForceOnTheFly = false
	mCPU, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 dequantized cpu model: %v", err)
	}

	// Quantized CPU/GPU path.
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()
	mQ, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	mCPU.Tok = tok
	mQ.Tok = tok
	wrapped := wrapGemma4PromptForTest(mQ, "Hello")
	traceStep := len(wrapped) - 1

	gpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "gpu" || step != traceStep || layer != 4 {
			return
		}
		if op != "gate_act" && op != "down" {
			return
		}
		gpuOps[opTraceKey{layer, op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()

	g, err := LoadGPUModel(mQ)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	gpuGate := gpuOps[opTraceKey{4, "gate_act"}]
	gpuDown := gpuOps[opTraceKey{4, "down"}]
	if len(gpuGate) == 0 || len(gpuDown) == 0 {
		t.Fatal("missing layer4 gate_act/down traces")
	}

	deqDown := make([]float32, len(gpuDown))
	mCPU.mv(deqDown, gpuGate, mCPU.Layers[4].DownW.Data(), len(gpuGate), len(gpuDown))
	bf16Slice(deqDown)
	maxAbs, meanAbs := diffStats(gpuDown, deqDown)
	t.Logf("L4 dequantized cpu(down from gpu gate_act) vs gpu down: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	qDown := make([]float32, len(gpuDown))
	if mQ.Layers[4].DownWm == nil {
		t.Fatal("expected quantized MLX down weight for layer4")
	}
	quant.GemvMLQ(qDown, gpuGate, mQ.Layers[4].DownWm)
	bf16Slice(qDown)
	maxAbs, meanAbs = diffStats(gpuDown, qDown)
	t.Logf("L4 quantized cpu(down from gpu gate_act) vs gpu down: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	maxAbs, meanAbs = diffStats(deqDown, qDown)
	t.Logf("L4 dequantized cpu down vs quantized cpu down: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
