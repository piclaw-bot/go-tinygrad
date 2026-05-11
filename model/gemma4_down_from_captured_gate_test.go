//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4Layer15DownFromCapturedGateAct(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 captured gate-act down trace")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !gpu.Available() {
		t.Skip("GPU not available")
	}
	t.Cleanup(gpu.Shutdown)

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
	t.Cleanup(g.Close)
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	gpuGate := gpuOps[opTraceKey{15, "gate_act"}]
	gpuDownTrace := gpuOps[opTraceKey{15, "down"}]
	cpuDown := cpuOps[opTraceKey{15, "down"}]
	if len(gpuGate) == 0 || len(gpuDownTrace) == 0 || len(cpuDown) == 0 {
		t.Fatal("missing layer15 gpu gate/down or cpu down traces")
	}
	w := g.Layers[15].DownWmg
	if w == nil {
		t.Fatal("missing layer15 DownWmg")
	}

	run := func(name string, fn func(out, in *gpu.DevBuf, w *gpu.GPUMLXWeight)) {
		inBuf := gpu.NewDevBufFrom(append([]float32(nil), gpuGate...))
		outBuf := gpu.NewDevBuf(len(gpuDownTrace))
		defer inBuf.Free()
		defer outBuf.Free()
		if err := inBuf.ToGPU(); err != nil {
			t.Fatalf("%s inBuf.ToGPU: %v", name, err)
		}
		if err := outBuf.ToGPU(); err != nil {
			t.Fatalf("%s outBuf.ToGPU: %v", name, err)
		}
		fn(outBuf, inBuf, w)
		gpu.DevToBF16(outBuf, len(gpuDownTrace))
		gpu.Sync()
		got := append([]float32(nil), outBuf.Data()[:len(gpuDownTrace)]...)
		maxVsGPU, meanVsGPU := diffStats(gpuDownTrace, got)
		maxVsCPU, meanVsCPU := diffStats(cpuDown, got)
		t.Logf("captured gate -> %s down vs gpuTrace: maxAbs=%.6g meanAbs=%.6g", name, maxVsGPU, meanVsGPU)
		t.Logf("captured gate -> %s down vs cpuDown : maxAbs=%.6g meanAbs=%.6g", name, maxVsCPU, meanVsCPU)
	}

	run("native", gpu.GemvMLXDirect)
	run("fast", gpu.GemvMLX)
}
