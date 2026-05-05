package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4StandaloneMLXGemvVsCPU(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 standalone GEMV trace")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !gpu.Available() {
		t.Skip("GPU not available")
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4: %v", err)
	}
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	cpuOps := map[opTraceKey][]float32{}
	targetLayers := map[int]bool{0: true, 14: true, 15: true, 34: true}
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
	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}

	checkGemv := func(layerIdx int, op string, in []float32, w *gpu.GPUMLXWeight, want []float32) {
		inBuf := gpu.NewDevBufFrom(append([]float32(nil), in...))
		outBuf := gpu.NewDevBuf(len(want))
		if err := inBuf.ToGPU(); err != nil {
			t.Fatalf("inBuf.ToGPU: %v", err)
		}
		if err := outBuf.ToGPU(); err != nil {
			t.Fatalf("outBuf.ToGPU: %v", err)
		}
		gpu.GemvMLXDirect(outBuf, inBuf, w)
		gpu.Sync()
		got := append([]float32(nil), outBuf.Data()[:len(want)]...)
		maxAbs, meanAbs := diffStats(want, got)
		t.Logf("standalone layer %d op %-4s: maxAbs=%.6g meanAbs=%.6g", layerIdx, op, maxAbs, meanAbs)
	}

	// layer 0 q should match nearly exactly on standalone input
	checkGemv(0, "q", cpuOps[opTraceKey{0, "normed"}], g.Layers[0].QWmg, cpuOps[opTraceKey{0, "q"}])
	// layer 14 q and o help isolate whether later drift is inherited or intrinsic to GEMV
	checkGemv(14, "q", cpuOps[opTraceKey{14, "normed"}], g.Layers[14].QWmg, cpuOps[opTraceKey{14, "q"}])
	checkGemv(14, "o", cpuOps[opTraceKey{14, "attn"}], g.Layers[14].OWmg, cpuOps[opTraceKey{14, "o"}])
	// layer 15 and 34 down projections exercise later MLP kernels on real inputs
	checkGemv(15, "down", cpuOps[opTraceKey{15, "gate_act"}], g.Layers[15].DownWmg, cpuOps[opTraceKey{15, "down"}])
	checkGemv(34, "down", cpuOps[opTraceKey{34, "gate_act"}], g.Layers[34].DownWmg, cpuOps[opTraceKey{34, "down"}])
}
