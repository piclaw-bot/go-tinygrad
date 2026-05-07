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
	t.Cleanup(gpu.Shutdown)

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

	checkGemv := func(layerIdx int, op string, in []float32, w *gpu.GPUMLXWeight, want []float32) {
		if w == nil {
			t.Logf("standalone layer %d op %-4s paths: weight=nil", layerIdx, op)
		} else {
			t.Logf("standalone layer %d op %-4s paths: inDim=%d outDim=%d native=%v gptq=%v correction=%v", layerIdx, op, w.InDim, w.OutDim, w.QWeight != nil, w.AsGPTQ != nil, w.Correction != nil)
		}

		inBuf := gpu.NewDevBufFrom(append([]float32(nil), in...))
		outBuf := gpu.NewDevBuf(len(want))
		defer inBuf.Free()
		defer outBuf.Free()
		if err := inBuf.ToGPU(); err != nil {
			t.Fatalf("inBuf.ToGPU: %v", err)
		}
		if err := outBuf.ToGPU(); err != nil {
			t.Fatalf("outBuf.ToGPU: %v", err)
		}
		gpu.GemvMLXDirect(outBuf, inBuf, w)
		if op == "q" || op == "gate" || op == "up" {
			gpu.DevToBF16(outBuf, len(want))
		}
		gpu.Sync()
		got := append([]float32(nil), outBuf.Data()[:len(want)]...)
		maxAbs, meanAbs := diffStats(want, got)
		mode := "native"
		if w != nil && w.QWeight == nil && w.AsGPTQ != nil {
			mode = "fallback-gptq"
		}
		t.Logf("standalone layer %d op %-4s direct(%s): maxAbs=%.6g meanAbs=%.6g", layerIdx, op, mode, maxAbs, meanAbs)

		if w != nil && w.AsGPTQ != nil {
			inBuf2 := gpu.NewDevBufFrom(append([]float32(nil), in...))
			outBuf2 := gpu.NewDevBuf(len(want))
			defer inBuf2.Free()
			defer outBuf2.Free()
			if err := inBuf2.ToGPU(); err != nil {
				t.Fatalf("inBuf2.ToGPU: %v", err)
			}
			if err := outBuf2.ToGPU(); err != nil {
				t.Fatalf("outBuf2.ToGPU: %v", err)
			}
			gpu.GemvMLX(outBuf2, inBuf2, w)
			if op == "q" || op == "gate" || op == "up" {
				gpu.DevToBF16(outBuf2, len(want))
			}
			gpu.Sync()
			got2 := append([]float32(nil), outBuf2.Data()[:len(want)]...)
			maxAbs2, meanAbs2 := diffStats(want, got2)
			t.Logf("standalone layer %d op %-4s fast(gptq): maxAbs=%.6g meanAbs=%.6g", layerIdx, op, maxAbs2, meanAbs2)
		}
	}

	// layer 0 q should match nearly exactly on standalone input
	checkGemv(0, "q", cpuOps[opTraceKey{0, "normed"}], g.Layers[0].QWmg, cpuOps[opTraceKey{0, "q"}])
	// layer 14 q and o help isolate whether later drift is inherited or intrinsic to GEMV
	checkGemv(14, "q", cpuOps[opTraceKey{14, "normed"}], g.Layers[14].QWmg, cpuOps[opTraceKey{14, "q"}])
	checkGemv(14, "o", cpuOps[opTraceKey{14, "attn"}], g.Layers[14].OWmg, cpuOps[opTraceKey{14, "o"}])
	// layer 15/34 MLP kernels on real inputs
	checkGemv(15, "gate", cpuOps[opTraceKey{15, "mlp_input"}], g.Layers[15].GateWmg, cpuOps[opTraceKey{15, "gate_pre"}])
	checkGemv(15, "up", cpuOps[opTraceKey{15, "mlp_input"}], g.Layers[15].UpWmg, cpuOps[opTraceKey{15, "up"}])
	checkGemv(15, "down", cpuOps[opTraceKey{15, "gate_act"}], g.Layers[15].DownWmg, cpuOps[opTraceKey{15, "down"}])
}
