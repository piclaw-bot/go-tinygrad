package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4StandaloneInputNormVsCPU(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 input-norm trace")
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
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	cpuOps := map[opTraceKey][]float32{}
	targetLayers := map[int]bool{0: true, 14: true}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || step != traceStep || !targetLayers[layer] {
			return
		}
		cpuOps[opTraceKey{layer: layer, op: op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()
	_ = m.Generate(tok.Encode("Hello"), 1)

	checkNorm := func(layerIdx int, input, want, weight []float32) {
		inBuf := gpu.NewDevBufFrom(append([]float32(nil), input...))
		outBuf := gpu.NewDevBuf(len(input))
		wbuf := gpu.NewDevBufFrom(append([]float32(nil), weight...))
		if err := inBuf.ToGPU(); err != nil {
			t.Fatalf("inBuf.ToGPU: %v", err)
		}
		if err := outBuf.ToGPU(); err != nil {
			t.Fatalf("outBuf.ToGPU: %v", err)
		}
		if err := wbuf.ToGPU(); err != nil {
			t.Fatalf("wbuf.ToGPU: %v", err)
		}
		gpu.DevRMSNorm(outBuf, inBuf, wbuf, float32(m.Config.RMSNormEps))
		gpu.DevToBF16(outBuf, len(input))
		gpu.Sync()
		got := append([]float32(nil), outBuf.Data()[:len(want)]...)
		maxAbs, meanAbs := diffStats(want, got)
		t.Logf("standalone layer %d input_norm: maxAbs=%.6g meanAbs=%.6g", layerIdx, maxAbs, meanAbs)
	}

	checkNorm(0, cpuOps[opTraceKey{0, "hidden_in"}], cpuOps[opTraceKey{0, "normed"}], m.Layers[0].InputNorm.Data())
	checkNorm(14, cpuOps[opTraceKey{14, "hidden_in"}], cpuOps[opTraceKey{14, "normed"}], m.Layers[14].InputNorm.Data())
}
