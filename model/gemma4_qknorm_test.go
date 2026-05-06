package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4StandaloneQKNormVsCPU(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 QK-norm trace")
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
	targetLayers := map[int]bool{0: true, 14: true}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || step != traceStep || !targetLayers[layer] {
			return
		}
		cpuOps[opTraceKey{layer: layer, op: op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()
	_ = m.Generate(tok.Encode("Hello"), 1)

	checkQKNorm := func(layerIdx int, kind string, input, want []float32, weight []float32, numHeads, headDim int) {
		buf := gpu.NewDevBufFrom(append([]float32(nil), input...))
		wbuf := gpu.NewDevBufFrom(append([]float32(nil), weight...))
		if err := buf.ToGPU(); err != nil {
			t.Fatalf("buf.ToGPU: %v", err)
		}
		if err := wbuf.ToGPU(); err != nil {
			t.Fatalf("wbuf.ToGPU: %v", err)
		}
		for head := 0; head < numHeads; head++ {
			sl := buf.Slice(head*headDim, headDim)
			gpu.DevRMSNorm(sl, sl, wbuf, float32(m.Config.RMSNormEps))
			gpu.DevToBF16(sl, headDim)
		}
		gpu.Sync()
		got := append([]float32(nil), buf.Data()[:len(want)]...)
		maxAbs, meanAbs := diffStats(want, got)
		t.Logf("standalone layer %d %s_qknorm: maxAbs=%.6g meanAbs=%.6g", layerIdx, kind, maxAbs, meanAbs)
	}

	for _, layerIdx := range []int{0, 14} {
		layer := m.Layers[layerIdx]
		headDim := layer.HeadDimLocal
		if headDim == 0 {
			headDim = m.Config.HeadDim
		}
		checkQKNorm(layerIdx, "q", cpuOps[opTraceKey{layerIdx, "q"}], cpuOps[opTraceKey{layerIdx, "q_qknorm"}], layer.QNorm.Data(), m.Config.NumHeads, headDim)
		if layer.HasKV {
			checkQKNorm(layerIdx, "k", cpuOps[opTraceKey{layerIdx, "k"}], cpuOps[opTraceKey{layerIdx, "k_qknorm"}], layer.KNorm.Data(), m.Config.NumKVHeads, headDim)
		}
	}
}
