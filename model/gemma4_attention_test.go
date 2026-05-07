package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4Layer0AttentionKernelVsCPU(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 attention kernel trace")
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

	var finalQ, finalAttn []float32
	var kvK, kvV []float32
	kByStep := map[int][]float32{}
	vByStep := map[int][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || layer != 0 {
			return
		}
		switch op {
		case "k_attn":
			if _, ok := kByStep[step]; !ok {
				kByStep[step] = append([]float32(nil), vec...)
			}
		case "v_attn":
			if _, ok := vByStep[step]; !ok {
				vByStep[step] = append([]float32(nil), vec...)
			}
		case "q_attn":
			if step == traceStep {
				finalQ = append([]float32(nil), vec...)
			}
		case "attn":
			if step == traceStep {
				finalAttn = append([]float32(nil), vec...)
			}
		}
	}
	defer func() { debugOpHook = nil }()
	_ = m.Generate(tok.Encode("Hello"), 1)

	seqLen := traceStep + 1
	for step := 0; step < seqLen; step++ {
		if k, ok := kByStep[step]; ok {
			kvK = append(kvK, k...)
		}
		if v, ok := vByStep[step]; ok {
			kvV = append(kvV, v...)
		}
	}
	if len(finalQ) == 0 || len(finalAttn) == 0 || len(kvK) == 0 || len(kvV) == 0 {
		t.Fatalf("missing traces: q=%d attn=%d k=%d v=%d", len(finalQ), len(finalAttn), len(kvK), len(kvV))
	}

	headDim := m.Layers[0].HeadDimLocal
	if headDim == 0 {
		headDim = m.Config.HeadDim
	}
	numHeads := m.Config.NumHeads
	numKVHeads := m.Config.NumKVHeads

	qBuf := gpu.NewDevBufFrom(append([]float32(nil), finalQ...))
	kBuf := gpu.NewDevBufFrom(append([]float32(nil), kvK...))
	vBuf := gpu.NewDevBufFrom(append([]float32(nil), kvV...))
	outBuf := gpu.NewDevBuf(len(finalAttn))
	if err := qBuf.ToGPU(); err != nil {
		t.Fatalf("qBuf.ToGPU: %v", err)
	}
	if err := kBuf.ToGPU(); err != nil {
		t.Fatalf("kBuf.ToGPU: %v", err)
	}
	if err := vBuf.ToGPU(); err != nil {
		t.Fatalf("vBuf.ToGPU: %v", err)
	}
	if err := outBuf.ToGPU(); err != nil {
		t.Fatalf("outBuf.ToGPU: %v", err)
	}

	gpu.DevAttention(outBuf, qBuf, kBuf, vBuf, seqLen, numHeads, numKVHeads, headDim, 1.0)
	gpu.Sync()
	got := append([]float32(nil), outBuf.Data()[:len(finalAttn)]...)
	maxAbs, meanAbs := diffStats(finalAttn, got)
	t.Logf("layer0 attention kernel vs cpu: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	// With corrected post-VNorm/post-QKNorm/post-RoPE traces, the CUDA attention
	// kernel now matches CPU closely for the captured layer-0 Gemma4 case.
	if maxAbs > 1e-4 {
		t.Fatalf("expected close attention match, got maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	}
}
