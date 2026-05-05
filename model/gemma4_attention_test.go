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
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || layer != 0 {
			return
		}
		switch op {
		case "k":
			kvK = append(kvK, append([]float32(nil), vec...)...)
		case "v":
			kvV = append(kvV, append([]float32(nil), vec...)...)
		case "q":
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

	if len(finalQ) == 0 || len(finalAttn) == 0 || len(kvK) == 0 || len(kvV) == 0 {
		t.Fatalf("missing traces: q=%d attn=%d k=%d v=%d", len(finalQ), len(finalAttn), len(kvK), len(kvV))
	}

	headDim := m.Layers[0].HeadDimLocal
	if headDim == 0 {
		headDim = m.Config.HeadDim
	}
	numHeads := m.Config.NumHeads
	numKVHeads := m.Config.NumKVHeads
	seqLen := traceStep + 1

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
	// Current known bug: the CUDA attention path diverges badly from CPU even when
	// fed CPU-captured q/k/v tensors for the same final prompt token.
	if maxAbs < 1e-3 {
		t.Fatalf("expected non-trivial attention divergence for current Gemma4 debug case, got maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	}
}
