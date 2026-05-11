//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func gqaAttentionScoresScale(q, kCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	kvDim := numKVHeads * headDim
	headsPerKV := numHeads / numKVHeads
	out := make([]float32, numHeads*seqLen)
	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV
		for t := 0; t < seqLen; t++ {
			sum := float32(0)
			for d := 0; d < headDim; d++ {
				sum += q[head*headDim+d] * kCache[t*kvDim+kvHead*headDim+d]
			}
			out[head*seqLen+t] = sum * scale
		}
	}
	return out
}

func TestGemma4Layer0AttentionScoresKernelVsCPU(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 attention score trace")
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

	var finalQ []float32
	var kvK []float32
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || layer != 0 {
			return
		}
		switch op {
		case "k_attn":
			kvK = append(kvK, append([]float32(nil), vec...)...)
		case "q_attn":
			if step == traceStep {
				finalQ = append([]float32(nil), vec...)
			}
		}
	}
	defer func() { debugOpHook = nil }()
	_ = m.Generate(tok.Encode("Hello"), 1)

	if len(finalQ) == 0 || len(kvK) == 0 {
		t.Fatalf("missing traces: q=%d k=%d", len(finalQ), len(kvK))
	}

	headDim := m.Layers[0].HeadDimLocal
	if headDim == 0 {
		headDim = m.Config.HeadDim
	}
	numHeads := m.Config.NumHeads
	numKVHeads := m.Config.NumKVHeads
	seqLen := traceStep + 1
	want := gqaAttentionScoresScale(finalQ, kvK, seqLen, numHeads, numKVHeads, headDim, 1.0)

	qBuf := gpu.NewDevBufFrom(append([]float32(nil), finalQ...))
	kBuf := gpu.NewDevBufFrom(append([]float32(nil), kvK...))
	outBuf := gpu.NewDevBuf(len(want))
	if err := qBuf.ToGPU(); err != nil {
		t.Fatalf("qBuf.ToGPU: %v", err)
	}
	if err := kBuf.ToGPU(); err != nil {
		t.Fatalf("kBuf.ToGPU: %v", err)
	}
	if err := outBuf.ToGPU(); err != nil {
		t.Fatalf("outBuf.ToGPU: %v", err)
	}
	if !gpu.DevAttentionScores(outBuf, qBuf, kBuf, seqLen, numHeads, numKVHeads, headDim, 1.0) {
		t.Fatal("DevAttentionScores unavailable")
	}
	gpu.Sync()
	got := append([]float32(nil), outBuf.Data()[:len(want)]...)

	overallMax, overallMean := diffStats(want, got)
	t.Logf("layer0 attention scores kernel vs cpu: maxAbs=%.6g meanAbs=%.6g", overallMax, overallMean)
	if overallMax > 1e-4 {
		t.Fatalf("expected close attention score match, got maxAbs=%.6g meanAbs=%.6g", overallMax, overallMean)
	}
	for head := 0; head < numHeads; head++ {
		start := head * seqLen
		end := start + seqLen
		maxAbs, meanAbs := diffStats(want[start:end], got[start:end])
		t.Logf("layer0 attention scores head %d: maxAbs=%.6g meanAbs=%.6g", head, maxAbs, meanAbs)
	}
}
