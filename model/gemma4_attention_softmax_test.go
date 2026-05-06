package model

import (
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func softmaxRowsCPU(scores []float32, nRows, seqLen int) []float32 {
	out := make([]float32, len(scores))
	for row := 0; row < nRows; row++ {
		start := row * seqLen
		end := start + seqLen
		mx := scores[start]
		for _, v := range scores[start+1 : end] {
			if v > mx {
				mx = v
			}
		}
		sum := float32(0)
		for i := start; i < end; i++ {
			out[i] = exp32(scores[i] - mx)
			sum += out[i]
		}
		inv := 1 / sum
		for i := start; i < end; i++ {
			out[i] *= inv
		}
	}
	return out
}

func attentionFromWeightsCPU(weights, vCache []float32, seqLen, numHeads, numKVHeads, headDim int) []float32 {
	kvDim := numKVHeads * headDim
	headsPerKV := numHeads / numKVHeads
	out := make([]float32, numHeads*headDim)
	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV
		row := weights[head*seqLen : (head+1)*seqLen]
		for d := 0; d < headDim; d++ {
			sum := float32(0)
			for t := 0; t < seqLen; t++ {
				sum += row[t] * vCache[t*kvDim+kvHead*headDim+d]
			}
			out[head*headDim+d] = sum
		}
	}
	return out
}

func exp32(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func TestGemma4Layer0AttentionSoftmaxKernelVsCPU(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 attention softmax trace")
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
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || layer != 0 {
			return
		}
		switch op {
		case "k_attn":
			kvK = append(kvK, append([]float32(nil), vec...)...)
		case "v_attn":
			kvV = append(kvV, append([]float32(nil), vec...)...)
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
	scores := gqaAttentionScoresScale(finalQ, kvK, seqLen, numHeads, numKVHeads, headDim, 1.0)
	wantWeights := softmaxRowsCPU(scores, numHeads, seqLen)

	scoreBuf := gpu.NewDevBufFrom(append([]float32(nil), scores...))
	weightBuf := gpu.NewDevBuf(len(wantWeights))
	if err := scoreBuf.ToGPU(); err != nil {
		t.Fatalf("scoreBuf.ToGPU: %v", err)
	}
	if err := weightBuf.ToGPU(); err != nil {
		t.Fatalf("weightBuf.ToGPU: %v", err)
	}
	if !gpu.DevSoftmaxRows(weightBuf, scoreBuf, numHeads, seqLen) {
		t.Fatal("DevSoftmaxRows unavailable")
	}
	gpu.Sync()
	gotWeights := append([]float32(nil), weightBuf.Data()[:len(wantWeights)]...)

	overallMax, overallMean := diffStats(wantWeights, gotWeights)
	t.Logf("layer0 attention softmax kernel vs cpu: maxAbs=%.6g meanAbs=%.6g", overallMax, overallMean)
	if overallMax > 1e-6 {
		t.Fatalf("expected close attention softmax match, got maxAbs=%.6g meanAbs=%.6g", overallMax, overallMean)
	}
	for head := 0; head < numHeads; head++ {
		start := head * seqLen
		end := start + seqLen
		maxAbs, meanAbs := diffStats(wantWeights[start:end], gotWeights[start:end])
		t.Logf("layer0 attention softmax head %d: maxAbs=%.6g meanAbs=%.6g", head, maxAbs, meanAbs)
	}

	attnFromGPUWeights := attentionFromWeightsCPU(gotWeights, kvV, seqLen, numHeads, numKVHeads, headDim)
	maxAbs, meanAbs := diffStats(finalAttn, attnFromGPUWeights)
	t.Logf("layer0 attention from gpu softmax weights + cpu V-sum vs cpu: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	if maxAbs > 1e-6 {
		t.Fatalf("expected close attention reconstruction from gpu softmax weights, got maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	}
}
