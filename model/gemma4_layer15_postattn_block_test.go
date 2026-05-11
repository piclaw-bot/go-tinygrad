package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4Layer15PostAttnBlockIsolation(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 layer15 post-attn isolation")
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
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	gpuHiddenIn := gpuOps[opTraceKey{15, "hidden_in"}]
	gpuO := gpuOps[opTraceKey{15, "o"}]
	gpuMLPIn := gpuOps[opTraceKey{15, "mlp_input"}]
	cpuMLPIn := cpuOps[opTraceKey{15, "mlp_input"}]
	if len(gpuHiddenIn) == 0 || len(gpuO) == 0 || len(gpuMLPIn) == 0 || len(cpuMLPIn) == 0 {
		t.Fatal("missing layer15 hidden_in/o/mlp_input traces")
	}

	// CPU recompute from GPU-captured inputs using the real Gemma4 path:
	// o_norm = PostNorm(o), hidden = residual + o_norm, mlp_input = PreFFNNorm(hidden)
	cpuONorm := append([]float32(nil), gpuO...)
	rmsNormInPlace(cpuONorm, m.Layers[15].PostNorm.Data(), float32(m.Config.RMSNormEps))
	cpuHidden := make([]float32, len(gpuHiddenIn))
	for i := range cpuHidden {
		cpuHidden[i] = gpuHiddenIn[i] + cpuONorm[i]
	}
	cpuRecon := append([]float32(nil), cpuHidden...)
	rmsNormInPlace(cpuRecon, m.Layers[15].PreFFNNorm.Data(), float32(m.Config.RMSNormEps))
	bf16Slice(cpuRecon)

	// GPU recompute on fresh buffers using the same captured inputs and the same norm sequence.
	hidBuf := gpu.NewDevBufFrom(append([]float32(nil), gpuHiddenIn...))
	oBuf := gpu.NewDevBufFrom(append([]float32(nil), gpuO...))
	tmpBuf := gpu.NewDevBuf(len(gpuHiddenIn))
	outBuf := gpu.NewDevBuf(len(gpuHiddenIn))
	postBuf := g.Layers[15].PostNorm
	preBuf := g.Layers[15].PreFFNNorm
	defer hidBuf.Free()
	defer oBuf.Free()
	defer tmpBuf.Free()
	defer outBuf.Free()
	if err := hidBuf.ToGPU(); err != nil {
		t.Fatalf("hidBuf.ToGPU: %v", err)
	}
	if err := oBuf.ToGPU(); err != nil {
		t.Fatalf("oBuf.ToGPU: %v", err)
	}
	if err := tmpBuf.ToGPU(); err != nil {
		t.Fatalf("tmpBuf.ToGPU: %v", err)
	}
	if err := outBuf.ToGPU(); err != nil {
		t.Fatalf("outBuf.ToGPU: %v", err)
	}
	gpu.DevRMSNorm(tmpBuf, oBuf, postBuf, float32(m.Config.RMSNormEps))
	gpu.DevAdd(outBuf, hidBuf, tmpBuf)
	gpu.DevRMSNorm(outBuf, outBuf, preBuf, float32(m.Config.RMSNormEps))
	gpu.DevToBF16(outBuf, len(gpuHiddenIn))
	gpu.Sync()
	gpuRecon := append([]float32(nil), outBuf.Data()[:len(gpuMLPIn)]...)

	maxAbs, meanAbs := diffStats(cpuRecon, gpuRecon)
	t.Logf("layer15 fresh gpu(preffn(residual+postnorm(gpuO))) vs cpu recompute: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	maxAbs, meanAbs = diffStats(gpuMLPIn, gpuRecon)
	t.Logf("layer15 fresh gpu(preffn(residual+postnorm(gpuO))) vs gpu mlp_input: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	maxAbs, meanAbs = diffStats(cpuMLPIn, gpuRecon)
	t.Logf("layer15 fresh gpu(preffn(residual+postnorm(gpuO))) vs cpu mlp_input: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
