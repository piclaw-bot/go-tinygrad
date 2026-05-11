package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4Layer15PostAttnModelBuffers(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 layer15 post-attn model-buffer test")
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

	gpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "gpu" || step != traceStep || layer != 15 {
			return
		}
		gpuOps[opTraceKey{layer: layer, op: op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()

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
	if len(gpuHiddenIn) == 0 || len(gpuO) == 0 || len(gpuMLPIn) == 0 {
		t.Fatal("missing layer15 hidden_in/o/mlp_input traces")
	}

	copy(g.residual.Data(), gpuHiddenIn)
	g.residual.MarkDirty()
	copy(g.oOut.Data(), gpuO)
	g.oOut.MarkDirty()
	gpu.DevAdd(g.hidden, g.residual, g.oOut)
	gpu.DevCopy(g.residual, g.hidden)
	gpu.DevRMSNorm(g.normed, g.hidden, g.Layers[15].PostNorm, float32(m.Config.RMSNormEps))
	gpu.Sync()
	got := append([]float32(nil), g.normed.Data()[:len(gpuMLPIn)]...)

	maxAbs, meanAbs := diffStats(gpuMLPIn, got)
	t.Logf("layer15 modelbuf postattn vs gpu mlp_input: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
