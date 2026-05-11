package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma4Layer15MLPInputSensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 layer15 mlp_input sensitivity")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

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

	oNorm := append([]float32(nil), gpuO...)
	rmsNormInPlace(oNorm, m.Layers[15].PostNorm.Data(), float32(m.Config.RMSNormEps))
	recon := make([]float32, len(gpuHiddenIn))
	for i := range recon {
		recon[i] = gpuHiddenIn[i] + oNorm[i]
	}
	reconMLP := append([]float32(nil), recon...)
	rmsNormInPlace(reconMLP, m.Layers[15].PreFFNNorm.Data(), float32(m.Config.RMSNormEps))
	bf16Slice(reconMLP)
	maxAbs, meanAbs := diffStats(gpuMLPIn, reconMLP)
	t.Logf("layer15 cpu(preffn(residual+postnorm(gpuO))) vs gpu mlp_input: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	maxAbs, meanAbs = diffStats(cpuMLPIn, reconMLP)
	t.Logf("layer15 cpu(preffn(residual+postnorm(gpuO))) vs cpu mlp_input: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
