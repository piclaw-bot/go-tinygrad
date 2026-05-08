package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4QuantizedLayer0PLISensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized layer0 PLI sensitivity")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !gpu.Available() {
		t.Skip("GPU not available")
	}
	t.Cleanup(gpu.Shutdown)

	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized model: %v", err)
	}
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	gpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "gpu" || step != traceStep || layer != 0 {
			return
		}
		if op != "hidden_post_ffn" && op != "hidden_post_pli" {
			return
		}
		gpuOps[opTraceKey{layer, op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()

	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	hiddenPostFFN := gpuOps[opTraceKey{0, "hidden_post_ffn"}]
	hiddenPostPLI := gpuOps[opTraceKey{0, "hidden_post_pli"}]
	if len(hiddenPostFFN) == 0 || len(hiddenPostPLI) == 0 {
		t.Fatal("missing layer0 hidden_post_ffn/hidden_post_pli traces")
	}

	perLayerInputs := gemma4PerLayerInputsForWrappedStep(m, wrapped, traceStep)
	hpl := m.Config.HiddenPerLayer
	gpuPLI := append([]float32(nil), g.perLayerProjBuf.Data()[:hpl]...)
	cpuPLI := append([]float32(nil), perLayerInputs[0]...)
	maxAbs, meanAbs := diffStats(cpuPLI, gpuPLI)
	t.Logf("L0 cpu per-layer input vs gpu per-layer input: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	layer := m.Layers[0]
	gate2 := make([]float32, hpl)
	gemvNT(gate2, hiddenPostFFN, layer.PLIGate, len(hiddenPostFFN), hpl)
	for i := range gate2 {
		gate2[i] = geluTanh(gate2[i])
		gate2[i] *= gpuPLI[i]
	}
	proj2 := make([]float32, len(hiddenPostFFN))
	gemvNT(proj2, gate2, layer.PLIProj, hpl, len(hiddenPostFFN))
	rmsNormInPlace(proj2, layer.PLIPostNorm, float32(m.Config.RMSNormEps))
	cpuRecon := append([]float32(nil), hiddenPostFFN...)
	for i := range cpuRecon {
		cpuRecon[i] += proj2[i]
	}
	maxAbs, meanAbs = diffStats(hiddenPostPLI, cpuRecon)
	t.Logf("L0 cpu(PLI from gpu hidden_post_ffn) vs gpu hidden_post_pli: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	hidBuf := gpu.NewDevBufFrom(append([]float32(nil), hiddenPostFFN...))
	pliBuf := gpu.NewDevBufFrom(append([]float32(nil), gpuPLI...))
	gateBuf := gpu.NewDevBuf(hpl)
	projBuf := gpu.NewDevBuf(len(hiddenPostFFN))
	outBuf := gpu.NewDevBuf(len(hiddenPostFFN))
	defer hidBuf.Free()
	defer pliBuf.Free()
	defer gateBuf.Free()
	defer projBuf.Free()
	defer outBuf.Free()
	for _, b := range []*gpu.DevBuf{hidBuf, pliBuf, gateBuf, projBuf, outBuf} {
		if err := b.ToGPU(); err != nil {
			t.Fatalf("ToGPU: %v", err)
		}
	}
	gpu.DevGemv(gateBuf, hidBuf, g.Layers[0].PLIGate, hpl, len(hiddenPostFFN))
	gpu.DevGELUTanhMul(gateBuf, pliBuf, hpl)
	gpu.DevGemv(projBuf, gateBuf, g.Layers[0].PLIProj, len(hiddenPostFFN), hpl)
	gpu.DevRMSNorm(projBuf, projBuf, g.Layers[0].PLIPostNorm, float32(m.Config.RMSNormEps))
	gpu.DevAdd(outBuf, hidBuf, projBuf)
	gpu.Sync()
	gpuRecon := append([]float32(nil), outBuf.Data()[:len(hiddenPostPLI)]...)
	maxAbs, meanAbs = diffStats(hiddenPostPLI, gpuRecon)
	t.Logf("L0 fresh gpu PLI from gpu hidden_post_ffn vs gpu hidden_post_pli: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
}
