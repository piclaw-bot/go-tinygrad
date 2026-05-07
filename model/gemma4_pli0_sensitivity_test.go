package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func computePLI0CPUFromHidden(m *LlamaModel, tokID int, hidden []float32) []float32 {
	cfg := m.Config
	hpl := cfg.HiddenPerLayer
	nl := cfg.NumLayers
	totalDim := nl * hpl
	proj := make([]float32, totalDim)
	gemvNT(proj, hidden, m.PerLayerModelProj, cfg.HiddenSize, totalDim)
	for i := range proj {
		proj[i] *= m.PerLayerProjScale
	}
	for l := 0; l < nl; l++ {
		sl := proj[l*hpl : (l+1)*hpl]
		rmsNormInPlace(sl, m.PerLayerProjNorm, float32(cfg.RMSNormEps))
	}
	if m.EmbedPerLayer != nil && tokID < cfg.VocabPerLayer {
		embRow := m.EmbedPerLayer[tokID*totalDim : (tokID+1)*totalDim]
		for i := range proj {
			proj[i] = (proj[i] + embRow[i]*m.EmbedPerLayerScale) * m.PerLayerInputScale
		}
	}
	return append([]float32(nil), proj[:hpl]...)
}

func TestGemma4Layer0PerLayerInputSensitivity(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 per-layer input sensitivity trace")
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
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1
	traceTokID := wrapped[traceStep]

	cpuOps := map[opTraceKey][]float32{}
	gpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if step != traceStep || layer != 0 {
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
	t.Cleanup(g.Close)
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	cpuEmbed := cpuOps[opTraceKey{0, "embed_scaled"}]
	gpuEmbed := gpuOps[opTraceKey{0, "embed_scaled"}]
	cpuPLI0 := cpuOps[opTraceKey{0, "pli0_input"}]
	gpuPLI0 := gpuOps[opTraceKey{0, "pli0_input"}]
	if len(cpuEmbed) == 0 || len(gpuEmbed) == 0 || len(cpuPLI0) == 0 || len(gpuPLI0) == 0 {
		t.Fatal("missing embed/pli traces for layer 0")
	}

	embedMax, embedMean := diffStats(cpuEmbed, gpuEmbed)
	pliMax, pliMean := diffStats(cpuPLI0, gpuPLI0)
	t.Logf("layer0 embed_scaled diff: maxAbs=%.6g meanAbs=%.6g", embedMax, embedMean)
	t.Logf("layer0 pli0_input diff : maxAbs=%.6g meanAbs=%.6g", pliMax, pliMean)

	cpuOnGPUEmbed := computePLI0CPUFromHidden(m, traceTokID, gpuEmbed)
	reconMax, reconMean := diffStats(cpuOnGPUEmbed, gpuPLI0)
	cpuVsReconMax, cpuVsReconMean := diffStats(cpuPLI0, cpuOnGPUEmbed)
	t.Logf("layer0 cpu(pli0 from gpuEmbed) vs gpuPLI0: maxAbs=%.6g meanAbs=%.6g", reconMax, reconMean)
	t.Logf("layer0 cpuPLI0 vs cpu(pli0 from gpuEmbed): maxAbs=%.6g meanAbs=%.6g", cpuVsReconMax, cpuVsReconMean)
}
