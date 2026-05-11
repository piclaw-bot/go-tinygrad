//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4StandalonePerLayerProjectionPipelineVsCPU(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 per-layer projection trace")
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
	traceTokID := wrapped[traceStep]

	cpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || step != traceStep || layer != 0 {
			return
		}
		cpuOps[opTraceKey{layer: layer, op: op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()
	_ = m.Generate(tok.Encode("Hello"), 1)

	cpuEmbed := cpuOps[opTraceKey{0, "embed_scaled"}]
	if len(cpuEmbed) == 0 {
		t.Fatal("missing cpu embed_scaled trace")
	}

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

	cfg := m.Config
	h := cfg.HiddenSize
	hpl := cfg.HiddenPerLayer
	nl := cfg.NumLayers
	totalDim := nl * hpl

	projCPU := make([]float32, totalDim)
	gemvNT(projCPU, cpuEmbed, m.PerLayerModelProj, h, totalDim)

	inBuf := gpu.NewDevBufFrom(append([]float32(nil), cpuEmbed...))
	projBuf := gpu.NewDevBuf(totalDim)
	if err := inBuf.ToGPU(); err != nil {
		t.Fatalf("inBuf.ToGPU: %v", err)
	}
	if err := projBuf.ToGPU(); err != nil {
		t.Fatalf("projBuf.ToGPU: %v", err)
	}
	gpu.DevGemv(projBuf, inBuf, g.perLayerModelProj, totalDim, h)
	gpu.Sync()
	projGPU := append([]float32(nil), projBuf.Data()[:totalDim]...)
	maxAbs, meanAbs := diffStats(projCPU, projGPU)
	t.Logf("per-layer proj raw: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	for i := range projCPU {
		projCPU[i] *= m.PerLayerProjScale
	}
	gpu.DevScale(projBuf, projBuf, m.PerLayerProjScale)
	gpu.Sync()
	projGPU = append([]float32(nil), projBuf.Data()[:totalDim]...)
	maxAbs, meanAbs = diffStats(projCPU, projGPU)
	t.Logf("per-layer proj scaled: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	for l := 0; l < nl; l++ {
		sl := projCPU[l*hpl : (l+1)*hpl]
		rmsNormInPlace(sl, m.PerLayerProjNorm, float32(cfg.RMSNormEps))
	}
	for l := 0; l < nl; l++ {
		sl := projBuf.Slice(l*hpl, hpl)
		gpu.DevRMSNorm(sl, sl, g.perLayerProjNorm, float32(cfg.RMSNormEps))
	}
	projBuf.MarkOnGPU()
	gpu.Sync()
	projGPU = append([]float32(nil), projBuf.Data()[:totalDim]...)
	maxAbs, meanAbs = diffStats(projCPU, projGPU)
	t.Logf("per-layer proj normed: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)

	if m.EmbedPerLayer != nil && traceTokID < cfg.VocabPerLayer {
		embRow := m.EmbedPerLayer[traceTokID*totalDim : (traceTokID+1)*totalDim]
		for i := range projCPU {
			projCPU[i] = (projCPU[i] + embRow[i]*m.EmbedPerLayerScale) * m.PerLayerInputScale
		}
		embBuf := gpu.NewDevBufFrom(append([]float32(nil), embRow...))
		if err := embBuf.ToGPU(); err != nil {
			t.Fatalf("embBuf.ToGPU: %v", err)
		}
		gpu.DevScale(embBuf, embBuf, m.EmbedPerLayerScale)
		gpu.DevAdd(projBuf, projBuf, embBuf)
		gpu.DevScale(projBuf, projBuf, m.PerLayerInputScale)
		projBuf.MarkOnGPU()
		gpu.Sync()
		projGPU = append([]float32(nil), projBuf.Data()[:totalDim]...)
		maxAbs, meanAbs = diffStats(projCPU, projGPU)
		t.Logf("per-layer proj final: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
		max0, mean0 := diffStats(projCPU[:hpl], projGPU[:hpl])
		t.Logf("per-layer proj final slice0: maxAbs=%.6g meanAbs=%.6g", max0, mean0)
	}
}
