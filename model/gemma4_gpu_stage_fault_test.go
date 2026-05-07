package model

import (
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4EarlyGPUStageSyncs(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 GPU stage sync trace")
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
		t.Fatalf("load gemma4 gpu model: %v", err)
	}
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	t.Cleanup(g.Close)

	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceTokID := wrapped[len(wrapped)-1]
	cfg := m.Config
	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	layer0 := &g.Layers[0]
	cpuLayer0 := &m.Layers[0]
	layerHeadDim := cpuLayer0.HeadDimLocal
	if layerHeadDim == 0 {
		layerHeadDim = cfg.HeadDim
	}
	qDim := numHeads * layerHeadDim
	layerKVDim := numKVHeads * layerHeadDim
	hpl := cfg.HiddenPerLayer
	nl := cfg.NumLayers
	totalDim := nl * hpl

	check := func(stage string) {
		if err := gpu.SyncErr(); err != nil {
			t.Fatalf("%s: %v", stage, err)
		}
	}

	// Embedding + scale + bf16
	embData := m.EmbedTokens.Data()
	copy(g.hidden.Data(), embData[traceTokID*h:(traceTokID+1)*h])
	g.hidden.MarkDirty()
	hd := g.hidden.Data()
	scale := float32(math.Sqrt(float64(h)))
	for i := range hd {
		hd[i] *= scale
	}
	g.hidden.MarkDirty()
	gpu.DevToBF16(g.hidden, h)
	check("embed_scaled")

	// Model-level per-layer projection
	gpu.DevGemv(g.perLayerProjBuf, g.hidden, g.perLayerModelProj, totalDim, h)
	check("pli_proj_raw")
	gpu.DevScale(g.perLayerProjBuf, g.perLayerProjBuf, m.PerLayerProjScale)
	check("pli_proj_scaled")
	for ll := 0; ll < nl; ll++ {
		sl := g.perLayerProjBuf.Slice(ll*hpl, hpl)
		gpu.DevRMSNorm(sl, sl, g.perLayerProjNorm, float32(cfg.RMSNormEps))
		if err := gpu.SyncErr(); err != nil {
			t.Fatalf("pli_proj_norm[%d]: %v", ll, err)
		}
	}
	g.perLayerProjBuf.MarkOnGPU()
	if m.EmbedPerLayer != nil && traceTokID < cfg.VocabPerLayer {
		embRow := m.EmbedPerLayer[traceTokID*totalDim : (traceTokID+1)*totalDim]
		copy(g.perLayerEmbedBuf.Data(), embRow)
		g.perLayerEmbedBuf.MarkDirty()
		gpu.DevScale(g.perLayerEmbedBuf, g.perLayerEmbedBuf, m.EmbedPerLayerScale)
		check("pli_embed_scaled")
		gpu.DevAdd(g.perLayerProjBuf, g.perLayerProjBuf, g.perLayerEmbedBuf)
		check("pli_add")
		gpu.DevScale(g.perLayerProjBuf, g.perLayerProjBuf, m.PerLayerInputScale)
		check("pli_input_scaled")
	}

	// Layer 0 entry
	gpu.DevCopy(g.residual, g.hidden)
	check("layer0_residual_copy")
	gpu.DevRMSNorm(g.normed, g.hidden, layer0.InputNorm, float32(cfg.RMSNormEps))
	check("layer0_inputnorm")
	gpu.DevToBF16(g.normed, h)
	check("layer0_inputnorm_bf16")

	if layer0.QWmg != nil {
		gpu.GemvMLXDirect(g.q, g.normed, layer0.QWmg)
		check("layer0_q_proj")
		gpu.GemvMLXDirect(g.k, g.normed, layer0.KWmg)
		check("layer0_k_proj")
		gpu.GemvMLXDirect(g.v, g.normed, layer0.VWmg)
		check("layer0_v_proj")
	} else {
		t.Skip("layer0 not using MLX quantized GPU weights")
	}
	gpu.DevToBF16(g.q, qDim)
	gpu.DevToBF16(g.k, layerKVDim)
	gpu.DevToBF16(g.v, layerKVDim)
	check("layer0_qkv_bf16")
}
