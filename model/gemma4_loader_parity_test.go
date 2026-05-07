package model

import (
	"os"
	"testing"
)

func TestGemma4LoaderParityForPerLayerArtifacts(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 loader parity trace")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	oldForce := ForceOnTheFly
	defer func() { ForceOnTheFly = oldForce }()

	ForceOnTheFly = false
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 cpu model: %v", err)
	}

	ForceOnTheFly = true
	mgpu, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 gpu model: %v", err)
	}

	check := func(name string, a, b []float32) {
		maxAbs, meanAbs := diffStats(a, b)
		t.Logf("loader parity %s: maxAbs=%.6g meanAbs=%.6g len=%d", name, maxAbs, meanAbs, len(a))
	}

	check("PerLayerModelProj", m.PerLayerModelProj, mgpu.PerLayerModelProj)
	check("PerLayerProjNorm", m.PerLayerProjNorm, mgpu.PerLayerProjNorm)
	if len(m.EmbedPerLayer) > 0 && len(mgpu.EmbedPerLayer) > 0 {
		n := m.Config.NumLayers * m.Config.HiddenPerLayer
		check("EmbedPerLayer[row0]", m.EmbedPerLayer[:n], mgpu.EmbedPerLayer[:n])
	}
	check("Layer0.InputNorm", m.Layers[0].InputNorm.Data(), mgpu.Layers[0].InputNorm.Data())
	if len(m.Layers) > 0 && len(mgpu.Layers) > 0 && len(m.Layers[0].PLIGate) > 0 {
		check("Layer0.PLIGate", m.Layers[0].PLIGate, mgpu.Layers[0].PLIGate)
		check("Layer0.PLIProj", m.Layers[0].PLIProj, mgpu.Layers[0].PLIProj)
		check("Layer0.PLIPostNorm", m.Layers[0].PLIPostNorm, mgpu.Layers[0].PLIPostNorm)
	}
}
