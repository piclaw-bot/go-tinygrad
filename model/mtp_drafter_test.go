package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadGemma4MTPDrafterLocalAsset(t *testing.T) {
	dir := filepath.Join("..", "models", "gemma4-e2b-mtp-drafter")
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Skipf("local Gemma4 MTP drafter asset not available: %v", err)
	}

	d, err := LoadGemma4MTPDrafter(dir)
	if err != nil {
		t.Fatalf("LoadGemma4MTPDrafter: %v", err)
	}

	if d.Config.ModelType != "gemma4_text" {
		t.Fatalf("nested model_type = %q, want gemma4_text", d.Config.ModelType)
	}
	if d.BackboneHiddenSize != 1536 {
		t.Fatalf("BackboneHiddenSize = %d, want 1536", d.BackboneHiddenSize)
	}
	if d.Config.HiddenSize != 256 || d.Config.NumLayers != 4 {
		t.Fatalf("unexpected drafter config: hidden=%d layers=%d", d.Config.HiddenSize, d.Config.NumLayers)
	}
	if got, want := len(d.PreProjection), 256*3072; got != want {
		t.Fatalf("PreProjection len = %d, want %d", got, want)
	}
	if got, want := len(d.PostProjection), 1536*256; got != want {
		t.Fatalf("PostProjection len = %d, want %d", got, want)
	}
	if got, want := d.EmbedTokens.Shape(), []int{262144, 256}; got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("EmbedTokens shape = %v, want %v", got, want)
	}
	if got, want := d.MaskedEmbeddingCentroids.Shape(), []int{2048, 256}; got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("MaskedEmbeddingCentroids shape = %v, want %v", got, want)
	}
	if got, want := len(d.MaskedEmbeddingOrdering), 262144; got != want {
		t.Fatalf("MaskedEmbeddingOrdering len = %d, want %d", got, want)
	}

	for i, layer := range d.Layers {
		headDim := 256
		qDim := 1024
		if i == 3 {
			headDim = 512
			qDim = 2048
		}
		if layer.HeadDimLocal != headDim {
			t.Fatalf("layer %d HeadDimLocal = %d, want %d", i, layer.HeadDimLocal, headDim)
		}
		if got, want := len(layer.QW), qDim*256; got != want {
			t.Fatalf("layer %d QW len = %d, want %d", i, got, want)
		}
		if got, want := len(layer.OW), 256*qDim; got != want {
			t.Fatalf("layer %d OW len = %d, want %d", i, got, want)
		}
		if got, want := layer.QNorm.Shape()[0], headDim; got != want {
			t.Fatalf("layer %d QNorm shape = %v, want [%d]", i, layer.QNorm.Shape(), headDim)
		}
		if layer.LayerScalar == 0 {
			t.Fatalf("layer %d LayerScalar is zero", i)
		}
	}
}
