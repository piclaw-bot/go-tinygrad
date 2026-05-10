package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadGemma4MTPDrafterLocalAsset(t *testing.T) {
	dir := filepath.Join("..", "models", "gemma4-e2b-mtp-drafter")
	if _, errSingle := os.Stat(filepath.Join(dir, "model.safetensors")); errSingle != nil {
		if _, errSharded := os.Stat(filepath.Join(dir, "model.safetensors.index.json")); errSharded != nil {
			t.Skipf("local Gemma4 MTP drafter asset not available: single=%v sharded=%v", errSingle, errSharded)
		}
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
		headDim := d.Config.HeadDim
		if d.Config.LayerTypes[i] == "full_attention" && d.Config.GlobalHeadDim > 0 {
			headDim = d.Config.GlobalHeadDim
		}
		qDim := d.Config.NumHeads * headDim
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
		if layer.KVSourceLayer != -1 {
			t.Fatalf("layer %d KVSourceLayer = %d, want -1 for external KV", i, layer.KVSourceLayer)
		}
		if layer.LayerScalar == 0 {
			t.Fatalf("layer %d LayerScalar is zero", i)
		}
	}
}

func TestValidateShapeRejectsTransposedShape(t *testing.T) {
	if err := validateShape("weight", []int{256, 3072}, []int{3072, 256}, 256*3072); err == nil {
		t.Fatal("validateShape accepted a transposed shape with the same element count")
	}
	if err := validateShape("weight", []int{256, 3072}, []int{256, 3072}, 256*3072); err != nil {
		t.Fatalf("validateShape rejected exact shape: %v", err)
	}
}

func TestLoadGemma4MTPDrafterRejectsMalformedConfigBeforeWeights(t *testing.T) {
	dir := t.TempDir()
	cfg := `{
		"model_type":"gemma4_assistant",
		"backbone_hidden_size":1536,
		"num_centroids":2048,
		"text_config":{
			"model_type":"gemma4_text",
			"vocab_size":262144,
			"hidden_size":256,
			"intermediate_size":2048,
			"num_hidden_layers":4,
			"num_attention_heads":0
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadGemma4MTPDrafter(dir)
	if err == nil || !strings.Contains(err.Error(), "num_attention_heads=0") {
		t.Fatalf("LoadGemma4MTPDrafter err = %v, want num_attention_heads validation", err)
	}
}
