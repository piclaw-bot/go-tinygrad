package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/tensor"
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

func TestValidateShapeRejectsInvalidDims(t *testing.T) {
	if err := validateShape("bad", []int{-1, 2}, nil, 2); err == nil {
		t.Fatal("validateShape accepted negative expected dim")
	}
	if err := validateShape("bad", []int{2}, []int{-2}, 2); err == nil {
		t.Fatal("validateShape accepted negative actual dim")
	}
	if got := shapeProduct([]int{int(^uint(0) >> 1), 2}); got >= 0 {
		t.Fatalf("shapeProduct overflow=%d, want negative sentinel", got)
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

func TestGemma4MTPDrafterProjectionHelpers(t *testing.T) {
	d := &Gemma4MTPDrafter{
		Config: LlamaConfig{
			VocabSize:  4,
			HiddenSize: 2,
		},
		BackboneHiddenSize:      3,
		EmbedTokens:             tensor.FromFloat32([]float32{0, 1, 2, 3, 4, 5, 6, 7}, []int{4, 2}),
		MaskedEmbeddingOrdering: []int{3, 2, 1, 0},
		PreProjection: []float32{
			1, 2, 3, 10, 20, 30,
			-1, -2, -3, -10, -20, -30,
		},
		PostProjection: []float32{
			1, 0,
			0, 1,
			1, 1,
		},
	}

	emb := make([]float32, 2)
	if err := d.AssistantTokenEmbeddingInto(emb, 2); err != nil {
		t.Fatalf("AssistantTokenEmbeddingInto: %v", err)
	}
	if emb[0] != 4 || emb[1] != 5 {
		t.Fatalf("assistant embedding = %v, want [4 5]", emb)
	}
	if order, err := d.MaskedEmbeddingOrder(2); err != nil || order != 1 {
		t.Fatalf("MaskedEmbeddingOrder = %d, %v; want 1, nil", order, err)
	}

	pre := make([]float32, 2)
	if err := d.PreProjectInto(pre, []float32{1, 1, 1}, []float32{2, 2, 2}); err != nil {
		t.Fatalf("PreProjectInto: %v", err)
	}
	if pre[0] != 126 || pre[1] != -126 {
		t.Fatalf("pre projection = %v, want [126 -126]", pre)
	}

	post := make([]float32, 3)
	if err := d.PostProjectInto(post, []float32{3, 4}); err != nil {
		t.Fatalf("PostProjectInto: %v", err)
	}
	wantPost := []float32{3, 4, 7}
	for i := range wantPost {
		if post[i] != wantPost[i] {
			t.Fatalf("post projection = %v, want %v", post, wantPost)
		}
	}

	if err := d.PreProjectInto(make([]float32, 2), []float32{1, 2}, []float32{1, 2, 3}); err == nil {
		t.Fatal("PreProjectInto accepted short backbone token embedding")
	}
	if err := d.PostProjectInto(make([]float32, 2), []float32{1, 2}); err == nil {
		t.Fatal("PostProjectInto accepted short destination")
	}
	if err := d.AssistantTokenEmbeddingInto(make([]float32, 2), 4); err == nil {
		t.Fatal("AssistantTokenEmbeddingInto accepted out-of-range token")
	}
}

func TestGemma4MTPDrafterProjectionHelpersAreAliasSafe(t *testing.T) {
	d := &Gemma4MTPDrafter{
		Config:             LlamaConfig{HiddenSize: 2},
		BackboneHiddenSize: 2,
		PreProjection: []float32{
			1, 0, 0, 1,
			0, 1, 1, 0,
		},
		PostProjection: []float32{
			1, 0,
			0, 1,
		},
	}
	shared := []float32{10, 20}
	if err := d.PreProjectInto(shared, shared, []float32{1, 2}); err != nil {
		t.Fatalf("PreProjectInto alias: %v", err)
	}
	if !sameFloat32s(shared, []float32{12, 21}) {
		t.Fatalf("PreProjectInto alias result=%v want [12 21]", shared)
	}
	postShared := []float32{3, 4}
	if err := d.PostProjectInto(postShared, postShared); err != nil {
		t.Fatalf("PostProjectInto alias: %v", err)
	}
	if !sameFloat32s(postShared, []float32{3, 4}) {
		t.Fatalf("PostProjectInto alias result=%v want [3 4]", postShared)
	}
}

func TestMTPDrafterHelpersRejectOverflowingProducts(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	d := &Gemma4MTPDrafter{BackboneHiddenSize: maxInt/2 + 1}
	d.Config.HiddenSize = 1
	if err := d.PreProjectInto(make([]float32, 1), nil, nil); err == nil {
		t.Fatal("PreProjectInto accepted overflowing projection width")
	}
	d.BackboneHiddenSize = maxInt/2 + 1
	d.Config.HiddenSize = 2
	if err := d.PostProjectInto(nil, make([]float32, 2)); err == nil {
		t.Fatal("PostProjectInto accepted overflowing projection size")
	}
}

func TestMTPDrafterHelpersRejectShortBackingData(t *testing.T) {
	d := &Gemma4MTPDrafter{}
	d.Config.HiddenSize = 2
	d.Config.VocabSize = 2
	d.EmbedTokens = tensor.FromFloat32([]float32{1, 2, 3}, []int{3})
	buf := make([]float32, 2)
	if err := d.AssistantTokenEmbeddingInto(buf, 1); err == nil {
		t.Fatal("AssistantTokenEmbeddingInto accepted short embedding data")
	}

	d.BackboneHiddenSize = 2
	d.PreProjection = make([]float32, 7)
	if err := d.PreProjectInto(buf, []float32{1, 2}, []float32{3, 4}); err == nil {
		t.Fatal("PreProjectInto accepted short projection")
	}
	d.PreProjection = make([]float32, 8)
	if err := d.PreProjectInto(buf, []float32{1, 2}, []float32{3, 4}); err != nil {
		t.Fatalf("PreProjectInto valid: %v", err)
	}

	post := make([]float32, 2)
	d.PostProjection = make([]float32, 3)
	if err := d.PostProjectInto(post, []float32{1, 2}); err == nil {
		t.Fatal("PostProjectInto accepted short projection")
	}
}

func TestSimdDotBoundsShortInputs(t *testing.T) {
	if got := simdDot([]float32{2, 3}, []float32{4}); got != 8 {
		t.Fatalf("simdDot short b=%v want 8", got)
	}
	if got := simdDot([]float32{2}, nil); got != 0 {
		t.Fatalf("simdDot nil b=%v want 0", got)
	}
	long := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	if got := simdDot(long, []float32{1}); got != 1 {
		t.Fatalf("simdDot long mismatched=%v want 1", got)
	}
}
