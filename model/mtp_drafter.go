package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/rcarmo/go-pherence/safetensors"
	"github.com/rcarmo/go-pherence/tensor"
)

// Gemma4MTPDrafter holds the Gemma4 assistant/MTP drafter weights.
//
// The assistant is not a normal Gemma4 decoder: its attention blocks are
// q-only and consume K/V from the main model. It also has pre/post projection
// tensors for activation-conditioned drafting.
type Gemma4MTPDrafter struct {
	Config LlamaConfig

	BackboneHiddenSize int
	NumCentroids       int
	UseOrderedEmbeds   bool

	EmbedTokens              *tensor.Tensor // [vocab, hidden]
	MaskedEmbeddingCentroids *tensor.Tensor // [numCentroids, hidden]
	MaskedEmbeddingOrdering  []int          // [vocab]

	PreProjection  []float32 // [hidden, 2*backboneHidden]
	PostProjection []float32 // [backboneHidden, hidden]

	Norm   *tensor.Tensor // [hidden]
	Layers []Gemma4MTPDrafterLayer
}

// Gemma4MTPDrafterLayer is one q-only assistant layer.
type Gemma4MTPDrafterLayer struct {
	InputNorm     *tensor.Tensor
	PostNorm      *tensor.Tensor
	PreFFNNorm    *tensor.Tensor
	PostFFNNorm   *tensor.Tensor
	LayerScalar   float32
	HeadDimLocal  int
	KVSourceLayer int

	QW    []float32 // [numHeads*headDim, hidden]
	QNorm *tensor.Tensor
	OW    []float32 // [hidden, numHeads*headDim]

	GateW []float32 // [intermediate, hidden]
	UpW   []float32 // [intermediate, hidden]
	DownW []float32 // [hidden, intermediate]
}

type gemma4AssistantConfig struct {
	Architectures        []string    `json:"architectures"`
	BackboneHiddenSize   int         `json:"backbone_hidden_size"`
	ModelType            string      `json:"model_type"`
	NumCentroids         int         `json:"num_centroids"`
	TextConfig           LlamaConfig `json:"text_config"`
	TieWordEmbeddings    bool        `json:"tie_word_embeddings"`
	UseOrderedEmbeddings bool        `json:"use_ordered_embeddings"`
}

type drafterSafetensors interface {
	GetFloat32(name string) ([]float32, []int, error)
	GetRaw(name string) ([]byte, string, []int, error)
	Close() error
}

// LoadGemma4MTPDrafter loads a local Gemma4 assistant drafter asset.
func LoadGemma4MTPDrafter(dir string) (*Gemma4MTPDrafter, error) {
	cfgData, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var acfg gemma4AssistantConfig
	if err := json.Unmarshal(cfgData, &acfg); err != nil {
		return nil, err
	}
	if acfg.ModelType != "gemma4_assistant" {
		return nil, fmt.Errorf("expected gemma4_assistant model_type, got %q", acfg.ModelType)
	}

	cfg := acfg.TextConfig
	if cfg.HiddenSize == 0 || cfg.NumLayers == 0 {
		return nil, fmt.Errorf("invalid nested text_config: hidden=%d layers=%d", cfg.HiddenSize, cfg.NumLayers)
	}
	if cfg.VocabSize == 0 {
		return nil, fmt.Errorf("invalid nested text_config: vocab_size=0")
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-6
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	if cfg.HiddenAct == "" {
		cfg.HiddenAct = "gelu_pytorch_tanh"
	}

	f, err := openDrafterSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	loadTensor := func(name string, shape []int) (*tensor.Tensor, error) {
		data, actualShape, err := f.GetFloat32(name)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", name, err)
		}
		shape, err = preferActualShape(shape, actualShape, len(data))
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", name, err)
		}
		return tensor.FromFloat32(data, shape), nil
	}
	loadData := func(name string, shape []int) ([]float32, error) {
		data, actualShape, err := f.GetFloat32(name)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", name, err)
		}
		if _, err := preferActualShape(shape, actualShape, len(data)); err != nil {
			return nil, fmt.Errorf("load %s: %w", name, err)
		}
		return data, nil
	}

	h := cfg.HiddenSize
	d := &Gemma4MTPDrafter{
		Config:             cfg,
		BackboneHiddenSize: acfg.BackboneHiddenSize,
		NumCentroids:       acfg.NumCentroids,
		UseOrderedEmbeds:   acfg.UseOrderedEmbeddings,
		Layers:             make([]Gemma4MTPDrafterLayer, cfg.NumLayers),
	}

	if d.BackboneHiddenSize == 0 {
		return nil, fmt.Errorf("backbone_hidden_size=0")
	}
	if d.NumCentroids == 0 {
		d.NumCentroids = 2048
	}

	if d.EmbedTokens, err = loadTensor("model.embed_tokens.weight", []int{cfg.VocabSize, h}); err != nil {
		return nil, err
	}
	if d.MaskedEmbeddingCentroids, err = loadTensor("masked_embedding.centroids.weight", []int{d.NumCentroids, h}); err != nil {
		return nil, err
	}
	if d.MaskedEmbeddingOrdering, err = loadIntTensor(f, "masked_embedding.token_ordering", cfg.VocabSize); err != nil {
		return nil, err
	}
	if d.PreProjection, err = loadData("pre_projection.weight", []int{h, 2 * d.BackboneHiddenSize}); err != nil {
		return nil, err
	}
	if d.PostProjection, err = loadData("post_projection.weight", []int{d.BackboneHiddenSize, h}); err != nil {
		return nil, err
	}
	if d.Norm, err = loadTensor("model.norm.weight", []int{h}); err != nil {
		return nil, err
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := fmt.Sprintf("model.layers.%d", l)
		headDim := cfg.HeadDim
		if l < len(cfg.LayerTypes) && cfg.LayerTypes[l] == "full_attention" && cfg.GlobalHeadDim > 0 {
			headDim = cfg.GlobalHeadDim
		}
		qDim := cfg.NumHeads * headDim

		layer := Gemma4MTPDrafterLayer{
			LayerScalar:   1,
			HeadDimLocal:  headDim,
			KVSourceLayer: l,
		}
		if layer.InputNorm, err = loadTensor(p+".input_layernorm.weight", []int{h}); err != nil {
			return nil, err
		}
		if layer.PostNorm, err = loadTensor(p+".post_attention_layernorm.weight", []int{h}); err != nil {
			return nil, err
		}
		if layer.PreFFNNorm, err = loadTensor(p+".pre_feedforward_layernorm.weight", []int{h}); err != nil {
			return nil, err
		}
		if layer.PostFFNNorm, err = loadTensor(p+".post_feedforward_layernorm.weight", []int{h}); err != nil {
			return nil, err
		}
		if scalar, err := loadData(p+".layer_scalar", []int{1}); err == nil && len(scalar) > 0 {
			layer.LayerScalar = scalar[0]
		} else if err != nil {
			return nil, err
		}

		if layer.QW, err = loadData(p+".self_attn.q_proj.weight", []int{qDim, h}); err != nil {
			return nil, err
		}
		if layer.QNorm, err = loadTensor(p+".self_attn.q_norm.weight", []int{headDim}); err != nil {
			return nil, err
		}
		if layer.OW, err = loadData(p+".self_attn.o_proj.weight", []int{h, qDim}); err != nil {
			return nil, err
		}

		if layer.GateW, err = loadData(p+".mlp.gate_proj.weight", []int{cfg.Intermediate, h}); err != nil {
			return nil, err
		}
		if layer.UpW, err = loadData(p+".mlp.up_proj.weight", []int{cfg.Intermediate, h}); err != nil {
			return nil, err
		}
		if layer.DownW, err = loadData(p+".mlp.down_proj.weight", []int{h, cfg.Intermediate}); err != nil {
			return nil, err
		}

		d.Layers[l] = layer
	}

	return d, nil
}

func openDrafterSafetensors(dir string) (drafterSafetensors, error) {
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(indexPath); err == nil {
		return safetensors.OpenSharded(indexPath)
	}
	return safetensors.Open(filepath.Join(dir, "model.safetensors"))
}

func preferActualShape(expected, actual []int, n int) ([]int, error) {
	if len(actual) > 0 && shapeProduct(actual) == n {
		return actual, nil
	}
	if shapeProduct(expected) != n {
		return nil, fmt.Errorf("shape mismatch: expected %v (%d elems), actual %v (%d elems)", expected, shapeProduct(expected), actual, n)
	}
	return expected, nil
}

func shapeProduct(shape []int) int {
	prod := 1
	for _, dim := range shape {
		prod *= dim
	}
	return prod
}

func loadIntTensor(f drafterSafetensors, name string, expectedLen int) ([]int, error) {
	raw, dtype, shape, err := f.GetRaw(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	if shapeProduct(shape) != expectedLen {
		return nil, fmt.Errorf("load %s: expected %d elems, shape %v has %d", name, expectedLen, shape, shapeProduct(shape))
	}
	out := make([]int, expectedLen)
	switch dtype {
	case "I64":
		if len(raw) != expectedLen*8 {
			return nil, fmt.Errorf("load %s: raw size %d does not match I64 len %d", name, len(raw), expectedLen)
		}
		for i := range out {
			out[i] = int(int64(binary.LittleEndian.Uint64(raw[i*8:])))
		}
	case "I32":
		if len(raw) != expectedLen*4 {
			return nil, fmt.Errorf("load %s: raw size %d does not match I32 len %d", name, len(raw), expectedLen)
		}
		for i := range out {
			out[i] = int(int32(binary.LittleEndian.Uint32(raw[i*4:])))
		}
	default:
		return nil, fmt.Errorf("load %s: unsupported integer dtype %s", name, dtype)
	}
	return out, nil
}
