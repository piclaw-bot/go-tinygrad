package config

import "encoding/json"

// QwenNativeMTPMetadata captures the small subset of Qwen3.5/Qwen3.6 config
// needed to route native MTP checkpoints before touching weight files.
type QwenNativeMTPMetadata struct {
	ModelType                 string   `json:"model_type"`
	Architecture              string   `json:"architecture,omitempty"`
	HiddenSize                int      `json:"hidden_size"`
	NumHiddenLayers           int      `json:"num_hidden_layers"`
	MTPNumHiddenLayers        int      `json:"mtp_num_hidden_layers"`
	MTPUseDedicatedEmbeddings bool     `json:"mtp_use_dedicated_embeddings"`
	LayerTypes                []string `json:"layer_types,omitempty"`
	HasNativeMTP              bool     `json:"has_native_mtp"`
	HasLinearAttention        bool     `json:"has_linear_attention"`
}

func ParseQwenNativeMTPMetadata(data []byte) (QwenNativeMTPMetadata, error) {
	var raw struct {
		ModelType     string   `json:"model_type"`
		Architectures []string `json:"architectures"`
		TextConfig    *struct {
			ModelType                 string   `json:"model_type"`
			HiddenSize                int      `json:"hidden_size"`
			NumHiddenLayers           int      `json:"num_hidden_layers"`
			MTPNumHiddenLayers        int      `json:"mtp_num_hidden_layers"`
			MTPUseDedicatedEmbeddings bool     `json:"mtp_use_dedicated_embeddings"`
			LayerTypes                []string `json:"layer_types"`
		} `json:"text_config"`
		HiddenSize                int      `json:"hidden_size"`
		NumHiddenLayers           int      `json:"num_hidden_layers"`
		MTPNumHiddenLayers        int      `json:"mtp_num_hidden_layers"`
		MTPUseDedicatedEmbeddings bool     `json:"mtp_use_dedicated_embeddings"`
		LayerTypes                []string `json:"layer_types"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return QwenNativeMTPMetadata{}, err
	}
	meta := QwenNativeMTPMetadata{ModelType: raw.ModelType}
	if len(raw.Architectures) > 0 {
		meta.Architecture = raw.Architectures[0]
	}
	if raw.TextConfig != nil {
		if raw.TextConfig.ModelType != "" {
			meta.ModelType = raw.TextConfig.ModelType
		}
		meta.HiddenSize = raw.TextConfig.HiddenSize
		meta.NumHiddenLayers = raw.TextConfig.NumHiddenLayers
		meta.MTPNumHiddenLayers = raw.TextConfig.MTPNumHiddenLayers
		meta.MTPUseDedicatedEmbeddings = raw.TextConfig.MTPUseDedicatedEmbeddings
		meta.LayerTypes = append([]string(nil), raw.TextConfig.LayerTypes...)
	} else {
		meta.HiddenSize = raw.HiddenSize
		meta.NumHiddenLayers = raw.NumHiddenLayers
		meta.MTPNumHiddenLayers = raw.MTPNumHiddenLayers
		meta.MTPUseDedicatedEmbeddings = raw.MTPUseDedicatedEmbeddings
		meta.LayerTypes = append([]string(nil), raw.LayerTypes...)
	}
	meta.HasNativeMTP = meta.MTPNumHiddenLayers > 0
	for _, lt := range meta.LayerTypes {
		if lt == "linear_attention" {
			meta.HasLinearAttention = true
			break
		}
	}
	return meta, nil
}

func IsQwenNativeMTPTensorName(name string) bool {
	if len(name) < 4 || name[:4] != "mtp." {
		return false
	}
	switch name {
	case "mtp.fc.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight":
		return true
	}
	return len(name) > len("mtp.layers.") && name[:len("mtp.layers.")] == "mtp.layers."
}
