package config

import (
	"encoding/json"
	"strconv"
)

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
	NumAttentionHeads         int      `json:"num_attention_heads,omitempty"`
	NumKeyValueHeads          int      `json:"num_key_value_heads,omitempty"`
	HeadDim                   int      `json:"head_dim,omitempty"`
	LinearConvKernelDim       int      `json:"linear_conv_kernel_dim,omitempty"`
	LinearKeyHeadDim          int      `json:"linear_key_head_dim,omitempty"`
	LinearNumKeyHeads         int      `json:"linear_num_key_heads,omitempty"`
	LinearNumValueHeads       int      `json:"linear_num_value_heads,omitempty"`
	LinearValueHeadDim        int      `json:"linear_value_head_dim,omitempty"`
	FullAttentionInterval     int      `json:"full_attention_interval,omitempty"`
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
			NumAttentionHeads         int      `json:"num_attention_heads"`
			NumKeyValueHeads          int      `json:"num_key_value_heads"`
			HeadDim                   int      `json:"head_dim"`
			LinearConvKernelDim       int      `json:"linear_conv_kernel_dim"`
			LinearKeyHeadDim          int      `json:"linear_key_head_dim"`
			LinearNumKeyHeads         int      `json:"linear_num_key_heads"`
			LinearNumValueHeads       int      `json:"linear_num_value_heads"`
			LinearValueHeadDim        int      `json:"linear_value_head_dim"`
			FullAttentionInterval     int      `json:"full_attention_interval"`
		} `json:"text_config"`
		HiddenSize                int      `json:"hidden_size"`
		NumHiddenLayers           int      `json:"num_hidden_layers"`
		MTPNumHiddenLayers        int      `json:"mtp_num_hidden_layers"`
		MTPUseDedicatedEmbeddings bool     `json:"mtp_use_dedicated_embeddings"`
		LayerTypes                []string `json:"layer_types"`
		NumAttentionHeads         int      `json:"num_attention_heads"`
		NumKeyValueHeads          int      `json:"num_key_value_heads"`
		HeadDim                   int      `json:"head_dim"`
		LinearConvKernelDim       int      `json:"linear_conv_kernel_dim"`
		LinearKeyHeadDim          int      `json:"linear_key_head_dim"`
		LinearNumKeyHeads         int      `json:"linear_num_key_heads"`
		LinearNumValueHeads       int      `json:"linear_num_value_heads"`
		LinearValueHeadDim        int      `json:"linear_value_head_dim"`
		FullAttentionInterval     int      `json:"full_attention_interval"`
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
		meta.NumAttentionHeads = raw.TextConfig.NumAttentionHeads
		meta.NumKeyValueHeads = raw.TextConfig.NumKeyValueHeads
		meta.HeadDim = raw.TextConfig.HeadDim
		meta.LinearConvKernelDim = raw.TextConfig.LinearConvKernelDim
		meta.LinearKeyHeadDim = raw.TextConfig.LinearKeyHeadDim
		meta.LinearNumKeyHeads = raw.TextConfig.LinearNumKeyHeads
		meta.LinearNumValueHeads = raw.TextConfig.LinearNumValueHeads
		meta.LinearValueHeadDim = raw.TextConfig.LinearValueHeadDim
		meta.FullAttentionInterval = raw.TextConfig.FullAttentionInterval
	} else {
		meta.HiddenSize = raw.HiddenSize
		meta.NumHiddenLayers = raw.NumHiddenLayers
		meta.MTPNumHiddenLayers = raw.MTPNumHiddenLayers
		meta.MTPUseDedicatedEmbeddings = raw.MTPUseDedicatedEmbeddings
		meta.LayerTypes = append([]string(nil), raw.LayerTypes...)
		meta.NumAttentionHeads = raw.NumAttentionHeads
		meta.NumKeyValueHeads = raw.NumKeyValueHeads
		meta.HeadDim = raw.HeadDim
		meta.LinearConvKernelDim = raw.LinearConvKernelDim
		meta.LinearKeyHeadDim = raw.LinearKeyHeadDim
		meta.LinearNumKeyHeads = raw.LinearNumKeyHeads
		meta.LinearNumValueHeads = raw.LinearNumValueHeads
		meta.LinearValueHeadDim = raw.LinearValueHeadDim
		meta.FullAttentionInterval = raw.FullAttentionInterval
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

func RequiredQwenNativeMTPTensors(numLayers int) []string {
	if numLayers <= 0 {
		return nil
	}
	base := []string{
		"mtp.fc.weight",
		"mtp.pre_fc_norm_embedding.weight",
		"mtp.pre_fc_norm_hidden.weight",
		"mtp.norm.weight",
	}
	perLayer := []string{
		"input_layernorm.weight",
		"self_attn.q_proj.weight",
		"self_attn.k_proj.weight",
		"self_attn.v_proj.weight",
		"self_attn.o_proj.weight",
		"self_attn.q_norm.weight",
		"self_attn.k_norm.weight",
		"post_attention_layernorm.weight",
		"mlp.gate_proj.weight",
		"mlp.up_proj.weight",
		"mlp.down_proj.weight",
	}
	out := append([]string(nil), base...)
	for i := 0; i < numLayers; i++ {
		for _, suffix := range perLayer {
			out = append(out, "mtp.layers."+strconv.Itoa(i)+"."+suffix)
		}
	}
	return out
}

func MissingQwenNativeMTPTensors(names []string, numLayers int) []string {
	required := RequiredQwenNativeMTPTensors(numLayers)
	if len(required) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		seen[name] = true
	}
	var missing []string
	for _, name := range required {
		if !seen[name] {
			missing = append(missing, name)
		}
	}
	return missing
}

func (m QwenNativeMTPMetadata) MainLayerCount() int {
	if m.NumHiddenLayers <= 0 {
		return 0
	}
	if m.MTPNumHiddenLayers <= 0 {
		return m.NumHiddenLayers
	}
	if m.MTPNumHiddenLayers >= m.NumHiddenLayers {
		return 0
	}
	return m.NumHiddenLayers - m.MTPNumHiddenLayers
}

type QwenNativeMTPLayerSummary struct {
	MainLayers      int `json:"main_layers"`
	MTPLayers       int `json:"mtp_layers"`
	LinearAttention int `json:"linear_attention"`
	FullAttention   int `json:"full_attention"`
}

func (m QwenNativeMTPMetadata) LayerSummary() QwenNativeMTPLayerSummary {
	main := m.MainLayerCount()
	s := QwenNativeMTPLayerSummary{MainLayers: main, MTPLayers: m.MTPNumHiddenLayers}
	for i := 0; i < main; i++ {
		if m.IsLinearAttentionLayer(i) {
			s.LinearAttention++
		} else if m.IsFullAttentionLayer(i) {
			s.FullAttention++
		}
	}
	return s
}

func (m QwenNativeMTPMetadata) IsMTPLayer(layer int) bool {
	main := m.MainLayerCount()
	return layer >= main && layer < m.NumHiddenLayers && m.MTPNumHiddenLayers > 0
}

func (m QwenNativeMTPMetadata) IsLinearAttentionLayer(layer int) bool {
	if layer < 0 || layer >= m.MainLayerCount() {
		return false
	}
	if layer < len(m.LayerTypes) {
		return m.LayerTypes[layer] == "linear_attention"
	}
	interval := m.FullAttentionInterval
	if interval <= 0 {
		interval = 4
	}
	return (layer+1)%interval != 0
}

func (m QwenNativeMTPMetadata) IsFullAttentionLayer(layer int) bool {
	if layer < 0 || layer >= m.MainLayerCount() {
		return false
	}
	if layer < len(m.LayerTypes) {
		return m.LayerTypes[layer] == "full_attention"
	}
	return !m.IsLinearAttentionLayer(layer)
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
