package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadJSONReturnsRawBytes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	data := []byte(`{"name":"demo","n":7}`)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Name string `json:"name"`
		N    int    `json:"n"`
	}
	raw, err := ReadJSON(path, &got)
	if err != nil {
		t.Fatalf("ReadJSON: %v", err)
	}
	if got.Name != "demo" || got.N != 7 {
		t.Fatalf("decoded=%+v", got)
	}
	if string(raw) != string(data) {
		t.Fatalf("raw=%q want %q", raw, data)
	}
}

func TestReadOptionalJSONMissing(t *testing.T) {
	var got struct{ N int }
	ok, err := ReadOptionalJSON(filepath.Join(t.TempDir(), "missing.json"), &got)
	if err != nil {
		t.Fatalf("ReadOptionalJSON: %v", err)
	}
	if ok {
		t.Fatal("ReadOptionalJSON reported missing file as present")
	}
}

func TestReadModelAndQuantizeConfig(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"x"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "quantize_config.json"), []byte(`{"bits":4}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		ModelType string `json:"model_type"`
	}
	if _, err := ReadModelConfig(dir, &cfg); err != nil {
		t.Fatalf("ReadModelConfig: %v", err)
	}
	if cfg.ModelType != "x" {
		t.Fatalf("ModelType=%q", cfg.ModelType)
	}
	var q struct {
		Bits int `json:"bits"`
	}
	ok, err := ReadQuantizeConfig(dir, &q)
	if err != nil {
		t.Fatalf("ReadQuantizeConfig: %v", err)
	}
	if !ok || q.Bits != 4 {
		t.Fatalf("ok=%v q=%+v", ok, q)
	}
}

func TestClassifyNVFP4TensorPrefix(t *testing.T) {
	cases := []struct {
		prefix string
		want   NVFP4TensorRole
	}{
		{"model.layers.0.self_attn.q_proj", NVFP4RoleAttentionQ},
		{"model.layers.0.self_attn.k_proj", NVFP4RoleAttentionK},
		{"model.layers.0.self_attn.v_proj", NVFP4RoleAttentionV},
		{"model.layers.0.self_attn.o_proj", NVFP4RoleAttentionO},
		{"model.language_model.layers.0.self_attn.q_proj", NVFP4RoleAttentionQ},
		{"model.layers.0.mlp.gate_proj", NVFP4RoleMLPGate},
		{"model.layers.0.mlp.up_proj", NVFP4RoleMLPUp},
		{"model.layers.0.mlp.down_proj", NVFP4RoleMLPDown},
		{"model.language_model.layers.0.mlp.gate_proj", NVFP4RoleMLPGate},
		{"model.language_model.layers.0.mlp.up_proj", NVFP4RoleMLPUp},
		{"model.language_model.layers.0.mlp.down_proj", NVFP4RoleMLPDown},
		{"model.layers.0.mlp.experts.7.gate_proj", NVFP4RoleMoEExpertGate},
		{"model.layers.0.mlp.experts.7.up_proj", NVFP4RoleMoEExpertUp},
		{"model.layers.0.mlp.experts.7.down_proj", NVFP4RoleMoEExpertDown},
		{"model.language_model.layers.0.experts.7.down_proj", NVFP4RoleMoEExpertDown},
		{"adapter.experts.7.down_proj", NVFP4RoleUnknown},
		{"model.layers.0.mlp.gate", NVFP4RoleRouter},
		{"model.language_model.layers.0.mlp.gate", NVFP4RoleRouter},
		{"lm_head", NVFP4RoleUnknown},
	}
	for _, tc := range cases {
		if got := ClassifyNVFP4TensorPrefix(tc.prefix); got != tc.want {
			t.Fatalf("ClassifyNVFP4TensorPrefix(%q)=%q want %q", tc.prefix, got, tc.want)
		}
	}
}

func TestNVFP4CompanionNames(t *testing.T) {
	w, s, s2, in := NVFP4CompanionNames("model.layers.0.self_attn.q_proj")
	if w != "model.layers.0.self_attn.q_proj.weight" || s != "model.layers.0.self_attn.q_proj.weight_scale" || s2 != "model.layers.0.self_attn.q_proj.weight_scale_2" || in != "model.layers.0.self_attn.q_proj.input_scale" {
		t.Fatalf("companions=%q %q %q %q", w, s, s2, in)
	}
}

func TestQwen35LinearAttentionShapesFor(t *testing.T) {
	got, err := Qwen35LinearAttentionShapesFor(5120, 1024, 128, 4, 16, 4)
	if err != nil {
		t.Fatalf("Qwen35LinearAttentionShapesFor: %v", err)
	}
	if got.KeyDim != 512 || got.ValueDim != 1024 || got.ConvDim != 2048 || got.HeadVDim != 64 {
		t.Fatalf("linear dims=%+v", got)
	}
	if got.QKV[0] != 5120 || got.QKV[1] != 2048 || got.Gate[1] != 1024 || got.Conv1D[0] != 4 || got.Conv1D[1] != 2048 || got.Out[0] != 1024 || got.Out[1] != 5120 {
		t.Fatalf("linear shapes=%+v", got)
	}
	if _, err := Qwen35LinearAttentionShapesFor(5120, 1025, 128, 4, 16, 4); err == nil {
		t.Fatal("non-divisible inner/dt_rank returned nil error")
	}
}

func TestQwen35FullAttentionShapesFor(t *testing.T) {
	got, err := Qwen35FullAttentionShapesFor(5120, 24, 4, 256)
	if err != nil {
		t.Fatalf("Qwen35FullAttentionShapesFor: %v", err)
	}
	if got.QProj[0] != 12288 || got.QProj[1] != 5120 || got.KProj[0] != 1024 || got.VProj[0] != 1024 || got.OProj[0] != 5120 || got.OProj[1] != 6144 || got.GateSize != 6144 {
		t.Fatalf("shapes=%+v", got)
	}
	if _, err := Qwen35FullAttentionShapesFor(0, 24, 4, 256); err == nil {
		t.Fatal("invalid dims returned nil error")
	}
}

func TestParseQwenNativeMTPMetadata(t *testing.T) {
	json := `{
		"architectures":["Qwen3_5ForConditionalGeneration"],
		"model_type":"qwen3_5",
		"text_config":{
			"model_type":"qwen3_5_text",
			"hidden_size":5120,
			"num_hidden_layers":64,
			"num_attention_heads":24,
			"num_key_value_heads":4,
			"head_dim":256,
			"linear_conv_kernel_dim":4,
			"linear_key_head_dim":128,
			"linear_num_key_heads":4,
			"linear_num_value_heads":16,
			"linear_value_head_dim":64,
			"full_attention_interval":4,
			"mtp_num_hidden_layers":1,
			"mtp_use_dedicated_embeddings":false,
			"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]
		}
	}`
	got, err := ParseQwenNativeMTPMetadata([]byte(json))
	if err != nil {
		t.Fatalf("ParseQwenNativeMTPMetadata: %v", err)
	}
	if got.ModelType != "qwen3_5_text" || got.Architecture != "Qwen3_5ForConditionalGeneration" || got.HiddenSize != 5120 || got.NumHiddenLayers != 64 || got.NumAttentionHeads != 24 || got.NumKeyValueHeads != 4 || got.HeadDim != 256 || got.LinearNumValueHeads != 16 || got.MTPNumHiddenLayers != 1 || !got.HasNativeMTP || !got.HasLinearAttention {
		t.Fatalf("metadata=%+v", got)
	}
}

func TestQwenNativeMTPLayerClassification(t *testing.T) {
	meta := QwenNativeMTPMetadata{
		NumHiddenLayers:       65,
		MTPNumHiddenLayers:    1,
		FullAttentionInterval: 4,
	}
	if got := meta.MainLayerCount(); got != 64 {
		t.Fatalf("MainLayerCount=%d want 64", got)
	}
	if !meta.IsLinearAttentionLayer(0) || !meta.IsLinearAttentionLayer(2) || meta.IsLinearAttentionLayer(3) {
		t.Fatalf("interval linear/full classification failed")
	}
	if !meta.IsFullAttentionLayer(3) || meta.IsFullAttentionLayer(0) {
		t.Fatalf("interval full classification failed")
	}
	if !meta.IsMTPLayer(64) || meta.IsMTPLayer(63) {
		t.Fatalf("MTP layer classification failed")
	}

	meta.LayerTypes = []string{"full_attention", "linear_attention"}
	if !meta.IsFullAttentionLayer(0) || !meta.IsLinearAttentionLayer(1) {
		t.Fatalf("layer_types classification failed")
	}
}

func TestRequiredAndMissingQwenNativeMTPTensors(t *testing.T) {
	req := RequiredQwenNativeMTPTensors(1)
	if len(req) != 15 {
		t.Fatalf("required len=%d want 15: %v", len(req), req)
	}
	missing := MissingQwenNativeMTPTensors(req, 1)
	if len(missing) != 0 {
		t.Fatalf("missing=%v want none", missing)
	}
	missing = MissingQwenNativeMTPTensors(req[:len(req)-1], 1)
	if len(missing) != 1 || missing[0] != req[len(req)-1] {
		t.Fatalf("missing=%v want %q", missing, req[len(req)-1])
	}
	if got := RequiredQwenNativeMTPTensors(0); got != nil {
		t.Fatalf("zero layers required=%v want nil", got)
	}
}

func TestIsQwenNativeMTPTensorName(t *testing.T) {
	cases := []struct {
		name string
		want bool
	}{
		{"mtp.fc.weight", true},
		{"mtp.pre_fc_norm_embedding.weight", true},
		{"mtp.pre_fc_norm_hidden.weight", true},
		{"mtp.norm.weight", true},
		{"mtp.layers.0.self_attn.q_proj.weight", true},
		{"model.layers.0.self_attn.q_proj.weight", false},
		{"lm_head.weight", false},
	}
	for _, tc := range cases {
		if got := IsQwenNativeMTPTensorName(tc.name); got != tc.want {
			t.Fatalf("IsQwenNativeMTPTensorName(%q)=%v want %v", tc.name, got, tc.want)
		}
	}
}

func TestParseQuantizationMetadata(t *testing.T) {
	cases := []struct {
		name            string
		json            string
		wantMethod      string
		wantAlgo        string
		wantBits        int
		wantGroup       int
		wantUnsupported bool
	}{
		{
			name:       "modelopt nvfp4",
			json:       `{"quantization_config":{"quant_algo":"NVFP4","quant_method":"modelopt","config_groups":{"group_0":{"weights":{"num_bits":4,"type":"float","group_size":16}}}}}`,
			wantMethod: "modelopt", wantAlgo: "NVFP4", wantBits: 4, wantGroup: 16, wantUnsupported: true,
		},
		{
			name:       "compressed tensors fp4",
			json:       `{"quantization_config":{"quant_method":"compressed-tensors","config_groups":{"group_0":{"weights":{"num_bits":4,"type":"float","group_size":32}}}}}`,
			wantMethod: "compressed-tensors", wantBits: 4, wantGroup: 32, wantUnsupported: true,
		},
		{
			name:       "compressed tensors nvfp4 group format",
			json:       `{"quantization_config":{"quant_method":"compressed-tensors","config_groups":{"group_0":{"format":"nvfp4-pack-quantized","weights":{"num_bits":4,"group_size":16}}}}}`,
			wantMethod: "compressed-tensors", wantBits: 4, wantGroup: 16, wantUnsupported: true,
		},
		{
			name:       "compressed tensors nvfp4 weight format",
			json:       `{"quantization_config":{"quant_method":"compressed-tensors","config_groups":{"group_0":{"weights":{"num_bits":4,"format":"nvfp4-pack-quantized","group_size":16}}}}}`,
			wantMethod: "compressed-tensors", wantBits: 4, wantGroup: 16, wantUnsupported: true,
		},
		{
			name:       "compressed tensors mixed groups detects fp4 regardless map order",
			json:       `{"quantization_config":{"quant_method":"compressed-tensors","config_groups":{"group_a":{"weights":{"num_bits":8,"type":"int","group_size":64}},"group_b":{"weights":{"num_bits":4,"type":"float","group_size":16}}}}}`,
			wantMethod: "compressed-tensors", wantBits: 4, wantGroup: 16, wantUnsupported: true,
		},
		{
			name:       "mlx int4 remains supported",
			json:       `{"quantization":{"bits":4,"group_size":64}}`,
			wantMethod: "mlx", wantBits: 4, wantGroup: 64, wantUnsupported: false,
		},
		{
			name:       "gptq style hf int4 remains supported",
			json:       `{"quantization_config":{"quant_method":"gptq","bits":4,"group_size":128,"sym":true}}`,
			wantMethod: "gptq", wantBits: 4, wantGroup: 128, wantUnsupported: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseQuantizationMetadata([]byte(tc.json))
			if err != nil {
				t.Fatalf("ParseQuantizationMetadata: %v", err)
			}
			if got.Method != tc.wantMethod || got.Algo != tc.wantAlgo || got.Bits != tc.wantBits || got.GroupSize != tc.wantGroup || got.UnsupportedFP4 != tc.wantUnsupported {
				t.Fatalf("got=%+v", got)
			}
		})
	}
}
