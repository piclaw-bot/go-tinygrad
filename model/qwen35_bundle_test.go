package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadQwen35NativeMTPBundleFromDir(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 1
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"full_attention"}
	dir := t.TempDir()
	config := `{"model_type":"qwen3_5_text","hidden_size":4,"intermediate_size":6,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"linear_conv_kernel_dim":3,"linear_key_head_dim":2,"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_value_head_dim":2,"layer_types":["full_attention"]}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := writeTinySafetensors(filepath.Join(dir, "model.safetensors"), mapFromQwen35Source(fullQwen35LayerSource(meta, "model.layers.0"))); err != nil {
		t.Fatalf("writeTinySafetensors: %v", err)
	}
	bundle, err := LoadQwen35NativeMTPBundleFromDir(dir)
	if err != nil {
		t.Fatalf("LoadQwen35NativeMTPBundleFromDir: %v", err)
	}
	if bundle.Meta.HiddenSize != 4 || bundle.Base == nil || len(bundle.Base.Layers) != 1 || bundle.MTP != nil {
		t.Fatalf("bundle=%+v", bundle)
	}
}
