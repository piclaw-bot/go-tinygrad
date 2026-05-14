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
