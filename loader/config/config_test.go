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
