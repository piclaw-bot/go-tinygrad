package model

import (
	"os"
	"strings"
	"testing"
)

func smolLMPath() string {
	p := os.Getenv("SMOLLM_PATH")
	if p == "" {
		p = "../../models/smollm2-135m"
		if _, err := os.Stat(p); err != nil {
			p = "../models/smollm2-135m"
		}
	}
	return p
}

func TestLoadSmolLM(t *testing.T) {
	dir := smolLMPath()
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if m.Config.NumLayers != 30 {
		t.Fatalf("layers=%d want 30", m.Config.NumLayers)
	}
	t.Logf("Loaded SmolLM2-135M: %d layers, h=%d, heads=%d, kv_heads=%d",
		m.Config.NumLayers, m.Config.HiddenSize, m.Config.NumHeads, m.Config.NumKVHeads)
}

func TestTokenizer(t *testing.T) {
	dir := smolLMPath()
	if _, err := os.Stat(dir + "/tokenizer.json"); err != nil {
		t.Skipf("tokenizer not found: %s", dir)
	}

	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	t.Logf("Vocab size: %d, merges: %d", tok.VocabSize(), len(tok.Merges))

	// Encode and decode
	text := "Hello world"
	ids := tok.Encode(text)
	t.Logf("'%s' → %v", text, ids)
	if len(ids) == 0 {
		t.Fatal("empty encoding")
	}

	decoded := tok.Decode(ids)
	t.Logf("Decoded: '%s'", decoded)
	if !strings.Contains(decoded, "ello") {
		t.Fatalf("decode doesn't contain 'ello': '%s'", decoded)
	}
}

func TestSmolLMGenerate(t *testing.T) {
	dir := smolLMPath()
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	prompt := "The meaning of life is"
	ids := tok.Encode(prompt)
	t.Logf("Prompt: '%s' → %d tokens: %v", prompt, len(ids), ids)

	output := m.Generate(ids, 20)
	text := tok.Decode(output)
	t.Logf("Generated: '%s'", text)

	if len(output) <= len(ids) {
		t.Fatal("no tokens generated")
	}
}
