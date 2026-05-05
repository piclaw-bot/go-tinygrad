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

func gemma4Path() string {
	p := os.Getenv("GEMMA4_PATH")
	if p == "" {
		p = "../../models/gemma4-e2b-mlx4"
		if _, err := os.Stat(p); err != nil {
			p = "../models/gemma4-e2b-mlx4"
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

func TestGemma4ChatTemplate(t *testing.T) {
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if _, err := os.Stat(dir + "/tokenizer.json"); err != nil {
		t.Skipf("tokenizer not found: %s", dir)
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4: %v", err)
	}
	if m.Config.ModelType != "gemma4_text" {
		t.Skipf("not gemma4_text: %s", m.Config.ModelType)
	}

	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok

	turnStart, turnEnd, newlineID := -1, -1, -1
	for id, tokStr := range tok.InvVocab {
		if tokStr == "<|turn>" {
			turnStart = id
		}
		if tokStr == "<turn|>" {
			turnEnd = id
		}
		if tokStr == "\n" {
			newlineID = id
		}
	}
	if turnStart < 0 || turnEnd < 0 || newlineID < 0 {
		t.Fatalf("missing special tokens: turnStart=%d turnEnd=%d newline=%d", turnStart, turnEnd, newlineID)
	}
	if newlineID != 107 {
		t.Fatalf("newline token=%d want 107", newlineID)
	}
	if ids := tok.Encode("\n"); len(ids) != 0 {
		t.Fatalf("expected bare newline encode to fail and require vocab scan, got %v", ids)
	}

	prompt := "Hello"
	ids := tok.Encode(prompt)
	ids = append([]int{m.Config.BOSTokenID}, ids...)
	user := tok.Encode("user")
	mdl := tok.Encode("model")
	wrapped := []int{m.Config.BOSTokenID, turnStart}
	wrapped = append(wrapped, user...)
	wrapped = append(wrapped, newlineID)
	wrapped = append(wrapped, ids[1:]...)
	wrapped = append(wrapped, turnEnd)
	wrapped = append(wrapped, newlineID)
	wrapped = append(wrapped, turnStart)
	wrapped = append(wrapped, mdl...)
	wrapped = append(wrapped, newlineID)

	decoded := tok.Decode(wrapped)
	want := "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"
	if decoded != want {
		t.Fatalf("template decode mismatch\n got: %q\nwant: %q", decoded, want)
	}
}
