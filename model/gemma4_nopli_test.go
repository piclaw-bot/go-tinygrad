package model

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma4NoPLIGenerate(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("tok: %v", err)
	}
	m.Tok = tok

	// Ablation: disable PLI
	for i := range m.Layers {
		m.Layers[i].PLIGate = nil
		m.Layers[i].PLIProj = nil
		m.Layers[i].PLIPostNorm = nil
	}
	m.PerLayerModelProj = nil

	ids := m.Generate(tok.Encode("Hello"), 20)
	var out []string
	for _, id := range ids {
		out = append(out, tok.InvVocab[id])
	}
	fmt.Printf("[gemma4-no-pli] %s\n", strings.Join(out, ""))
	t.Logf("output: %s", strings.Join(out, ""))
}

func TestGemma4NoScalarGenerate(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("tok: %v", err)
	}
	m.Tok = tok

	// Ablation: disable layer scalars
	for i := range m.Layers {
		m.Layers[i].LayerScalar = 1.0
	}

	ids := m.Generate(tok.Encode("Hello"), 20)
	var out []string
	for _, id := range ids {
		out = append(out, tok.InvVocab[id])
	}
	fmt.Printf("[gemma4-no-scalar] %s\n", strings.Join(out, ""))
	t.Logf("output: %s", strings.Join(out, ""))
}
