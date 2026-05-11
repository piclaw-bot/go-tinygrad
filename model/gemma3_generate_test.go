package model

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma3DequantizedCPUGenerate(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1")
	}
	dir := "../../models/gemma3-1b-mlx4"
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		dir = "../models/gemma3-1b-mlx4"
	}
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
	ids := m.Generate(tok.Encode("Hello"), 20)
	var out []string
	for _, id := range ids {
		out = append(out, tok.InvVocab[id])
	}
	fmt.Printf("[gemma3-deq-cpu] %s\n", strings.Join(out, ""))
	t.Logf("gemma3-deq-cpu output: %s", strings.Join(out, ""))
}
