package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma4FirstTokenDequantized(t *testing.T) {
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
	ids := m.Generate(tok.Encode("Hello"), 1)
	t.Logf("first token id: %d → %q", ids[len(ids)-1], tok.InvVocab[ids[len(ids)-1]])

	ids5 := m.Generate(tok.Encode("Hello"), 5)
	for i, id := range ids5[len(ids5)-5:] {
		t.Logf("token %d: id=%d → %q", i, id, tok.InvVocab[id])
	}
}
