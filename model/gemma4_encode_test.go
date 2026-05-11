//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma4EncodeHello(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("tok: %v", err)
	}
	ids := tok.Encode("Hello")
	t.Logf("Encode('Hello') = %v", ids)
	for _, id := range ids {
		t.Logf("  %d → %q", id, tok.InvVocab[id])
	}
	// Check BOS
	m, _ := LoadLlama(dir)
	t.Logf("BOSTokenID = %d", m.Config.BOSTokenID)
}
