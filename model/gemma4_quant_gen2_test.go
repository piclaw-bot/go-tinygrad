//go:build diagnostic
// +build diagnostic

package model

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma4QuantizedCPUGenerate2(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("tok: %v", err)
	}
	m.Tok = tok
	ids := m.Generate(tok.Encode("Hello"), 30)
	var out []string
	for _, id := range ids {
		out = append(out, tok.InvVocab[id])
	}
	result := strings.Join(out, "")
	fmt.Printf("[quant-cpu] %s\n", result)
	t.Logf("quant-cpu output: %s", result)
}
