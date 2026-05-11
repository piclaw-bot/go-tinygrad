package model

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma4NoChatTemplateGenerate(t *testing.T) {
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
	// Temporarily override model type to skip chat template
	origType := m.Config.ModelType
	m.Config.ModelType = "gemma3_text" // use Gemma3 code path (no chat template)
	m.Tok = tok
	ids := m.Generate(tok.Encode("Hello"), 20)
	m.Config.ModelType = origType
	var out []string
	for _, id := range ids {
		out = append(out, tok.InvVocab[id])
	}
	fmt.Printf("[gemma4-no-chat] %s\n", strings.Join(out, ""))
	t.Logf("output: %s", strings.Join(out, ""))
}
