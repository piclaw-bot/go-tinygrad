package model

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4GPUGenerate(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !gpu.Available() {
		t.Skip("GPU not available")
	}
	t.Cleanup(gpu.Shutdown)

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
	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok

	ids := g.Generate(tok.Encode("Hello"), 30)
	var out []string
	for _, id := range ids {
		out = append(out, tok.InvVocab[id])
	}
	result := strings.Join(out, "")
	fmt.Printf("[gpu] %s\n", result)
	t.Logf("gpu output: %s", result)

	// Show first generated tokens
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	for i := len(wrapped); i < len(ids) && i < len(wrapped)+10; i++ {
		t.Logf("  gen token %d: id=%d → %q", i-len(wrapped), ids[i], tok.InvVocab[ids[i]])
	}
}
