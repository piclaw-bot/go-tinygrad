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

func TestGemma4AblationLayerScalar(t *testing.T) {
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

	// Log layer scalars
	for i := 0; i < 5; i++ {
		t.Logf("layer %d scalar: %.6g", i, m.Layers[i].LayerScalar)
	}
	t.Logf("layer 14 scalar: %.6g", m.Layers[14].LayerScalar)
	t.Logf("layer 15 scalar: %.6g", m.Layers[15].LayerScalar)
	t.Logf("layer 34 scalar: %.6g", m.Layers[34].LayerScalar)

	// Log PLI presence
	t.Logf("PLIGate present: %v", m.Layers[0].PLIGate != nil)
	t.Logf("PerLayerModelProj present: %v", m.PerLayerModelProj != nil)

	// Log KV sharing
	for i := 15; i < 20; i++ {
		t.Logf("layer %d HasKV=%v KVSourceLayer=%d", i, m.Layers[i].HasKV, m.Layers[i].KVSourceLayer)
	}

	ids := m.Generate(tok.Encode("Hello"), 20)
	var out []string
	for _, id := range ids {
		out = append(out, tok.InvVocab[id])
	}
	fmt.Printf("[gemma4-ablation] %s\n", strings.Join(out, ""))
	t.Logf("output: %s", strings.Join(out, ""))
}
