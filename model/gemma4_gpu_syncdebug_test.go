package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4GenerateSyncDebug(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 sync-debug generate")
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
	oldSync := os.Getenv("GEMMA4_GPU_SYNC_DEBUG")
	ForceOnTheFly = true
	os.Setenv("GEMMA4_GPU_SYNC_DEBUG", "1")
	defer func() {
		ForceOnTheFly = oldForce
		if oldSync == "" {
			os.Unsetenv("GEMMA4_GPU_SYNC_DEBUG")
		} else {
			os.Setenv("GEMMA4_GPU_SYNC_DEBUG", oldSync)
		}
	}()

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 gpu model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	t.Cleanup(g.Close)
	g.CPU.Tok = tok

	_ = g.Generate(tok.Encode("Hello"), 1)
}
