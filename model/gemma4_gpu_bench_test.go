package model

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4GPUBench(t *testing.T) {
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

	t0 := time.Now()
	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("tok: %v", err)
	}
	m.Tok = tok
	tLoad := time.Since(t0)

	t1 := time.Now()
	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	tGPU := time.Since(t1)

	t2 := time.Now()
	ids := g.Generate(tok.Encode("Hello"), 30)
	tGen := time.Since(t2)

	wrapped := wrapGemma4PromptForTest(m, "Hello")
	promptToks := len(wrapped)
	genToks := len(ids) - promptToks
	tokPerSec := float64(genToks) / tGen.Seconds()

	fmt.Printf("[bench] load=%.1fs gpu=%.1fs gen=%.1fs prompt=%d gen=%d tok/s=%.1f\n",
		tLoad.Seconds(), tGPU.Seconds(), tGen.Seconds(), promptToks, genToks, tokPerSec)
	t.Logf("load=%.1fs gpu=%.1fs gen=%.1fs prompt=%d gen=%d tok/s=%.1f",
		tLoad.Seconds(), tGPU.Seconds(), tGen.Seconds(), promptToks, genToks, tokPerSec)
}
