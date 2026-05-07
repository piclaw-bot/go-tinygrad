package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4QuantizedCPUvsGPUTrace(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 quantized CPU vs GPU trace")
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

	mq, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized cpu model: %v", err)
	}
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	mq.Tok = tok
	prompt := "Hello"
	wrapped := wrapGemma4PromptForTest(mq, prompt)
	traceStep := len(wrapped) - 1
	targets := map[int]bool{0: true, 14: true, 15: true, 34: true}

	type trace struct {
		layers map[int][]float32
		logits []float32
	}
	cpuqTrace := trace{layers: map[int][]float32{}}
	gpuTrace := trace{layers: map[int][]float32{}}

	debugLayerHook = func(backend string, step, layer int, hidden []float32) {
		if step != traceStep || !targets[layer] {
			return
		}
		cp := append([]float32(nil), hidden...)
		if backend == "cpu" {
			cpuqTrace.layers[layer] = cp
		} else if backend == "gpu" {
			gpuTrace.layers[layer] = cp
		}
	}
	debugLogitsHook = func(backend string, step int, hidden, logits []float32) {
		if step != traceStep {
			return
		}
		cp := append([]float32(nil), logits...)
		if backend == "cpu" {
			cpuqTrace.logits = cp
		} else if backend == "gpu" {
			gpuTrace.logits = cp
		}
	}
	defer func() {
		debugLayerHook = nil
		debugLogitsHook = nil
	}()

	_ = mq.Generate(tok.Encode(prompt), 1)

	mgpu, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized gpu model: %v", err)
	}
	mgpu.Tok = tok
	g, err := LoadGPUModel(mgpu)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode(prompt), 1)

	for _, l := range []int{0, 14, 15, 34} {
		c, ok1 := cpuqTrace.layers[l]
		g2, ok2 := gpuTrace.layers[l]
		if !ok1 || !ok2 {
			t.Fatalf("missing layer trace %d (cpuq=%v gpu=%v)", l, ok1, ok2)
		}
		maxAbs, meanAbs := diffStats(c, g2)
		t.Logf("layer %d: maxAbs=%.6g meanAbs=%.6g", l, maxAbs, meanAbs)
	}
	if len(cpuqTrace.logits) == 0 || len(gpuTrace.logits) == 0 {
		t.Fatal("missing final logits trace")
	}
	maxAbs, meanAbs := diffStats(cpuqTrace.logits, gpuTrace.logits)
	cpuTop := topLogits(cpuqTrace.logits, 5)
	gpuTop := topLogits(gpuTrace.logits, 5)
	t.Logf("final logits: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	t.Logf("cpuq top5 ids=%v toks=%v", cpuTop, tokenNames(tok, cpuTop))
	t.Logf("gpu top5 ids=%v toks=%v", gpuTop, tokenNames(tok, gpuTop))
}
