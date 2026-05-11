//go:build diagnostic
// +build diagnostic

package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestGemma4CPUQuantizedVsCPUTrace(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 CPU quantized trace")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 cpu model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	prompt := "Hello"
	wrapped := wrapGemma4PromptForTest(m, prompt)
	traceStep := len(wrapped) - 1
	targets := map[int]bool{0: true, 14: true, 15: true, 34: true}

	type trace struct {
		layers map[int][]float32
		logits []float32
	}
	cpuTrace := trace{layers: map[int][]float32{}}
	qTrace := trace{layers: map[int][]float32{}}

	debugLayerHook = func(backend string, step, layer int, hidden []float32) {
		if step != traceStep || !targets[layer] {
			return
		}
		cp := append([]float32(nil), hidden...)
		if backend == "cpu" {
			cpuTrace.layers[layer] = cp
		} else if backend == "cpuq" {
			qTrace.layers[layer] = cp
		}
	}
	debugLogitsHook = func(backend string, step int, hidden, logits []float32) {
		if step != traceStep {
			return
		}
		cp := append([]float32(nil), logits...)
		if backend == "cpu" {
			cpuTrace.logits = cp
		} else if backend == "cpuq" {
			qTrace.logits = cp
		}
	}
	defer func() {
		debugLayerHook = nil
		debugLogitsHook = nil
	}()

	_ = m.Generate(tok.Encode(prompt), 1)

	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()
	mq, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 quantized cpu model: %v", err)
	}
	mq.Tok = tok

	prevLayerHook := debugLayerHook
	prevLogitsHook := debugLogitsHook
	debugLayerHook = func(backend string, step, layer int, hidden []float32) {
		if step != traceStep || !targets[layer] {
			return
		}
		cp := append([]float32(nil), hidden...)
		qTrace.layers[layer] = cp
	}
	debugLogitsHook = func(backend string, step int, hidden, logits []float32) {
		if step != traceStep {
			return
		}
		qTrace.logits = append([]float32(nil), logits...)
	}
	_ = mq.Generate(tok.Encode(prompt), 1)
	debugLayerHook = prevLayerHook
	debugLogitsHook = prevLogitsHook

	for _, l := range []int{0, 14, 15, 34} {
		c, ok1 := cpuTrace.layers[l]
		qv, ok2 := qTrace.layers[l]
		if !ok1 || !ok2 {
			t.Fatalf("missing layer trace %d (cpu=%v cpuq=%v)", l, ok1, ok2)
		}
		maxAbs, meanAbs := diffStats(c, qv)
		t.Logf("layer %d: maxAbs=%.6g meanAbs=%.6g", l, maxAbs, meanAbs)
	}
	if len(cpuTrace.logits) == 0 || len(qTrace.logits) == 0 {
		t.Fatal("missing final logits trace")
	}
	maxAbs, meanAbs := diffStats(cpuTrace.logits, qTrace.logits)
	cpuTop := topLogits(cpuTrace.logits, 5)
	qTop := topLogits(qTrace.logits, 5)
	t.Logf("final logits: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	t.Logf("cpu top5 ids=%v toks=%v", cpuTop, tokenNames(tok, cpuTop))
	t.Logf("cpuq top5 ids=%v toks=%v", qTop, tokenNames(tok, qTop))
}
