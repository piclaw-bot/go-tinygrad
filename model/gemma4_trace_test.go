//go:build diagnostic
// +build diagnostic

package model

import (
	"fmt"
	"math"
	"os"
	"sort"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

type gemma4Trace struct {
	layers map[int][]float32
	logits []float32
}

func wrapGemma4PromptForTest(m *LlamaModel, prompt string) []int {
	ids := m.Tok.Encode(prompt)
	cfg := m.Config
	if cfg.BOSTokenID > 0 {
		ids = append([]int{cfg.BOSTokenID}, ids...)
	}
	turnStart, turnEnd, newlineID := -1, -1, -1
	for id, tok := range m.Tok.InvVocab {
		if tok == "<|turn>" {
			turnStart = id
		}
		if tok == "<turn|>" {
			turnEnd = id
		}
		if tok == "\n" {
			newlineID = id
		}
	}
	user := m.Tok.Encode("user")
	mdl := m.Tok.Encode("model")
	wrapped := []int{cfg.BOSTokenID, turnStart}
	wrapped = append(wrapped, user...)
	wrapped = append(wrapped, newlineID)
	wrapped = append(wrapped, ids[1:]...)
	wrapped = append(wrapped, turnEnd)
	wrapped = append(wrapped, newlineID)
	wrapped = append(wrapped, turnStart)
	wrapped = append(wrapped, mdl...)
	wrapped = append(wrapped, newlineID)
	return wrapped
}

func topLogits(logits []float32, n int) []int {
	type pair struct {
		id int
		v  float32
	}
	ps := make([]pair, len(logits))
	for i, v := range logits {
		ps[i] = pair{i, v}
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].v > ps[j].v })
	if n > len(ps) {
		n = len(ps)
	}
	out := make([]int, n)
	for i := 0; i < n; i++ {
		out[i] = ps[i].id
	}
	return out
}

func diffStats(a, b []float32) (maxAbs, meanAbs float64) {
	if len(a) != len(b) {
		return math.Inf(1), math.Inf(1)
	}
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		meanAbs += d
		if d > maxAbs {
			maxAbs = d
		}
	}
	if len(a) > 0 {
		meanAbs /= float64(len(a))
	}
	return
}

func tokenNames(tok *tokenizer.Tokenizer, ids []int) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = tok.InvVocab[id]
	}
	return out
}

func TestGemma4CPUGPUTrace(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 CPU/GPU layer trace")
	}
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !gpu.Available() {
		t.Skip("GPU not available")
	}
	t.Cleanup(gpu.Shutdown)

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4: %v", err)
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
	cpuTrace := &gemma4Trace{layers: map[int][]float32{}}
	gpuTrace := &gemma4Trace{layers: map[int][]float32{}}

	debugLayerHook = func(backend string, step, layer int, hidden []float32) {
		if step != traceStep || !targets[layer] {
			return
		}
		cp := append([]float32(nil), hidden...)
		if backend == "cpu" {
			cpuTrace.layers[layer] = cp
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
			cpuTrace.logits = cp
		} else if backend == "gpu" {
			gpuTrace.logits = cp
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
	mgpu, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4 gpu model: %v", err)
	}
	mgpu.Tok = tok
	g, err := LoadGPUModel(mgpu)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	t.Cleanup(g.Close)
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode(prompt), 1)

	for _, l := range []int{0, 14, 15, 34} {
		c, ok1 := cpuTrace.layers[l]
		g2, ok2 := gpuTrace.layers[l]
		if !ok1 || !ok2 {
			t.Fatalf("missing layer trace %d (cpu=%v gpu=%v)", l, ok1, ok2)
		}
		maxAbs, meanAbs := diffStats(c, g2)
		t.Logf("layer %d: maxAbs=%.6g meanAbs=%.6g", l, maxAbs, meanAbs)
	}
	if len(cpuTrace.logits) == 0 || len(gpuTrace.logits) == 0 {
		t.Fatal("missing final logits trace")
	}
	maxAbs, meanAbs := diffStats(cpuTrace.logits, gpuTrace.logits)
	cpuTop := topLogits(cpuTrace.logits, 5)
	gpuTop := topLogits(gpuTrace.logits, 5)
	t.Logf("final logits: maxAbs=%.6g meanAbs=%.6g", maxAbs, meanAbs)
	t.Logf("cpu top5 ids=%v toks=%v", cpuTop, tokenNames(tok, cpuTop))
	t.Logf("gpu top5 ids=%v toks=%v", gpuTop, tokenNames(tok, gpuTop))
	fmt.Printf("[gemma4-trace] cpu top1=%d(%q) gpu top1=%d(%q)\n", cpuTop[0], tok.InvVocab[cpuTop[0]], gpuTop[0], tok.InvVocab[gpuTop[0]])
}
