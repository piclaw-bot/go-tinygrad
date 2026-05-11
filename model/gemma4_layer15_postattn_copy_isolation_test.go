package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4Layer15PostAttnCopyIsolation(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 layer15 post-attn copy isolation")
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
		t.Fatalf("load gemma4 gpu model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	gpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "gpu" || step != traceStep || layer != 15 {
			return
		}
		gpuOps[opTraceKey{layer: layer, op: op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()

	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}
	defer g.Close()
	g.CPU.Tok = tok
	_ = g.Generate(tok.Encode("Hello"), 1)

	gpuHiddenIn := gpuOps[opTraceKey{15, "hidden_in"}]
	gpuO := gpuOps[opTraceKey{15, "o"}]
	gpuMLPIn := gpuOps[opTraceKey{15, "mlp_input"}]
	if len(gpuHiddenIn) == 0 || len(gpuO) == 0 || len(gpuMLPIn) == 0 {
		t.Fatal("missing layer15 hidden_in/o/mlp_input traces")
	}
	wBuf := g.Layers[15].PostNorm

	run := func(name string, useModel bool, doCopy bool) {
		var residual, oBuf, hidden, normed *gpu.DevBuf
		var freeList []*gpu.DevBuf
		if useModel {
			residual = g.residual
			oBuf = g.oOut
			hidden = g.hidden
			normed = g.normed
		} else {
			residual = gpu.NewDevBufFrom(append([]float32(nil), gpuHiddenIn...))
			oBuf = gpu.NewDevBufFrom(append([]float32(nil), gpuO...))
			hidden = gpu.NewDevBuf(len(gpuHiddenIn))
			normed = gpu.NewDevBuf(len(gpuHiddenIn))
			freeList = []*gpu.DevBuf{residual, oBuf, hidden, normed}
			for _, b := range freeList {
				if err := b.ToGPU(); err != nil {
					t.Fatalf("%s ToGPU: %v", name, err)
				}
			}
		}
		copy(residual.Data(), gpuHiddenIn)
		residual.MarkDirty()
		copy(oBuf.Data(), gpuO)
		oBuf.MarkDirty()
		gpu.DevAdd(hidden, residual, oBuf)
		if doCopy {
			gpu.DevCopy(residual, hidden)
		}
		gpu.DevRMSNorm(normed, hidden, wBuf, float32(m.Config.RMSNormEps))
		gpu.Sync()
		got := append([]float32(nil), normed.Data()[:len(gpuMLPIn)]...)
		maxAbs, meanAbs := diffStats(gpuMLPIn, got)
		t.Logf("%s: maxAbs=%.6g meanAbs=%.6g", name, maxAbs, meanAbs)
		for _, b := range freeList {
			b.Free()
		}
	}

	run("fresh_no_copy", false, false)
	run("fresh_with_copy", false, true)
	run("model_no_copy", true, false)
	run("model_with_copy", true, true)
}
