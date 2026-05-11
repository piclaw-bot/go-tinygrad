package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"

	"github.com/rcarmo/go-pherence/backends/simd"
	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4InputNormBufferIsolation(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 inputnorm buffer isolation")
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
		t.Fatalf("load gemma4 cpu model: %v", err)
	}
	tok, err := tokenizer.Load(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	cpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || step != traceStep || layer != 0 {
			return
		}
		cpuOps[opTraceKey{layer: layer, op: op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()
	_ = m.Generate(tok.Encode("Hello"), 1)
	cpuHidden := cpuOps[opTraceKey{0, "hidden_in"}]
	if len(cpuHidden) == 0 {
		t.Fatal("missing cpu layer0 hidden_in trace")
	}
	want := append([]float32(nil), cpuHidden...)
	simd.RMSNormBF16(want, m.Layers[0].InputNorm.Data(), float32(m.Config.RMSNormEps))

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
	weight := g.Layers[0].InputNorm

	cases := []struct {
		name        string
		useModelIn  bool
		useModelOut bool
	}{
		{"fresh_in_fresh_out", false, false},
		{"model_in_fresh_out", true, false},
		{"fresh_in_model_out", false, true},
		{"model_in_model_out", true, true},
	}

	for _, tc := range cases {
		var inBuf, outBuf *gpu.DevBuf
		if tc.useModelIn {
			inBuf = g.hidden
			copy(inBuf.Data(), cpuHidden)
			inBuf.MarkDirty()
		} else {
			inBuf = gpu.NewDevBufFrom(append([]float32(nil), cpuHidden...))
			if err := inBuf.ToGPU(); err != nil {
				t.Fatalf("%s inBuf.ToGPU: %v", tc.name, err)
			}
		}
		if tc.useModelOut {
			outBuf = g.normed
		} else {
			outBuf = gpu.NewDevBuf(len(cpuHidden))
			if err := outBuf.ToGPU(); err != nil {
				t.Fatalf("%s outBuf.ToGPU: %v", tc.name, err)
			}
		}
		gpu.DevRMSNorm(outBuf, inBuf, weight, float32(m.Config.RMSNormEps))
		gpu.DevToBF16(outBuf, len(cpuHidden))
		gpu.Sync()
		got := append([]float32(nil), outBuf.Data()[:len(cpuHidden)]...)
		maxAbs, meanAbs := diffStats(want, got)
		t.Logf("%s: maxAbs=%.6g meanAbs=%.6g", tc.name, maxAbs, meanAbs)
	}
}
