package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func TestGemma4Layer15DownBufferIsolation(t *testing.T) {
	if os.Getenv("GEMMA4_TRACE_TEST") == "" {
		t.Skip("set GEMMA4_TRACE_TEST=1 for Gemma4 layer15 down isolation")
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
	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok
	wrapped := wrapGemma4PromptForTest(m, "Hello")
	traceStep := len(wrapped) - 1

	cpuOps := map[opTraceKey][]float32{}
	debugOpHook = func(backend string, step, layer int, op string, vec []float32) {
		if backend != "cpu" || step != traceStep || layer != 15 {
			return
		}
		cpuOps[opTraceKey{layer: layer, op: op}] = append([]float32(nil), vec...)
	}
	defer func() { debugOpHook = nil }()
	_ = m.Generate(tok.Encode("Hello"), 1)
	gateAct := cpuOps[opTraceKey{15, "gate_act"}]
	want := cpuOps[opTraceKey{15, "down"}]
	if len(gateAct) == 0 || len(want) == 0 {
		t.Fatal("missing cpu layer15 gate_act/down traces")
	}

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
	w := g.Layers[15].DownWmg
	if w == nil {
		t.Fatal("missing layer15 DownWmg")
	}

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
			inBuf = g.gate
			copy(inBuf.Data(), gateAct)
			inBuf.MarkDirty()
		} else {
			inBuf = gpu.NewDevBufFrom(append([]float32(nil), gateAct...))
			if err := inBuf.ToGPU(); err != nil {
				t.Fatalf("%s inBuf.ToGPU: %v", tc.name, err)
			}
		}
		if tc.useModelOut {
			outBuf = g.down
		} else {
			outBuf = gpu.NewDevBuf(len(want))
			if err := outBuf.ToGPU(); err != nil {
				t.Fatalf("%s outBuf.ToGPU: %v", tc.name, err)
			}
		}
		gpu.GemvMLX(outBuf, inBuf, w)
		gpu.Sync()
		got := append([]float32(nil), outBuf.Data()[:len(want)]...)
		maxAbs, meanAbs := diffStats(want, got)
		t.Logf("%s fast-down: maxAbs=%.6g meanAbs=%.6g", tc.name, maxAbs, meanAbs)
		if !tc.useModelIn {
			inBuf.Free()
		}
		if !tc.useModelOut {
			outBuf.Free()
		}
	}
}
