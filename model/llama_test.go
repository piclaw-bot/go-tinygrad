package model

import (
	"os"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/gpu"
)

func smolLMPath() string {
	p := os.Getenv("SMOLLM_PATH")
	if p == "" {
		p = "../../models/smollm2-135m"
		if _, err := os.Stat(p); err != nil {
			p = "../models/smollm2-135m"
		}
	}
	return p
}

func gemma4Path() string {
	p := os.Getenv("GEMMA4_PATH")
	if p == "" {
		p = "../../models/gemma4-e2b-mlx4"
		if _, err := os.Stat(p); err != nil {
			p = "../models/gemma4-e2b-mlx4"
		}
	}
	return p
}

func TestLoadSmolLM(t *testing.T) {
	dir := smolLMPath()
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if m.Config.NumLayers != 30 {
		t.Fatalf("layers=%d want 30", m.Config.NumLayers)
	}
	t.Logf("Loaded SmolLM2-135M: %d layers, h=%d, heads=%d, kv_heads=%d",
		m.Config.NumLayers, m.Config.HiddenSize, m.Config.NumHeads, m.Config.NumKVHeads)
}

func TestTokenizer(t *testing.T) {
	dir := smolLMPath()
	if _, err := os.Stat(dir + "/tokenizer.json"); err != nil {
		t.Skipf("tokenizer not found: %s", dir)
	}

	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	t.Logf("Vocab size: %d, merges: %d", tok.VocabSize(), len(tok.Merges))

	// Encode and decode
	text := "Hello world"
	ids := tok.Encode(text)
	t.Logf("'%s' → %v", text, ids)
	if len(ids) == 0 {
		t.Fatal("empty encoding")
	}

	decoded := tok.Decode(ids)
	t.Logf("Decoded: '%s'", decoded)
	if !strings.Contains(decoded, "ello") {
		t.Fatalf("decode doesn't contain 'ello': '%s'", decoded)
	}
}

func TestSmolLMGenerate(t *testing.T) {
	dir := smolLMPath()
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	prompt := "The meaning of life is"
	ids := tok.Encode(prompt)
	t.Logf("Prompt: '%s' → %d tokens: %v", prompt, len(ids), ids)

	output := m.Generate(ids, 20)
	text := tok.Decode(output)
	t.Logf("Generated: '%s'", text)

	if len(output) <= len(ids) {
		t.Fatal("no tokens generated")
	}
}

func TestGemma4ChatTemplate(t *testing.T) {
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if _, err := os.Stat(dir + "/tokenizer.json"); err != nil {
		t.Skipf("tokenizer not found: %s", dir)
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4: %v", err)
	}
	if m.Config.ModelType != "gemma4_text" {
		t.Skipf("not gemma4_text: %s", m.Config.ModelType)
	}

	tok, err := LoadTokenizer(dir + "/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	m.Tok = tok

	turnStart, turnEnd, newlineID := -1, -1, -1
	for id, tokStr := range tok.InvVocab {
		if tokStr == "<|turn>" {
			turnStart = id
		}
		if tokStr == "<turn|>" {
			turnEnd = id
		}
		if tokStr == "\n" {
			newlineID = id
		}
	}
	if turnStart < 0 || turnEnd < 0 || newlineID < 0 {
		t.Fatalf("missing special tokens: turnStart=%d turnEnd=%d newline=%d", turnStart, turnEnd, newlineID)
	}
	if newlineID != 107 {
		t.Fatalf("newline token=%d want 107", newlineID)
	}
	if ids := tok.Encode("\n"); len(ids) != 0 {
		t.Fatalf("expected bare newline encode to fail and require vocab scan, got %v", ids)
	}

	prompt := "Hello"
	ids := tok.Encode(prompt)
	ids = append([]int{m.Config.BOSTokenID}, ids...)
	user := tok.Encode("user")
	mdl := tok.Encode("model")
	wrapped := []int{m.Config.BOSTokenID, turnStart}
	wrapped = append(wrapped, user...)
	wrapped = append(wrapped, newlineID)
	wrapped = append(wrapped, ids[1:]...)
	wrapped = append(wrapped, turnEnd)
	wrapped = append(wrapped, newlineID)
	wrapped = append(wrapped, turnStart)
	wrapped = append(wrapped, mdl...)
	wrapped = append(wrapped, newlineID)

	decoded := tok.Decode(wrapped)
	want := "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"
	if decoded != want {
		t.Fatalf("template decode mismatch\n got: %q\nwant: %q", decoded, want)
	}
}

func TestGemma4KVSharingCPU(t *testing.T) {
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4: %v", err)
	}
	if m.Config.ModelType != "gemma4_text" {
		t.Skipf("not gemma4_text: %s", m.Config.ModelType)
	}

	firstKVShared := m.Config.NumLayers - m.Config.NumKVSharedLayers
	if firstKVShared != 15 {
		t.Fatalf("firstKVShared=%d want 15", firstKVShared)
	}

	for l, layer := range m.Layers {
		if l < firstKVShared {
			if !layer.HasKV {
				t.Fatalf("layer %d should own KV", l)
			}
			continue
		}
		if layer.HasKV {
			t.Fatalf("layer %d should share KV", l)
		}
		expectSrc := 13
		if m.Config.LayerTypes[l] == "full_attention" {
			expectSrc = 14
		}
		if layer.KVSourceLayer != expectSrc {
			t.Fatalf("layer %d type=%s source=%d want %d", l, m.Config.LayerTypes[l], layer.KVSourceLayer, expectSrc)
		}
		if m.Config.LayerTypes[layer.KVSourceLayer] != m.Config.LayerTypes[l] {
			t.Fatalf("layer %d type=%s source type mismatch: src=%d type=%s", l, m.Config.LayerTypes[l], layer.KVSourceLayer, m.Config.LayerTypes[layer.KVSourceLayer])
		}
	}

	kvCacheK := make([][]float32, len(m.Layers))
	for step := 0; step < 2; step++ {
		for l, layer := range m.Layers {
			layerKVDim := m.Config.NumKVHeads * layer.HeadDimLocal
			if layer.HasKV {
				kvCacheK[l] = append(kvCacheK[l], make([]float32, layerKVDim)...)
			}
			kvLayer := l
			if !layer.HasKV {
				kvLayer = layer.KVSourceLayer
			}
			wantLen := (step + 1) * layerKVDim
			if got := len(kvCacheK[kvLayer]); got != wantLen {
				t.Fatalf("step=%d layer=%d kvLayer=%d len=%d want=%d", step, l, kvLayer, got, wantLen)
			}
			if !layer.HasKV && len(kvCacheK[l]) != 0 {
				t.Fatalf("shared layer %d should not append its own KV cache", l)
			}
		}
	}
}

func TestGemma4KVSharingGPU(t *testing.T) {
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !gpu.Available() {
		t.Skip("GPU not available")
	}

	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4: %v", err)
	}
	if m.Config.ModelType != "gemma4_text" {
		t.Skipf("not gemma4_text: %s", m.Config.ModelType)
	}

	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}

	firstKVShared := m.Config.NumLayers - m.Config.NumKVSharedLayers
	for l, layer := range m.Layers {
		if l < firstKVShared || layer.HasKV {
			continue
		}
		src := layer.KVSourceLayer
		if src < 0 || src >= firstKVShared {
			t.Fatalf("layer %d invalid source %d", l, src)
		}
		selected := g.kvGPU_K[src]
		if selected == nil || selected.GPUPtr() == nil {
			t.Fatalf("layer %d source buffer missing for src=%d", l, src)
		}
		if g.kvGPU_K[l] == nil || g.kvGPU_K[l].GPUPtr() == nil {
			t.Fatalf("layer %d own GPU KV buffer missing", l)
		}
		if selected.GPUPtr().Ptr == g.kvGPU_K[l].GPUPtr().Ptr {
			t.Fatalf("layer %d unexpectedly uses its own KV buffer instead of source %d", l, src)
		}
		expectSrc := 13
		if m.Config.LayerTypes[l] == "full_attention" {
			expectSrc = 14
		}
		if src != expectSrc {
			t.Fatalf("layer %d type=%s source=%d want=%d", l, m.Config.LayerTypes[l], src, expectSrc)
		}
	}
}

func TestGemma4PerLayerInputGatingGPUBuffers(t *testing.T) {
	dir := gemma4Path()
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("model not found: %s", dir)
	}
	if !gpu.Available() {
		t.Skip("GPU not available")
	}

	oldForce := ForceOnTheFly
	ForceOnTheFly = true
	defer func() { ForceOnTheFly = oldForce }()

	m, err := LoadLlama(dir)
	if err != nil {
		t.Fatalf("load gemma4: %v", err)
	}
	g, err := LoadGPUModel(m)
	if err != nil {
		t.Fatalf("LoadGPUModel: %v", err)
	}

	if g.perLayerModelProj == nil || g.perLayerModelProj.GPUPtr() == nil {
		t.Fatal("perLayerModelProj not uploaded to GPU")
	}
	if g.perLayerProjNorm == nil || g.perLayerProjNorm.GPUPtr() == nil {
		t.Fatal("perLayerProjNorm not uploaded to GPU")
	}
	if g.perLayerProjBuf == nil || g.perLayerProjBuf.GPUPtr() == nil {
		t.Fatal("perLayerProjBuf work buffer not on GPU")
	}
	if g.pliGateBuf == nil || g.pliGateBuf.GPUPtr() == nil {
		t.Fatal("pliGateBuf work buffer not on GPU")
	}
	if g.pliProjBuf == nil || g.pliProjBuf.GPUPtr() == nil {
		t.Fatal("pliProjBuf work buffer not on GPU")
	}
	for i, layer := range g.Layers {
		if m.Layers[i].PLIGate == nil {
			continue
		}
		if layer.PLIGate == nil || layer.PLIGate.GPUPtr() == nil {
			t.Fatalf("layer %d PLIGate not uploaded to GPU", i)
		}
		if layer.PLIProj == nil || layer.PLIProj.GPUPtr() == nil {
			t.Fatalf("layer %d PLIProj not uploaded to GPU", i)
		}
		if layer.PLIPostNorm == nil || layer.PLIPostNorm.GPUPtr() == nil {
			t.Fatalf("layer %d PLIPostNorm not uploaded to GPU", i)
		}
		break
	}
}
