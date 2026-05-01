package model

import (
	"encoding/json"
	"fmt"
	
	"math"
	"os"
	"unsafe"

	"github.com/rcarmo/go-tinygrad/safetensors"
	"github.com/rcarmo/go-tinygrad/simd"
	"github.com/rcarmo/go-tinygrad/tensor"
)

// LlamaConfig holds model hyperparameters.
// QuantWeight holds GPTQ INT4 weight data for on-the-fly dequantization.
type QuantWeight struct {
	QWeight []int32   // [inDim/8, outDim] packed
	GIdx    []int32   // [inDim] group index
	Scales  []float32 // [numGroups, outDim]
	InDim   int
	OutDim  int
}

type LlamaConfig struct {
	VocabSize      int     `json:"vocab_size"`
	HiddenSize     int     `json:"hidden_size"`
	Intermediate   int     `json:"intermediate_size"`
	NumLayers      int     `json:"num_hidden_layers"`
	NumHeads       int     `json:"num_attention_heads"`
	NumKVHeads     int     `json:"num_key_value_heads"`
	MaxSeqLen      int     `json:"max_position_embeddings"`
	RopeTheta      float64 `json:"rope_theta"`
	RMSNormEps     float64 `json:"rms_norm_eps"`

	// Quantization (populated from quantize_config.json)
	QuantBits     int  `json:"-"`
	QuantGroup    int  `json:"-"`
	QuantSym      bool `json:"-"`
}

// LlamaModel holds loaded weights for a LLaMA-style decoder.
type LlamaModel struct {
	Config LlamaConfig

	EmbedTokens *tensor.Tensor // [vocab, hidden]
	Norm        *tensor.Tensor // [hidden]
	LMHead      *tensor.Tensor // [vocab, hidden] (may share with embed)

	Layers []LlamaLayer

	// Pre-computed RoPE frequencies
	RopeFreqs []float32
	Large     bool // true if weights are NOT pre-transposed
	Quantized     bool // true if using GPTQ INT4 weights
	OnTheFlyQuant bool // true = keep INT4 in memory, dequant per token (slow but low memory) // [maxSeqLen, headDim/2, 2] (cos, sin interleaved)
}

// LlamaLayer holds weights for one decoder layer.
type LlamaLayer struct {
	InputNorm *tensor.Tensor // [hidden] RMSNorm weight
	PostNorm  *tensor.Tensor // [hidden]

	QW, KW, VW, OW *tensor.Tensor // pre-transposed
	QB, KB, VB     *tensor.Tensor // optional biases (Qwen2 has these)

	// GPTQ INT4 quantized weights (nil if not quantized)
	QWq, KWq, VWq, OWq         *QuantWeight
	GateWq, UpWq, DownWq       *QuantWeight
	GateW, UpW, DownW *tensor.Tensor // pre-transposed
}

// LoadLlama loads a LLaMA-style model from safetensors + config.json.
func LoadLlama(dir string) (*LlamaModel, error) {
	// Load config
	cfgData, err := os.ReadFile(dir + "/config.json")
	if err != nil {
		return nil, err
	}
	var cfg LlamaConfig
	if err := json.Unmarshal(cfgData, &cfg); err != nil {
		return nil, err
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-5
	}

	// Try sharded first, then single file
	type loader interface {
		GetFloat32(name string) ([]float32, []int, error)
		GetInt32(name string) ([]int32, []int, error)
		GetRaw(name string) ([]byte, string, []int, error)
	}
	var f loader
	if _, err := os.Stat(dir + "/model.safetensors.index.json"); err == nil {
		sf, err := safetensors.OpenSharded(dir + "/model.safetensors.index.json")
		if err != nil {
			return nil, fmt.Errorf("open sharded: %w", err)
		}
		f = sf
	} else {
		sf, err := safetensors.Open(dir + "/model.safetensors")
		if err != nil {
			return nil, fmt.Errorf("open single: %w", err)
		}
		f = sf
	}

	// Try loading quantization config
	if qcData, err := os.ReadFile(dir + "/quantize_config.json"); err == nil {
		var qc struct {
			Bits      int  `json:"bits"`
			GroupSize int  `json:"group_size"`
			Sym       bool `json:"sym"`
		}
		if err := json.Unmarshal(qcData, &qc); err == nil && qc.Bits > 0 {
			cfg.QuantBits = qc.Bits
			cfg.QuantGroup = qc.GroupSize
			cfg.QuantSym = qc.Sym
			fmt.Printf("  GPTQ: %d-bit, group=%d, sym=%v\n", qc.Bits, qc.GroupSize, qc.Sym)
		}
	}

	m := &LlamaModel{Config: cfg}
	h := cfg.HiddenSize
	// For large models (>2B params), skip pre-transpose to save memory
	large := cfg.HiddenSize >= 3000
	m.Large = large
	m.Quantized = cfg.QuantBits > 0
	// Heuristic: dequant at load if enough RAM, on-the-fly for very large models
	// F32 dequant needs ~4× model_params bytes. For 7B = ~28GB.
	// On-the-fly keeps INT4 packed (~4GB for 7B) but inference is 20× slower.
	onTheFly := false // default: dequant at load for speed
	m.OnTheFlyQuant = onTheFly

	load := func(name string, shape []int) *tensor.Tensor {
		data, _, err := f.GetFloat32(name)
		if err != nil {
			panic(fmt.Sprintf("load %s: %v", name, err))
		}
		return tensor.FromFloat32(data, shape)
	}
	loadT := func(name string, shape []int) *tensor.Tensor {
		if large {
			return load(name, shape) // keep original layout, use NT path
		}
		return load(name, shape).Transpose2D()
	}

	// loadQW loads raw GPTQ quantized weight without dequantization.
	loadQW := func(name string, outDim, inDim int) *QuantWeight {
		qw, _, err := f.GetInt32(name + ".qweight")
		if err != nil {
			panic(fmt.Sprintf("loadQW %s.qweight: %v", name, err))
		}
		gIdx, _, err := f.GetInt32(name + ".g_idx")
		if err != nil {
			panic(fmt.Sprintf("loadQW %s.g_idx: %v", name, err))
		}
		scRaw, scDtype, _, err := f.GetRaw(name + ".scales")
		if err != nil {
			panic(fmt.Sprintf("loadQW %s.scales: %v", name, err))
		}
		var scales []float32
		if scDtype == "F16" {
			n := len(scRaw) / 2
			scales = make([]float32, n)
			for i := 0; i < n; i++ {
				h := uint16(scRaw[i*2]) | uint16(scRaw[i*2+1])<<8
				scales[i] = float16ToFloat32(h)
			}
		} else {
			scales, _, _ = f.GetFloat32(name + ".scales")
		}
		return &QuantWeight{QWeight: qw, GIdx: gIdx, Scales: scales, InDim: inDim, OutDim: outDim}
	}
	_ = loadQW

	// loadQ loads a GPTQ quantized weight, dequantizes to F32.
	// name is the base name (e.g. "model.layers.0.mlp.gate_proj")
	// shape is [outFeatures, inFeatures] (the original weight shape)
	loadQ := func(name string, outDim, inDim int) *tensor.Tensor {
		qw, _, err := f.GetInt32(name + ".qweight")
		if err != nil {
			panic(fmt.Sprintf("loadQ %s.qweight: %v", name, err))
		}
		gIdx, _, err := f.GetInt32(name + ".g_idx")
		if err != nil {
			panic(fmt.Sprintf("loadQ %s.g_idx: %v", name, err))
		}
		// Scales are F16, load raw and convert
		scRaw, scDtype, _, err := f.GetRaw(name + ".scales")
		if err != nil {
			panic(fmt.Sprintf("loadQ %s.scales: %v", name, err))
		}
		var scales []float32
		if scDtype == "F16" {
			n := len(scRaw) / 2
			scales = make([]float32, n)
			for i := 0; i < n; i++ {
				h := uint16(scRaw[i*2]) | uint16(scRaw[i*2+1])<<8
				scales[i] = float16ToFloat32(h)
			}
		} else {
			scales, _, _ = f.GetFloat32(name + ".scales")
		}

		var data []float32
		if cfg.QuantSym {
			data = DequantGPTQSym(qw, gIdx, scales, inDim, outDim)
		} else {
			qz, _, _ := f.GetInt32(name + ".qzeros")
			data = DequantGPTQ(qw, qz, gIdx, scales, inDim, outDim, false)
		}
		// data is [outDim, inDim] row-major
		t := tensor.FromFloat32(data, []int{outDim, inDim})
		if !large {
			t = t.Transpose2D() // pre-transpose for NN path
		}
		return t
	}
	_ = loadQ

	m.EmbedTokens = load("model.embed_tokens.weight", []int{cfg.VocabSize, h})
	m.Norm = load("model.norm.weight", []int{h})

	// LM head: often tied to embed_tokens
	if _, _, err := f.GetFloat32("lm_head.weight"); err == nil {
		m.LMHead = load("lm_head.weight", []int{cfg.VocabSize, h})
	} else {
		m.LMHead = m.EmbedTokens // tied weights
	}

	kvDim := h / cfg.NumHeads * cfg.NumKVHeads

	// tryLoad checks if a tensor exists
	tryLoad := func(name string) bool {
		_, _, err := f.GetFloat32(name)
		return err == nil
	}
	_ = tryLoad

	m.Layers = make([]LlamaLayer, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		p := fmt.Sprintf("model.layers.%d", l)
		var layer LlamaLayer
		if cfg.QuantBits > 0 && onTheFly {
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QWq:    loadQW(p+".self_attn.q_proj", h, h),
				KWq:    loadQW(p+".self_attn.k_proj", kvDim, h),
				VWq:    loadQW(p+".self_attn.v_proj", kvDim, h),
				OWq:    loadQW(p+".self_attn.o_proj", h, h),
				GateWq: loadQW(p+".mlp.gate_proj", cfg.Intermediate, h),
				UpWq:   loadQW(p+".mlp.up_proj", cfg.Intermediate, h),
				DownWq: loadQW(p+".mlp.down_proj", h, cfg.Intermediate),
			}
		} else if cfg.QuantBits > 0 {
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QW: loadQ(p+".self_attn.q_proj", h, h),
				KW: loadQ(p+".self_attn.k_proj", kvDim, h),
				VW: loadQ(p+".self_attn.v_proj", kvDim, h),
				OW: loadQ(p+".self_attn.o_proj", h, h),
				GateW: loadQ(p+".mlp.gate_proj", cfg.Intermediate, h),
				UpW:   loadQ(p+".mlp.up_proj", cfg.Intermediate, h),
				DownW: loadQ(p+".mlp.down_proj", h, cfg.Intermediate),
			}
		} else {
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QW: loadT(p+".self_attn.q_proj.weight", []int{h, h}),
				KW: loadT(p+".self_attn.k_proj.weight", []int{kvDim, h}),
				VW: loadT(p+".self_attn.v_proj.weight", []int{kvDim, h}),
				OW: loadT(p+".self_attn.o_proj.weight", []int{h, h}),
				GateW: loadT(p+".mlp.gate_proj.weight", []int{cfg.Intermediate, h}),
				UpW:   loadT(p+".mlp.up_proj.weight", []int{cfg.Intermediate, h}),
				DownW: loadT(p+".mlp.down_proj.weight", []int{h, cfg.Intermediate}),
			}
		}
		// Optional Q/K/V biases (Qwen2 has these, LLaMA doesn't)
		if tryLoad(p+".self_attn.q_proj.bias") {
			layer.QB = load(p+".self_attn.q_proj.bias", []int{h})
			layer.KB = load(p+".self_attn.k_proj.bias", []int{kvDim})
			layer.VB = load(p+".self_attn.v_proj.bias", []int{kvDim})
		}
		m.Layers[l] = layer
	}

	// Pre-compute RoPE frequencies
	m.precomputeRoPE()

	return m, nil
}

func (m *LlamaModel) precomputeRoPE() {
	cfg := m.Config
	headDim := cfg.HiddenSize / cfg.NumHeads
	halfDim := headDim / 2
	maxSeq := cfg.MaxSeqLen
	if maxSeq > 2048 {
		maxSeq = 2048 // cap for memory
	}

	m.RopeFreqs = make([]float32, maxSeq*halfDim*2)
	theta := cfg.RopeTheta
	for pos := 0; pos < maxSeq; pos++ {
		for i := 0; i < halfDim; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq
			off := (pos*halfDim + i) * 2
			m.RopeFreqs[off] = float32(math.Cos(angle))
			m.RopeFreqs[off+1] = float32(math.Sin(angle))
		}
	}
}

// Generate produces tokens autoregressively.
func (m *LlamaModel) mvQ(out, x []float32, qw *QuantWeight) {
	if qw != nil {
		gemvQ4Sym(out, x, qw.QWeight, qw.GIdx, qw.Scales, qw.InDim, qw.OutDim)
	}
}

func (m *LlamaModel) mv(out, x, w []float32, inDim, outDim int) {
	if m.Large {
		gemvNT(out, x, w, inDim, outDim)
	} else {
		gemv(out, x, w, inDim, outDim)
	}
}

func (m *LlamaModel) Generate(tokenIDs []int, maxTokens int) []int {
	cfg := m.Config
	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	headDim := h / numHeads
	kvDim := headDim * numKVHeads
	inter := cfg.Intermediate

	// Allocate KV cache
	kvCacheK := make([][]float32, cfg.NumLayers) // [layers][seqLen * kvDim]
	kvCacheV := make([][]float32, cfg.NumLayers)
	for l := range kvCacheK {
		kvCacheK[l] = make([]float32, 0, 2048*kvDim)
		kvCacheV[l] = make([]float32, 0, 2048*kvDim)
	}

	output := make([]int, len(tokenIDs), len(tokenIDs)+maxTokens)
	copy(output, tokenIDs)

	// Process prompt + generate
	for step := 0; step < len(tokenIDs)+maxTokens-1; step++ {
		var tokID int
		if step < len(tokenIDs) {
			tokID = tokenIDs[step]
		} else {
			tokID = output[len(output)-1]
		}

		// Embed single token
		embData := m.EmbedTokens.Data()
		hidden := make([]float32, h)
		copy(hidden, embData[tokID*h:(tokID+1)*h])

		pos := step

		for l := 0; l < cfg.NumLayers; l++ {
			layer := &m.Layers[l]
			residual := make([]float32, h)
			copy(residual, hidden)

			// RMS Norm
			rmsNormInPlace(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))

			// Q, K, V projections (single token: [1, h] @ [h, dim])
			q := make([]float32, h)
			k := make([]float32, kvDim)
			v := make([]float32, kvDim)
			if layer.QWq != nil {
				m.mvQ(q, hidden, layer.QWq)
				m.mvQ(k, hidden, layer.KWq)
				m.mvQ(v, hidden, layer.VWq)
			} else {
				m.mv(q, hidden, layer.QW.Data(), h, h)
				m.mv(k, hidden, layer.KW.Data(), h, kvDim)
				m.mv(v, hidden, layer.VW.Data(), h, kvDim)
			}

			// Add bias if present (Qwen2)
			if layer.QB != nil {
				qb, kb, vb := layer.QB.Data(), layer.KB.Data(), layer.VB.Data()
				for i := range q { q[i] += qb[i] }
				for i := range k { k[i] += kb[i] }
				for i := range v { v[i] += vb[i] }
			}

			// RoPE on Q and K
			applyRoPE(q, m.RopeFreqs, pos, numHeads, headDim)
			applyRoPE(k, m.RopeFreqs, pos, numKVHeads, headDim)

			// Append to KV cache
			kvCacheK[l] = append(kvCacheK[l], k...)
			kvCacheV[l] = append(kvCacheV[l], v...)

			// Attention: Q against all cached K, V
			seqLen := pos + 1
			attnOut := gqaAttention(q, kvCacheK[l], kvCacheV[l], seqLen, numHeads, numKVHeads, headDim)

			// Output projection
			oOut := make([]float32, h)
			if layer.OWq != nil {
				m.mvQ(oOut, attnOut, layer.OWq)
			} else {
				m.mv(oOut, attnOut, layer.OW.Data(), h, h)
			}

			// Residual
			for i := range hidden {
				hidden[i] = residual[i] + oOut[i]
			}

			// Post-attention norm
			copy(residual, hidden)
			rmsNormInPlace(hidden, layer.PostNorm.Data(), float32(cfg.RMSNormEps))

			// MLP: gate * up → SiLU → down
			gate := make([]float32, inter)
			up := make([]float32, inter)
			if layer.GateWq != nil {
				m.mvQ(gate, hidden, layer.GateWq)
				m.mvQ(up, hidden, layer.UpWq)
			} else {
				m.mv(gate, hidden, layer.GateW.Data(), h, inter)
				m.mv(up, hidden, layer.UpW.Data(), h, inter)
			}

			// SiLU(gate) * up
			for i := range gate {
				gate[i] = gate[i] / (1 + float32(math.Exp(float64(-gate[i])))) * up[i]
			}

			// Down projection
			down := make([]float32, h)
			if layer.DownWq != nil {
				m.mvQ(down, gate, layer.DownWq)
			} else {
				m.mv(down, gate, layer.DownW.Data(), inter, h)
			}

			// Residual
			for i := range hidden {
				hidden[i] = residual[i] + down[i]
			}
		}

		// Final norm
		rmsNormInPlace(hidden, m.Norm.Data(), float32(cfg.RMSNormEps))

		// LM head: logits = hidden @ lm_head^T (greedy: take argmax)
		if step >= len(tokenIDs)-1 {
			logits := make([]float32, cfg.VocabSize)
			lmData := m.LMHead.Data()
			for v := 0; v < cfg.VocabSize; v++ {
				sum := float32(0)
				row := lmData[v*h : (v+1)*h]
				if h >= 8 {
					sum = simd.Sdot(hidden, row)
				} else {
					for d := 0; d < h; d++ {
						sum += hidden[d] * row[d]
					}
				}
				logits[v] = sum
			}

			// Argmax
			maxIdx := 0
			maxVal := logits[0]
			for i, v := range logits[1:] {
				if v > maxVal {
					maxVal = v
					maxIdx = i + 1
				}
			}
			output = append(output, maxIdx)
		}
	}
	return output
}

// --- Low-level helpers ---

func rmsNormInPlace(x, weight []float32, eps float32) {
	h := len(x)
	ss := float32(0)
	for _, v := range x {
		ss += v * v
	}
	ss = float32(1.0 / math.Sqrt(float64(ss/float32(h)+eps)))
	for i := range x {
		x[i] = weight[i] * x[i] * ss
	}
}

// gemv: out = x @ w where w is either:
//   pre-transposed [inDim, outDim] (use NN), or
//   original [outDim, inDim] (use NT via dot products)
func gemv(out, x []float32, w []float32, inDim, outDim int) {
	for i := range out {
		out[i] = 0
	}
	if len(w) == inDim*outDim {
		// Detect layout: if w is [inDim, outDim] (pre-transposed), use NN
		// If w is [outDim, inDim] (original), use NT (dot per output)
		// Heuristic: try NN first (pre-transposed path)
		if simd.HasSgemmAsm {
			simd.SgemmNN(1, outDim, inDim, 1.0,
				unsafe.Pointer(&x[0]), unsafe.Pointer(&w[0]), unsafe.Pointer(&out[0]),
				inDim, outDim, outDim)
		} else {
			for j := 0; j < outDim; j++ {
				sum := float32(0)
				for p := 0; p < inDim; p++ {
					sum += x[p] * w[p*outDim+j]
				}
				out[j] = sum
			}
		}
	}
}

// gemvNT: out = x @ w^T where w is [outDim, inDim] (original layout)
func gemvNT(out, x []float32, w []float32, inDim, outDim int) {
	for j := 0; j < outDim; j++ {
		sum := float32(0)
		row := w[j*inDim : (j+1)*inDim]
		if inDim >= 8 {
			sum = simd.Sdot(x, row)
		} else {
			for p := 0; p < inDim; p++ {
				sum += x[p] * row[p]
			}
		}
		out[j] = sum
	}
}

func applyRoPE(x, freqs []float32, pos, numHeads, headDim int) {
	halfDim := headDim / 2
	for h := 0; h < numHeads; h++ {
		for i := 0; i < halfDim; i++ {
			freqOff := (pos*halfDim + i) * 2
			cos := freqs[freqOff]
			sin := freqs[freqOff+1]
			idx0 := h*headDim + i
			idx1 := h*headDim + i + halfDim
			x0 := x[idx0]
			x1 := x[idx1]
			x[idx0] = x0*cos - x1*sin
			x[idx1] = x0*sin + x1*cos
		}
	}
}

func gqaAttention(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int) []float32 {
	h := numHeads * headDim
	kvDim := numKVHeads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	headsPerKV := numHeads / numKVHeads

	out := make([]float32, h)

	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV

		// Compute attention scores for this head against all cached K
		scores := make([]float32, seqLen)
		for t := 0; t < seqLen; t++ {
			sum := float32(0)
			for d := 0; d < headDim; d++ {
				sum += q[head*headDim+d] * kCache[t*kvDim+kvHead*headDim+d]
			}
			scores[t] = sum * scale
		}

		// Causal softmax (all positions are visible for single-token decode)
		mx := scores[0]
		for _, v := range scores[1:] {
			if v > mx {
				mx = v
			}
		}
		expSum := float32(0)
		for i := range scores {
			scores[i] = float32(math.Exp(float64(scores[i] - mx)))
			expSum += scores[i]
		}
		inv := 1.0 / expSum
		for i := range scores {
			scores[i] *= inv
		}

		// Weighted sum of V
		for d := 0; d < headDim; d++ {
			sum := float32(0)
			for t := 0; t < seqLen; t++ {
				sum += scores[t] * vCache[t*kvDim+kvHead*headDim+d]
			}
			out[head*headDim+d] = sum
		}
	}
	return out
}
