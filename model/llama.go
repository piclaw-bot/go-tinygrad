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
}

// LlamaModel holds loaded weights for a LLaMA-style decoder.
type LlamaModel struct {
	Config LlamaConfig

	EmbedTokens *tensor.Tensor // [vocab, hidden]
	Norm        *tensor.Tensor // [hidden]
	LMHead      *tensor.Tensor // [vocab, hidden] (may share with embed)

	Layers []LlamaLayer

	// Pre-computed RoPE frequencies
	RopeFreqs []float32 // [maxSeqLen, headDim/2, 2] (cos, sin interleaved)
}

// LlamaLayer holds weights for one decoder layer.
type LlamaLayer struct {
	InputNorm *tensor.Tensor // [hidden] RMSNorm weight
	PostNorm  *tensor.Tensor // [hidden]

	QW, KW, VW, OW *tensor.Tensor // pre-transposed
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

	f, err := safetensors.Open(dir + "/model.safetensors")
	if err != nil {
		return nil, err
	}

	m := &LlamaModel{Config: cfg}
	h := cfg.HiddenSize

	load := func(name string, shape []int) *tensor.Tensor {
		data, _, err := f.GetFloat32(name)
		if err != nil {
			panic(fmt.Sprintf("load %s: %v", name, err))
		}
		return tensor.FromFloat32(data, shape)
	}
	loadT := func(name string, shape []int) *tensor.Tensor {
		return load(name, shape).Transpose2D()
	}

	m.EmbedTokens = load("model.embed_tokens.weight", []int{cfg.VocabSize, h})
	m.Norm = load("model.norm.weight", []int{h})

	// LM head: often tied to embed_tokens
	if _, ok := f.Tensors["lm_head.weight"]; ok {
		m.LMHead = load("lm_head.weight", []int{cfg.VocabSize, h})
	} else {
		m.LMHead = m.EmbedTokens // tied weights
	}

	kvDim := h / cfg.NumHeads * cfg.NumKVHeads

	m.Layers = make([]LlamaLayer, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		p := fmt.Sprintf("model.layers.%d", l)
		m.Layers[l] = LlamaLayer{
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
			gemv(q, hidden, layer.QW.Data(), h, h)
			gemv(k, hidden, layer.KW.Data(), h, kvDim)
			gemv(v, hidden, layer.VW.Data(), h, kvDim)

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
			gemv(oOut, attnOut, layer.OW.Data(), h, h)

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
			gemv(gate, hidden, layer.GateW.Data(), h, inter)
			gemv(up, hidden, layer.UpW.Data(), h, inter)

			// SiLU(gate) * up
			for i := range gate {
				gate[i] = gate[i] / (1 + float32(math.Exp(float64(-gate[i])))) * up[i]
			}

			// Down projection
			down := make([]float32, h)
			gemv(down, gate, layer.DownW.Data(), inter, h)

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

// gemv: out = x @ wT where wT is pre-transposed [inDim, outDim]
func gemv(out, x, wT []float32, inDim, outDim int) {
	if simd.HasSgemmAsm {
		for i := range out {
			out[i] = 0
		}
		simd.SgemmNN(1, outDim, inDim, 1.0,
			unsafe.Pointer(&x[0]), unsafe.Pointer(&wT[0]), unsafe.Pointer(&out[0]),
			inDim, outDim, outDim)
	} else {
		for j := 0; j < outDim; j++ {
			sum := float32(0)
			for p := 0; p < inDim; p++ {
				sum += x[p] * wT[p*outDim+j]
			}
			out[j] = sum
		}
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
