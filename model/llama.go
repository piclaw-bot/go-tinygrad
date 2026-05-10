package model

import (
	"encoding/json"
	"fmt"

	"math"
	"os"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/safetensors"
	"github.com/rcarmo/go-pherence/simd"
	"github.com/rcarmo/go-pherence/tensor"
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
	VocabSize            int      `json:"vocab_size"`
	HiddenSize           int      `json:"hidden_size"`
	Intermediate         int      `json:"intermediate_size"`
	NumLayers            int      `json:"num_hidden_layers"`
	NumHeads             int      `json:"num_attention_heads"`
	NumKVHeads           int      `json:"num_key_value_heads"`
	MaxSeqLen            int      `json:"max_position_embeddings"`
	RopeTheta            float64  `json:"rope_theta"`
	RMSNormEps           float64  `json:"rms_norm_eps"`
	ModelType            string   `json:"model_type"`
	TieEmbeddings        bool     `json:"tie_word_embeddings"`
	HeadDim              int      `json:"head_dim"`
	SlidingWindow        int      `json:"sliding_window"`
	SlidingWindowPattern int      `json:"sliding_window_pattern"`
	RopeLocalBaseFreq    float64  `json:"rope_local_base_freq"`
	BOSTokenID           int      `json:"bos_token_id"`
	LayerTypes           []string `json:"layer_types"`
	NumKVSharedLayers    int      `json:"num_kv_shared_layers"`
	GlobalHeadDim        int      `json:"global_head_dim"`
	HiddenPerLayer       int      `json:"hidden_size_per_layer_input"`
	VocabPerLayer        int      `json:"vocab_size_per_layer_input"`
	TensorPrefix         string   `json:"-"` // "language_model.model." for Gemma4
	HiddenAct            string   `json:"hidden_activation"`

	// Quantization (populated from quantize_config.json or config.json)
	QuantBits   int    `json:"-"`
	QuantGroup  int    `json:"-"`
	QuantSym    bool   `json:"-"`
	QuantFormat string `json:"-"` // "gptq" or "mlx"

	// MoE (Mixture of Experts)
	NumExperts        int  `json:"num_experts"`
	NumExpertsPerTok  int  `json:"num_experts_per_tok"`
	MoEIntermediate   int  `json:"moe_intermediate_size"`
	DecoderSparseStep int  `json:"decoder_sparse_step"`
	NormTopKProb      bool `json:"norm_topk_prob"`
}

// LlamaModel holds loaded weights for a LLaMA-style decoder.
type LlamaModel struct {
	Config LlamaConfig
	Tok    *Tokenizer // optional, for chat templates

	EmbedTokens *tensor.Tensor // [vocab, hidden]

	// Gemma4 per-layer input gating (model-level weights)
	EmbedPerLayer      []float32      // [vocabPerLayer, numLayers * hiddenPerLayer] dequantized
	PerLayerModelProj  []float32      // [numLayers * hiddenPerLayer, hidden] F32
	PerLayerProjNorm   []float32      // [hiddenPerLayer] F32
	PerLayerInputScale float32        // 2^-0.5
	PerLayerProjScale  float32        // hidden^-0.5
	EmbedPerLayerScale float32        // hiddenPerLayer^0.5
	Norm               *tensor.Tensor // [hidden]
	LMHead             *tensor.Tensor // [vocab, hidden] (may share with embed)

	Layers []LlamaLayer

	// Pre-computed RoPE frequencies
	RopeFreqs     []float32
	RopeFreqsSWA  []float32 // Gemma4: sliding window RoPE (theta=10000, full rotation)
	RopeFreqsFull []float32 // Gemma4: full attention RoPE (theta=1M, partial rotation)
	RopeHalfSWA   int       // half-dim for SWA RoPE
	RopeHalfFull  int       // half-dim for full attention RoPE
	Large         bool      // true if weights are NOT pre-transposed
	Quantized     bool      // true if using GPTQ INT4 weights
	OnTheFlyQuant bool      // true = keep INT4 in memory, dequant per token (slow but low memory) // [maxSeqLen, headDim/2, 2] (cos, sin interleaved)

	// TurboQuant controls optional CPU KV cache compression. State is keyed by
	// headDim because Gemma4 uses per-layer head dimensions, so each distinct
	// headDim needs its own orthogonal rotation matrix.
	EnableTurboQuant bool
	TurboQuantStates map[int]*TurboQuantState
}

// LlamaLayer holds weights for one decoder layer.
type LlamaLayer struct {
	InputNorm     *tensor.Tensor // [hidden] RMSNorm weight
	PostNorm      *tensor.Tensor // [hidden]
	PreFFNNorm    *tensor.Tensor // [hidden] Gemma3: pre-feedforward norm
	VNorm         *tensor.Tensor // Gemma4: V projection norm
	LayerScalar   float32        // Gemma4: per-layer output scaling
	HeadDimLocal  int            // per-layer head dim (may differ from config)
	HasKV         bool           // Gemma4: false for KV-sharing layers
	KVSourceLayer int            // Gemma4: which layer to share KV from

	// Gemma4 per-layer input gating (per-layer weights)
	PLIGate     []float32      // [hiddenPerLayer, hidden] dequantized
	PLIProj     []float32      // [hidden, hiddenPerLayer] dequantized
	PLIPostNorm []float32      // [hidden]
	PostFFNNorm *tensor.Tensor // [hidden] Gemma3: post-feedforward norm

	QW, KW, VW, OW *tensor.Tensor // pre-transposed
	QB, KB, VB     *tensor.Tensor // optional biases (Qwen2 has these)
	QNorm, KNorm   *tensor.Tensor // optional QK-Norm (Qwen3 has these)

	// GPTQ INT4 quantized weights (nil if not quantized)
	QWq, KWq, VWq, OWq   *QuantWeight
	GateWq, UpWq, DownWq *QuantWeight

	// MLX affine quantized weights (nil if not MLX)
	QWm, KWm, VWm, OWm   *MLXQuantWeight
	GateWm, UpWm, DownWm *MLXQuantWeight

	GateW, UpW, DownW *tensor.Tensor // pre-transposed

	// MoE (Mixture of Experts)
	IsMoE       bool              // true if this layer uses MoE
	RouterW     *MLXQuantWeight   // router gate weight [numExperts, hidden]
	ExpertGateW []*MLXQuantWeight // [numExperts] gate projections
	ExpertUpW   []*MLXQuantWeight // [numExperts] up projections
	ExpertDownW []*MLXQuantWeight // [numExperts] down projections
}

// LoadLlama loads a LLaMA-style model from safetensors + config.json.
// ForceOnTheFly controls whether quantized models keep INT4 packed weights.
// Set to true before LoadLlama when using GPU forward pass (GPU Q4 GEMV needs packed weights).
var ForceOnTheFly bool

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
	// Gemma4: text config is nested under text_config
	if cfg.HiddenSize == 0 {
		var nested struct {
			TextConfig LlamaConfig `json:"text_config"`
			ModelType  string      `json:"model_type"`
		}
		if err := json.Unmarshal(cfgData, &nested); err == nil && nested.TextConfig.HiddenSize > 0 {
			// Preserve top-level model_type
			outerType := nested.ModelType
			cfg = nested.TextConfig
			if cfg.ModelType == "" {
				cfg.ModelType = outerType + "_text"
			}
		}
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-5
	}
	// hidden_act fallback: some models use "hidden_act" instead of "hidden_activation"
	if cfg.HiddenAct == "" {
		var actFallback struct {
			HiddenAct string `json:"hidden_act"`
		}
		if err := json.Unmarshal(cfgData, &actFallback); err == nil && actFallback.HiddenAct != "" {
			cfg.HiddenAct = actFallback.HiddenAct
		}
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	// Gemma4: infer sliding_window_pattern from layer_types
	if len(cfg.LayerTypes) > 0 && cfg.SlidingWindowPattern == 0 {
		for i, lt := range cfg.LayerTypes {
			if lt == "full_attention" {
				cfg.SlidingWindowPattern = i + 1
				break
			}
		}
	}

	// Detect tensor prefix (Gemma4 uses "language_model.model.")
	cfg.TensorPrefix = ""
	if cfg.ModelType == "gemma4_text" || cfg.ModelType == "gemma4" {
		cfg.TensorPrefix = "language_model."
	}

	// Try sharded first, then single file
	type loader interface {
		GetFloat32(name string) ([]float32, []int, error)
		GetInt32(name string) ([]int32, []int, error)
		GetRaw(name string) ([]byte, string, []int, error)
		Close() error
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
	defer f.Close()

	if os.Getenv("GO_PHERENCE_EAGER_LOAD") == "1" {
		if ef, ok := f.(interface{ EagerLoad() (int64, error) }); ok {
			t0 := time.Now()
			bytes, err := ef.EagerLoad()
			if err != nil {
				return nil, fmt.Errorf("eager load safetensors: %w", err)
			}
			fmt.Printf("  Eager loaded %.1f MB of mmap'd weights in %.2fs\n", float64(bytes)/(1024*1024), time.Since(t0).Seconds())
		}
	}

	// Try loading quantization config (GPTQ)
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
			cfg.QuantFormat = "gptq"
			fmt.Printf("  GPTQ: %d-bit, group=%d, sym=%v\n", qc.Bits, qc.GroupSize, qc.Sym)
		}
	}

	// Try MLX quantization from config.json
	if cfg.QuantBits == 0 {
		var mlxCfg struct {
			Quantization struct {
				Bits      int `json:"bits"`
				GroupSize int `json:"group_size"`
			} `json:"quantization"`
		}
		if err := json.Unmarshal(cfgData, &mlxCfg); err == nil && mlxCfg.Quantization.Bits > 0 {
			cfg.QuantBits = mlxCfg.Quantization.Bits
			cfg.QuantGroup = mlxCfg.Quantization.GroupSize
			cfg.QuantSym = false // MLX uses bias, not symmetric
			cfg.QuantFormat = "mlx"
			fmt.Printf("  MLX: %d-bit, group=%d\n", cfg.QuantBits, cfg.QuantGroup)
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
	// If GPU is available and model is quantized, keep INT4 packed for GPU upload
	// OnTheFlyQuant: keep INT4 packed (for GPU upload) vs dequant-at-load (fast CPU).
	// Set ForceOnTheFly=true before LoadLlama when using GPU forward pass.
	onTheFly := ForceOnTheFly && cfg.QuantBits > 0
	m.OnTheFlyQuant = onTheFly

	prefix := cfg.TensorPrefix
	load := func(name string, shape []int) *tensor.Tensor {
		data, actualShape, err := f.GetFloat32(prefix + name)
		if err != nil && prefix != "" {
			data, actualShape, err = f.GetFloat32(name)
		}
		if err != nil {
			panic(fmt.Sprintf("load %s: %v", name, err))
		}
		// Use actual shape from safetensors if it matches element count
		if len(actualShape) > 0 {
			n := 1
			for _, d := range actualShape {
				n *= d
			}
			if n == len(data) {
				shape = actualShape
			}
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
		qw, _, err := f.GetInt32(prefix + name + ".qweight")
		if err != nil && prefix != "" {
			qw, _, err = f.GetInt32(name + ".qweight")
		}
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

	// loadMLXW loads an MLX affine quantized weight.
	loadMLXW := func(name string, outDim, inDim int) *MLXQuantWeight {
		qw, err := loadMLXWeight(f, prefix+name, outDim, inDim, cfg.QuantGroup, cfg.QuantBits)
		if err != nil && prefix != "" {
			qw, err = loadMLXWeight(f, name, outDim, inDim, cfg.QuantGroup, cfg.QuantBits)
		}
		if err != nil {
			panic(fmt.Sprintf("loadMLXW %s: %v", name, err))
		}
		return qw
	}
	_ = loadMLXW

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

	// Load embeddings — MLX may quantize these
	if cfg.QuantFormat == "mlx" {
		// Try to load quantized embedding, dequantize for lookup
		if emb, err := loadMLXWeight(f, prefix+"model.embed_tokens", cfg.VocabSize, h, cfg.QuantGroup, cfg.QuantBits); err == nil {
			data := DequantMLX(emb)
			m.EmbedTokens = tensor.FromFloat32(data, []int{cfg.VocabSize, h})
		} else {
			m.EmbedTokens = load("model.embed_tokens.weight", []int{cfg.VocabSize, h})
		}
	} else {
		m.EmbedTokens = load("model.embed_tokens.weight", []int{cfg.VocabSize, h})
	}
	m.Norm = load("model.norm.weight", []int{h})

	// LM head: often tied to embed_tokens. MLX may quantize it too.
	if cfg.QuantFormat == "mlx" {
		if lm, err := loadMLXWeight(f, prefix+"lm_head", cfg.VocabSize, h, cfg.QuantGroup, cfg.QuantBits); err == nil {
			data := DequantMLX(lm)
			m.LMHead = tensor.FromFloat32(data, []int{cfg.VocabSize, h})
		} else {
			m.LMHead = m.EmbedTokens // tied weights
		}
	} else if _, _, err := f.GetFloat32("lm_head.weight"); err == nil {
		m.LMHead = load("lm_head.weight", []int{cfg.VocabSize, h})
	} else {
		m.LMHead = m.EmbedTokens // tied weights
	}

	kvDim := cfg.HeadDim * cfg.NumKVHeads

	// tryLoad checks if a tensor exists
	tryLoad := func(name string) bool {
		_, _, _, err := f.GetRaw(prefix + name)
		if err != nil && prefix != "" {
			_, _, _, err = f.GetRaw(name)
		}
		return err == nil
	}
	_ = tryLoad

	m.Layers = make([]LlamaLayer, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		p := fmt.Sprintf("model.layers.%d", l)
		var layer LlamaLayer

		// Per-layer Q/K/V/O dimensions (Gemma4: varies by layer type)
		qDimL := h      // Q output = numHeads * headDim
		kvDimL := kvDim // K/V output = numKVHeads * headDim
		oDimIn := h     // O input = numHeads * headDim
		if len(cfg.LayerTypes) > l {
			lt := cfg.LayerTypes[l]
			var lhd int
			if lt == "full_attention" && cfg.GlobalHeadDim > 0 {
				lhd = cfg.GlobalHeadDim
			} else {
				lhd = cfg.HeadDim
			}
			qDimL = cfg.NumHeads * lhd
			kvDimL = cfg.NumKVHeads * lhd
			oDimIn = qDimL
		}

		// Check if this layer uses MoE (switch_mlp format)
		isMoELayer := cfg.NumExperts > 0 && tryLoad(p+".mlp.switch_mlp.gate_proj.weight")

		if cfg.QuantFormat == "mlx" && onTheFly {
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QWm:       loadMLXW(p+".self_attn.q_proj", qDimL, h),
				KWm:       loadMLXW(p+".self_attn.k_proj", kvDimL, h),
				VWm:       loadMLXW(p+".self_attn.v_proj", kvDimL, h),
				OWm:       loadMLXW(p+".self_attn.o_proj", h, oDimIn),
			}
			if !isMoELayer {
				layer.GateWm = loadMLXW(p+".mlp.gate_proj", cfg.Intermediate, h)
				layer.UpWm = loadMLXW(p+".mlp.up_proj", cfg.Intermediate, h)
				layer.DownWm = loadMLXW(p+".mlp.down_proj", h, cfg.Intermediate)
			}
		} else if cfg.QuantFormat == "mlx" {
			// MLX dequant-at-load
			loadMLXDeq := func(name string, outDim, inDim int) *tensor.Tensor {
				qw := loadMLXW(name, outDim, inDim)
				data := DequantMLX(qw)
				// Use actual dims from loaded weight (may differ from caller's hint)
				if large {
					return tensor.FromFloat32(data, []int{qw.OutDim, qw.InDim})
				}
				return tensor.FromFloat32(data, []int{qw.OutDim, qw.InDim}).Transpose2D()
			}
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QW:        loadMLXDeq(p+".self_attn.q_proj", qDimL, h),
				KW:        loadMLXDeq(p+".self_attn.k_proj", kvDimL, h),
				VW:        loadMLXDeq(p+".self_attn.v_proj", kvDimL, h),
				OW:        loadMLXDeq(p+".self_attn.o_proj", h, oDimIn),
			}
			if !isMoELayer {
				layer.GateW = loadMLXDeq(p+".mlp.gate_proj", cfg.Intermediate, h)
				layer.UpW = loadMLXDeq(p+".mlp.up_proj", cfg.Intermediate, h)
				layer.DownW = loadMLXDeq(p+".mlp.down_proj", h, cfg.Intermediate)
			}
		} else if cfg.QuantBits > 0 && onTheFly {
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QWq:       loadQW(p+".self_attn.q_proj", qDimL, h),
				KWq:       loadQW(p+".self_attn.k_proj", kvDimL, h),
				VWq:       loadQW(p+".self_attn.v_proj", kvDimL, h),
				OWq:       loadQW(p+".self_attn.o_proj", h, oDimIn),
			}
			if !isMoELayer {
				layer.GateWq = loadQW(p+".mlp.gate_proj", cfg.Intermediate, h)
				layer.UpWq = loadQW(p+".mlp.up_proj", cfg.Intermediate, h)
				layer.DownWq = loadQW(p+".mlp.down_proj", h, cfg.Intermediate)
			}
		} else if cfg.QuantBits > 0 {
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QW:        loadQ(p+".self_attn.q_proj", qDimL, h),
				KW:        loadQ(p+".self_attn.k_proj", kvDimL, h),
				VW:        loadQ(p+".self_attn.v_proj", kvDimL, h),
				OW:        loadQ(p+".self_attn.o_proj", h, oDimIn),
			}
			if !isMoELayer {
				layer.GateW = loadQ(p+".mlp.gate_proj", cfg.Intermediate, h)
				layer.UpW = loadQ(p+".mlp.up_proj", cfg.Intermediate, h)
				layer.DownW = loadQ(p+".mlp.down_proj", h, cfg.Intermediate)
			}
		} else {
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QW:        loadT(p+".self_attn.q_proj.weight", []int{qDimL, h}),
				KW:        loadT(p+".self_attn.k_proj.weight", []int{kvDimL, h}),
				VW:        loadT(p+".self_attn.v_proj.weight", []int{kvDimL, h}),
				OW:        loadT(p+".self_attn.o_proj.weight", []int{h, oDimIn}),
			}
			if !isMoELayer {
				layer.GateW = loadT(p+".mlp.gate_proj.weight", []int{cfg.Intermediate, h})
				layer.UpW = loadT(p+".mlp.up_proj.weight", []int{cfg.Intermediate, h})
				layer.DownW = loadT(p+".mlp.down_proj.weight", []int{h, cfg.Intermediate})
			}
		}
		// Optional Q/K/V biases (Qwen2 has these, LLaMA doesn't)
		if tryLoad(p + ".self_attn.q_proj.bias") {
			layer.QB = load(p+".self_attn.q_proj.bias", []int{qDimL})
			layer.KB = load(p+".self_attn.k_proj.bias", []int{kvDimL})
			layer.VB = load(p+".self_attn.v_proj.bias", []int{kvDimL})
		}
		// Optional pre/post FFN norms (Gemma3 has these)
		if tryLoad(p + ".pre_feedforward_layernorm.weight") {
			layer.PreFFNNorm = load(p+".pre_feedforward_layernorm.weight", []int{h})
			layer.PostFFNNorm = load(p+".post_feedforward_layernorm.weight", []int{h})
		}
		// Optional QK-Norm (Qwen3/Gemma3 have these)
		layerHD := qDimL / cfg.NumHeads // per-layer head dim
		if tryLoad(p + ".self_attn.q_norm.weight") {
			layer.QNorm = load(p+".self_attn.q_norm.weight", []int{layerHD})
			layer.KNorm = load(p+".self_attn.k_norm.weight", []int{layerHD})
		}

		// MoE: load router and expert weights from switch_mlp format
		if cfg.NumExperts > 0 && cfg.MoEIntermediate > 0 {
			moePath := p + ".mlp"
			if tryLoad(moePath + ".gate.weight") {
				layer.IsMoE = true
				// Router gate: [numExperts, hidden] — load as MLX quantized
				if cfg.QuantFormat == "mlx" && onTheFly {
					layer.RouterW = loadMLXW(moePath+".gate", cfg.NumExperts, h)
				}
				// Expert weights: switch_mlp format [numExperts, moeInter, packed]
				moeI := cfg.MoEIntermediate
				expGate, err := LoadSwitchMLXExperts(f, moePath+".switch_mlp.gate_proj", cfg.NumExperts, moeI, h, cfg.QuantGroup, cfg.QuantBits)
				if err == nil {
					layer.ExpertGateW = expGate
				} else {
					fmt.Printf("  MoE layer %d gate_proj: %v\n", l, err)
				}
				expUp, err := LoadSwitchMLXExperts(f, moePath+".switch_mlp.up_proj", cfg.NumExperts, moeI, h, cfg.QuantGroup, cfg.QuantBits)
				if err == nil {
					layer.ExpertUpW = expUp
				} else {
					fmt.Printf("  MoE layer %d up_proj: %v\n", l, err)
				}
				expDown, err := LoadSwitchMLXExperts(f, moePath+".switch_mlp.down_proj", cfg.NumExperts, h, moeI, cfg.QuantGroup, cfg.QuantBits)
				if err == nil {
					layer.ExpertDownW = expDown
				} else {
					fmt.Printf("  MoE layer %d down_proj: %v\n", l, err)
				}
				// Clear the non-MoE MLP weights (they don't apply)
				layer.GateWm = nil
				layer.UpWm = nil
				layer.DownWm = nil
				layer.GateW = nil
				layer.UpW = nil
				layer.DownW = nil
			}
		}
		// Gemma4: per-layer properties
		if len(cfg.LayerTypes) > l {
			lt := cfg.LayerTypes[l]
			// Head dim: global layers use GlobalHeadDim, sliding use HeadDim
			if lt == "full_attention" && cfg.GlobalHeadDim > 0 {
				layer.HeadDimLocal = cfg.GlobalHeadDim
			} else {
				layer.HeadDimLocal = cfg.HeadDim
			}
			// KV sharing: first N layers have own K/V, rest share
			firstKVShared := cfg.NumLayers - cfg.NumKVSharedLayers
			if l < firstKVShared || cfg.NumKVSharedLayers == 0 {
				layer.HasKV = true
			} else {
				layer.HasKV = false
				// Find the source layer (same layer type, in the first M layers)
				for src := 0; src < firstKVShared; src++ {
					if cfg.LayerTypes[src] == lt {
						layer.KVSourceLayer = src
					}
				}
			}
		} else {
			layer.HeadDimLocal = cfg.HeadDim
			layer.HasKV = true
		}

		// Layer scalar (Gemma4)
		layer.LayerScalar = 1.0
		if tryLoad(p + ".layer_scalar") {
			d := load(p+".layer_scalar", []int{1})
			layer.LayerScalar = d.Data()[0]
		}

		// V norm (Gemma4)
		if tryLoad(p + ".self_attn.v_norm.weight") {
			layer.VNorm = load(p+".self_attn.v_norm.weight", []int{layer.HeadDimLocal})
		}

		// Per-layer input gating weights (Gemma4)
		if cfg.HiddenPerLayer > 0 {
			hpl := cfg.HiddenPerLayer
			if cfg.QuantFormat == "mlx" && cfg.QuantBits > 0 {
				if qw, err := loadMLXWeight(f, prefix+p+".per_layer_input_gate", hpl, h, cfg.QuantGroup, cfg.QuantBits); err == nil {
					layer.PLIGate = DequantMLX(qw)
					qw2, _ := loadMLXWeight(f, prefix+p+".per_layer_projection", h, hpl, cfg.QuantGroup, cfg.QuantBits)
					if qw2 != nil {
						layer.PLIProj = DequantMLX(qw2)
					}
					if tryLoad(p + ".post_per_layer_input_norm.weight") {
						layer.PLIPostNorm = load(p+".post_per_layer_input_norm.weight", []int{h}).Data()
					}
				}
			} else if tryLoad(p + ".per_layer_input_gate.weight") {
				layer.PLIGate = load(p+".per_layer_input_gate.weight", nil).Data()
				layer.PLIProj = load(p+".per_layer_projection.weight", nil).Data()
				layer.PLIPostNorm = load(p+".post_per_layer_input_norm.weight", []int{h}).Data()
			}
		}

		m.Layers[l] = layer
	}

	// Gemma3: norm formula is (1 + weight) — confirmed in mlx-lm gemma3_text.py line 111
	// Gemma4 inherits from Gemma3n which uses raw weight (NOT 1+w)
	if cfg.ModelType == "gemma3_text" {
		for l := range m.Layers {
			for _, norm := range []*tensor.Tensor{
				m.Layers[l].InputNorm, m.Layers[l].PostNorm,
				m.Layers[l].PreFFNNorm, m.Layers[l].PostFFNNorm,
				m.Layers[l].QNorm, m.Layers[l].KNorm,
			} {
				if norm != nil {
					d := norm.Data()
					for i := range d {
						d[i] += 1.0
					}
				}
			}
		}
		nd := m.Norm.Data()
		for i := range nd {
			nd[i] += 1.0
		}
	}

	// Gemma4: load model-level per-layer projection weights
	if cfg.HiddenPerLayer > 0 {
		hpl := cfg.HiddenPerLayer
		totalDim := cfg.NumLayers * hpl
		// per_layer_model_projection: [totalDim, hidden] BF16 (not quantized)
		if tryLoad("model.per_layer_model_projection.weight") {
			m.PerLayerModelProj = load("model.per_layer_model_projection.weight", []int{totalDim, h}).Data()
		}
		// per_layer_projection_norm: [hpl]
		if tryLoad("model.per_layer_projection_norm.weight") {
			m.PerLayerProjNorm = load("model.per_layer_projection_norm.weight", []int{hpl}).Data()
		}
		// embed_tokens_per_layer: [vocabPerLayer, totalDim] quantized
		vpl := cfg.VocabPerLayer
		if vpl == 0 {
			vpl = 262144
		} // default for Gemma4
		if cfg.QuantFormat == "mlx" && cfg.QuantBits > 0 {
			if qw, err := loadMLXWeight(f, prefix+"model.embed_tokens_per_layer", vpl, totalDim, cfg.QuantGroup, cfg.QuantBits); err == nil {
				m.EmbedPerLayer = DequantMLX(qw)
				fmt.Printf("  Loaded per-layer embedding: [%d, %d]\n", vpl, totalDim)
			}
		} else if tryLoad("model.embed_tokens_per_layer.weight") {
			m.EmbedPerLayer = load("model.embed_tokens_per_layer.weight", []int{vpl, totalDim}).Data()
		}
		m.PerLayerInputScale = 0.7071067811865476 // 2^-0.5
		m.PerLayerProjScale = float32(1.0 / math.Sqrt(float64(h)))
		m.EmbedPerLayerScale = float32(math.Sqrt(float64(hpl)))
	}

	// Pre-compute RoPE frequencies
	m.precomputeRoPE()

	// Gemma4: precompute separate RoPE for SWA and full attention
	if cfg.ModelType == "gemma4_text" {
		maxSeq := cfg.MaxSeqLen
		if maxSeq > 2048 {
			maxSeq = 2048
		}

		// SWA: head_dim=256, theta=10000, partial_rotary_factor=1.0
		swaHD := cfg.HeadDim // 256
		swaHalf := swaHD / 2 // 128 rotated pairs
		m.RopeHalfSWA = swaHalf
		m.RopeFreqsSWA = make([]float32, maxSeq*swaHalf*2)
		swaTheta := 10000.0
		for pos := 0; pos < maxSeq; pos++ {
			for i := 0; i < swaHalf; i++ {
				// exponent denominator uses full head_dim (MLX: arange(0, rotated_dims, 2) / dims)
				freq := 1.0 / math.Pow(swaTheta, float64(2*i)/float64(swaHD))
				angle := float64(pos) * freq
				off := (pos*swaHalf + i) * 2
				m.RopeFreqsSWA[off] = float32(math.Cos(angle))
				m.RopeFreqsSWA[off+1] = float32(math.Sin(angle))
			}
		}

		// Full: head_dim=512, theta=1000000, partial_rotary_factor=0.25
		fullHD := cfg.GlobalHeadDim                // 512
		rotatedDims := int(float64(fullHD) * 0.25) // 128
		fullHalf := rotatedDims / 2                // 64 rotated pairs
		m.RopeHalfFull = fullHalf
		m.RopeFreqsFull = make([]float32, maxSeq*fullHalf*2)
		fullTheta := 1000000.0
		// Proportional RoPE: inv_freq = 1/(base^(arange(0, 2*rope_angles, 2) / head_dim))
		// Per HuggingFace modeling_rope_utils.py: denominator is head_dim (512), NOT rotated_dims
		for pos := 0; pos < maxSeq; pos++ {
			for i := 0; i < fullHalf; i++ {
				freq := 1.0 / math.Pow(fullTheta, float64(2*i)/float64(fullHD))
				angle := float64(pos) * freq
				off := (pos*fullHalf + i) * 2
				m.RopeFreqsFull[off] = float32(math.Cos(angle))
				m.RopeFreqsFull[off+1] = float32(math.Sin(angle))
			}
		}
		fmt.Printf("  RoPE: SWA half=%d (theta=10k), Full half=%d (theta=1M, partial=0.25)\n", swaHalf, fullHalf)
	}

	return m, nil
}

func (m *LlamaModel) precomputeRoPE() {
	cfg := m.Config
	headDim := cfg.HeadDim
	if headDim == 0 {
		headDim = cfg.HiddenSize / cfg.NumHeads
	}
	// For models with variable head_dim (Gemma4), use the max
	if cfg.GlobalHeadDim > headDim {
		headDim = cfg.GlobalHeadDim
	}
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

	// BOS token for Gemma
	if cfg.BOSTokenID > 0 && (cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text") {
		tokenIDs = append([]int{cfg.BOSTokenID}, tokenIDs...)
	}
	// Gemma4 instruct chat template: <bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n
	if cfg.ModelType == "gemma4_text" && m.Tok != nil {
		turnStart, turnEnd := -1, -1
		newlineID := -1
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
		if turnStart >= 0 && turnEnd >= 0 && newlineID >= 0 {
			user := m.Tok.Encode("user")
			mdl := m.Tok.Encode("model")
			wrapped := []int{cfg.BOSTokenID, turnStart}
			wrapped = append(wrapped, user...)
			wrapped = append(wrapped, newlineID)
			wrapped = append(wrapped, tokenIDs[1:]...) // skip BOS
			wrapped = append(wrapped, turnEnd)
			wrapped = append(wrapped, newlineID)
			wrapped = append(wrapped, turnStart)
			wrapped = append(wrapped, mdl...)
			wrapped = append(wrapped, newlineID)
			tokenIDs = wrapped
		}
	}
	// Qwen3/Qwen3-MoE instruct chat template: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
	if (cfg.ModelType == "qwen3" || cfg.ModelType == "qwen3_moe") && m.Tok != nil {
		imStart, imEnd, nlID := -1, -1, -1
		for id, tok := range m.Tok.InvVocab {
			if tok == "<|im_start|>" {
				imStart = id
			}
			if tok == "<|im_end|>" {
				imEnd = id
			}
			if tok == "\n" || tok == "\u010a" || tok == "Ċ" {
				nlID = id
			}
		}
		if imStart >= 0 && imEnd >= 0 && nlID >= 0 {
			user := m.Tok.Encode("user")
			assistant := m.Tok.Encode("assistant")
			wrapped := []int{imStart}
			wrapped = append(wrapped, user...)
			wrapped = append(wrapped, nlID)
			wrapped = append(wrapped, tokenIDs...)
			wrapped = append(wrapped, imEnd, nlID, imStart)
			wrapped = append(wrapped, assistant...)
			wrapped = append(wrapped, nlID)
			tokenIDs = wrapped
		}
	}

	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	headDim := cfg.HeadDim
	inter := cfg.Intermediate

	// Allocate KV cache (with optional TurboQuant compression)
	kvCacheK := make([][]float32, cfg.NumLayers) // [layers][seqLen * layerKVDim]
	kvCacheV := make([][]float32, cfg.NumLayers)
	var compressedKV []*CompressedKVCache
	if m.EnableTurboQuant || os.Getenv("TURBO_QUANT") == "1" {
		tqCfg := DefaultTurboQuantConfig()
		if m.TurboQuantStates == nil {
			m.TurboQuantStates = make(map[int]*TurboQuantState)
		}
		getTQ := func(layerHeadDim int) *TurboQuantState {
			if tq := m.TurboQuantStates[layerHeadDim]; tq != nil {
				return tq
			}
			tq := NewTurboQuantState(layerHeadDim, cfg.NumLayers, tqCfg)
			m.TurboQuantStates[layerHeadDim] = tq
			return tq
		}
		fmt.Printf("  TurboQuant: %d-bit keys, %d-bit values, window=%d\n",
			tqCfg.KeyBits, tqCfg.ValueBits, tqCfg.ResidualWindow)

		compressedKV = make([]*CompressedKVCache, cfg.NumLayers)
		for l := range compressedKV {
			layerHD := headDim
			if m.Layers[l].HeadDimLocal > 0 {
				layerHD = m.Layers[l].HeadDimLocal
			}
			layerKVDim := numKVHeads * layerHD
			tq := getTQ(layerHD)
			compressedKV[l] = NewCompressedKVCache(layerKVDim, numKVHeads, layerHD, tq, tq.IsProtectedLayer(l))
		}
	} else {
		for l := range kvCacheK {
			layerHD := headDim
			if m.Layers[l].HeadDimLocal > 0 {
				layerHD = m.Layers[l].HeadDimLocal
			}
			layerKVDim := numKVHeads * layerHD
			kvCacheK[l] = make([]float32, 0, 2048*layerKVDim)
			kvCacheV[l] = make([]float32, 0, 2048*layerKVDim)
		}
	}

	output := make([]int, len(tokenIDs), len(tokenIDs)+maxTokens)
	copy(output, tokenIDs)

	// Reusable CPU decode scratch for GQA attention.
	maxHeadDim := headDim
	for i := range m.Layers {
		if m.Layers[i].HeadDimLocal > maxHeadDim {
			maxHeadDim = m.Layers[i].HeadDimLocal
		}
	}
	maxSeqLen := len(tokenIDs) + maxTokens
	if maxSeqLen < 1 {
		maxSeqLen = 1
	}
	attnScoresScratch := make([]float32, maxSeqLen)
	attnOutScratch := make([]float32, numHeads*maxHeadDim)

	// Process prompt + generate
	for step := 0; step < len(tokenIDs)+maxTokens-1; step++ {
		var tokID int
		if step < len(tokenIDs) {
			tokID = tokenIDs[step]
		} else {
			tokID = output[len(output)-1]
		}

		// Embed single token using the same helper exposed for verifier/MTP paths.
		hidden := make([]float32, h)
		if err := m.ScaledTokenEmbeddingInto(hidden, tokID); err != nil {
			panic(err)
		}

		pos := step

		if debugOpHook != nil {
			debugOpHook("cpu", step, 0, "embed_scaled", hidden)
		}

		// Gemma4: compute per-layer inputs for this token
		var perLayerInputs [][]float32
		if m.PerLayerModelProj != nil && cfg.HiddenPerLayer > 0 {
			hpl := cfg.HiddenPerLayer
			nl := cfg.NumLayers
			totalDim := nl * hpl
			// Project hidden → [numLayers * hiddenPerLayer]
			proj := make([]float32, totalDim)
			gemvNT(proj, hidden, m.PerLayerModelProj, h, totalDim)
			for i := range proj {
				proj[i] *= m.PerLayerProjScale
			}
			// RMSNorm each layer's slice
			for l := 0; l < nl; l++ {
				sl := proj[l*hpl : (l+1)*hpl]
				rmsNormInPlace(sl, m.PerLayerProjNorm, float32(cfg.RMSNormEps))
			}
			// Add per-layer embedding if available
			if m.EmbedPerLayer != nil && tokID < cfg.VocabPerLayer {
				embRow := m.EmbedPerLayer[tokID*totalDim : (tokID+1)*totalDim]
				for i := range proj {
					proj[i] = (proj[i] + embRow[i]*m.EmbedPerLayerScale) * m.PerLayerInputScale
				}
			}
			// Split into per-layer slices
			perLayerInputs = make([][]float32, nl)
			for l := 0; l < nl; l++ {
				perLayerInputs[l] = proj[l*hpl : (l+1)*hpl]
			}
			if debugCPUPerLayerInputsOverrideHook != nil {
				debugCPUPerLayerInputsOverrideHook(step, perLayerInputs)
			}
			if debugOpHook != nil && len(perLayerInputs) > 0 {
				debugOpHook("cpu", step, 0, "pli0_input", perLayerInputs[0])
			}
		}

		for l := 0; l < cfg.NumLayers; l++ {
			layer := &m.Layers[l]
			if debugCPUHiddenInOverrideHook != nil {
				debugCPUHiddenInOverrideHook(step, l, hidden)
			}
			residual := make([]float32, h)
			copy(residual, hidden)
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "hidden_in", hidden)
			}

			// RMS Norm (BF16 for Gemma3)
			if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
				simd.RMSNormBF16(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))
			} else {
				rmsNormInPlace(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "normed", hidden)
			}

			// BF16 embed scaling was already applied above

			// Q, K, V projections (single token: [1, h] @ [h, dim])
			layerHeadDim := headDim
			if layer.HeadDimLocal > 0 {
				layerHeadDim = layer.HeadDimLocal
			}
			qDim := numHeads * layerHeadDim
			q := make([]float32, qDim)
			layerKVDim := numKVHeads * layerHeadDim

			// Always compute Q
			if layer.QWq != nil {
				m.mvQ(q, hidden, layer.QWq)
			} else if layer.QWm != nil {
				GemvMLQ(q, hidden, layer.QWm)
			} else {
				m.mv(q, hidden, layer.QW.Data(), h, qDim)
			}

			// K, V: only compute for HasKV layers; shared layers reuse source KV cache
			var k, v []float32
			if layer.HasKV {
				k = make([]float32, layerKVDim)
				v = make([]float32, layerKVDim)
				if layer.KWq != nil {
					m.mvQ(k, hidden, layer.KWq)
					m.mvQ(v, hidden, layer.VWq)
				} else if layer.KWm != nil {
					GemvMLQ(k, hidden, layer.KWm)
					GemvMLQ(v, hidden, layer.VWm)
				} else {
					m.mv(k, hidden, layer.KW.Data(), h, layerKVDim)
					m.mv(v, hidden, layer.VW.Data(), h, layerKVDim)
				}
			}

			if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
				simd.ToBF16(q)
				if k != nil {
					simd.ToBF16(k)
					simd.ToBF16(v)
				}
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "q", q)
				if k != nil {
					debugOpHook("cpu", step, l, "k", k)
					debugOpHook("cpu", step, l, "v", v)
				}
			}

			// Add bias if present (Qwen2)
			if layer.QB != nil {
				qb := layer.QB.Data()
				simd.VecAdd(q, q, qb)
				if k != nil {
					kb, vb := layer.KB.Data(), layer.VB.Data()
					simd.VecAdd(k, k, kb)
					simd.VecAdd(v, v, vb)
				}
			}

			// Select norm function
			normFn := rmsNormInPlace
			if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
				normFn = rmsNormBF16
			}

			// V norm (Gemma4: RMSNormNoScale — normalize without weight)
			if cfg.ModelType == "gemma4_text" && v != nil {
				eps := float32(cfg.RMSNormEps)
				for head := 0; head < numKVHeads; head++ {
					simd.RMSNormNoScale(v[head*layerHeadDim:(head+1)*layerHeadDim], eps)
				}
			} else if layer.VNorm != nil && v != nil {
				vnorm := layer.VNorm.Data()
				for head := 0; head < numKVHeads; head++ {
					normFn(v[head*layerHeadDim:(head+1)*layerHeadDim], vnorm, float32(cfg.RMSNormEps))
				}
			}

			// QK-Norm (Qwen3/Gemma3/4): RMSNorm each head of Q and K separately
			if layer.QNorm != nil {
				qNorm := layer.QNorm.Data()
				for head := 0; head < numHeads; head++ {
					normFn(q[head*layerHeadDim:(head+1)*layerHeadDim], qNorm, float32(cfg.RMSNormEps))
				}
				if k != nil {
					kNorm := layer.KNorm.Data()
					for head := 0; head < numKVHeads; head++ {
						normFn(k[head*layerHeadDim:(head+1)*layerHeadDim], kNorm, float32(cfg.RMSNormEps))
					}
				}
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "q_qknorm", q)
				if k != nil {
					debugOpHook("cpu", step, l, "k_qknorm", k)
					debugOpHook("cpu", step, l, "v_attn", v)
				}
			}

			// RoPE on Q (always) and K (only if HasKV)
			if cfg.ModelType == "gemma4_text" && m.RopeFreqsSWA != nil {
				// Gemma4: per-layer RoPE with different theta and partial rotation
				isSWA := true
				if len(cfg.LayerTypes) > l {
					isSWA = cfg.LayerTypes[l] == "sliding_attention"
				}
				if isSWA {
					// SWA: full rotation, theta=10k, head_dim=256
					applyRoPEPartial(q, m.RopeFreqsSWA, pos, numHeads, layerHeadDim, m.RopeHalfSWA)
					if k != nil {
						applyRoPEPartial(k, m.RopeFreqsSWA, pos, numKVHeads, layerHeadDim, m.RopeHalfSWA)
					}
				} else {
					// Full: partial rotation (25%), theta=1M, head_dim=512
					applyRoPEPartial(q, m.RopeFreqsFull, pos, numHeads, layerHeadDim, m.RopeHalfFull)
					if k != nil {
						applyRoPEPartial(k, m.RopeFreqsFull, pos, numKVHeads, layerHeadDim, m.RopeHalfFull)
					}
				}
			} else {
				applyRoPE(q, m.RopeFreqs, pos, numHeads, layerHeadDim)
				if k != nil {
					applyRoPE(k, m.RopeFreqs, pos, numKVHeads, layerHeadDim)
				}
			}

			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "q_attn", q)
				if k != nil {
					debugOpHook("cpu", step, l, "k_attn", k)
					debugOpHook("cpu", step, l, "v_attn", v)
				}
			}

			// KV cache: append for HasKV layers, reuse source for shared layers
			kvLayer := l
			if !layer.HasKV {
				kvLayer = layer.KVSourceLayer
			}
			if k != nil {
				if compressedKV != nil {
					compressedKV[kvLayer].Append(k, v)
				} else {
					kvCacheK[kvLayer] = append(kvCacheK[kvLayer], k...)
					kvCacheV[kvLayer] = append(kvCacheV[kvLayer], v...)
				}
			}

			// Attention: Q against cached K, V (may be from source layer)
			seqLen := pos + 1
			attnSeqLen := seqLen
			attnKVOffset := 0
			// SWA layers: restrict attention to sliding_window most recent positions
			if cfg.SlidingWindow > 0 && len(cfg.LayerTypes) > l && cfg.LayerTypes[l] == "sliding_attention" {
				if seqLen > cfg.SlidingWindow {
					attnSeqLen = cfg.SlidingWindow
					attnKVOffset = seqLen - cfg.SlidingWindow
				}
			}
			var attnOut []float32
			var kCache, vCache []float32
			if compressedKV != nil {
				kCache = compressedKV[kvLayer].GetK()
				vCache = compressedKV[kvLayer].GetV()
			} else {
				kCache = kvCacheK[kvLayer]
				vCache = kvCacheV[kvLayer]
			}
			attnOut = attnOutScratch[:qDim]
			attnScores := attnScoresScratch[:attnSeqLen]
			if cfg.ModelType == "gemma4_text" {
				gqaAttentionScaleInto(attnOut, attnScores, q, kCache[attnKVOffset*numKVHeads*layerHeadDim:], vCache[attnKVOffset*numKVHeads*layerHeadDim:], attnSeqLen, numHeads, numKVHeads, layerHeadDim, 1.0)
			} else {
				gqaAttentionScaleInto(attnOut, attnScores, q, kCache[attnKVOffset*numKVHeads*layerHeadDim:], vCache[attnKVOffset*numKVHeads*layerHeadDim:], attnSeqLen, numHeads, numKVHeads, layerHeadDim, float32(1.0/math.Sqrt(float64(layerHeadDim))))
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "attn", attnOut)
			}

			// Output projection
			oOut := make([]float32, h)
			if layer.OWq != nil {
				m.mvQ(oOut, attnOut, layer.OWq)
			} else if layer.OWm != nil {
				GemvMLQ(oOut, attnOut, layer.OWm)
			} else {
				m.mv(oOut, attnOut, layer.OW.Data(), qDim, h)
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "o", oOut)
			}

			// Gemma3: post-attn norm BEFORE residual add
			if layer.PreFFNNorm != nil {
				// Gemma3 pattern: norm(attn_output), then add residual
				rmsNormInPlace(oOut, layer.PostNorm.Data(), float32(cfg.RMSNormEps))
				for i := range hidden {
					hidden[i] = residual[i] + oOut[i]
				}
				copy(residual, hidden)
			} else {
				// Qwen/LLaMA pattern: add residual, then norm
				simd.VecAdd(hidden, residual, oOut)
				copy(residual, hidden)
				rmsNormInPlace(hidden, layer.PostNorm.Data(), float32(cfg.RMSNormEps))
			}

			// MLP input: preFFNNorm for Gemma3, postNorm already applied for Qwen
			mlpInput := hidden
			if layer.PreFFNNorm != nil {
				mlpInput = make([]float32, h)
				copy(mlpInput, hidden)
				if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
					simd.RMSNormBF16(mlpInput, layer.PreFFNNorm.Data(), float32(cfg.RMSNormEps))
					// mlpInput is already BF16 from RMSNormBF16
				} else {
					rmsNormInPlace(mlpInput, layer.PreFFNNorm.Data(), float32(cfg.RMSNormEps))
				}
			}

			layerInter := inter
			if layer.GateWq != nil && layer.GateWq.OutDim > 0 {
				layerInter = layer.GateWq.OutDim
			} else if layer.GateWm != nil && layer.GateWm.OutDim > 0 {
				layerInter = layer.GateWm.OutDim
			} else if layer.GateW != nil {
				s := layer.GateW.Shape()
				if len(s) >= 2 {
					if m.Large {
						layerInter = s[0]
					} else {
						layerInter = s[1]
					}
				} else if len(s) == 1 && s[0] > 0 {
					layerInter = s[0]
				}
			}

			if debugCPUMLPInputOverrideHook != nil {
				debugCPUMLPInputOverrideHook(step, l, mlpInput)
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "mlp_input", mlpInput)
			}

			// MLP: gate * up → SiLU → down (or MoE for expert layers)
			var down []float32
			if layer.IsMoE && layer.ExpertGateW != nil {
				// MoE forward: router → top-k experts → weighted sum
				down = moeForward(mlpInput, layer, cfg)
			} else {
				gate := make([]float32, layerInter)
				up := make([]float32, layerInter)
				if layer.GateWq != nil {
					m.mvQ(gate, mlpInput, layer.GateWq)
					m.mvQ(up, mlpInput, layer.UpWq)
				} else if layer.GateWm != nil {
					GemvMLQ(gate, mlpInput, layer.GateWm)
					GemvMLQ(up, mlpInput, layer.UpWm)
				} else {
					m.mv(gate, mlpInput, layer.GateW.Data(), h, layerInter)
					m.mv(up, mlpInput, layer.UpW.Data(), h, layerInter)
				}

				if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
					simd.ToBF16(gate)
					simd.ToBF16(up)
				}
				if debugOpHook != nil {
					debugOpHook("cpu", step, l, "gate_pre", gate)
					debugOpHook("cpu", step, l, "up", up)
				}
				// Activation(gate) * up
				if cfg.HiddenAct == "gelu_pytorch_tanh" {
					simd.GELUTanhMul(gate, gate, up)
					simd.ToBF16(gate)
				} else {
					simd.VecSiLUMul(gate, gate, up)
				}
				if debugOpHook != nil {
					debugOpHook("cpu", step, l, "gate_act", gate)
				}

				// Down projection
				down = make([]float32, h)
				if layer.DownWq != nil {
					m.mvQ(down, gate, layer.DownWq)
				} else if layer.DownWm != nil {
					GemvMLQ(down, gate, layer.DownWm)
				} else {
					m.mv(down, gate, layer.DownW.Data(), layerInter, h)
				}
			}

			// BF16 down projection output for Gemma3
			if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
				simd.ToBF16(down)
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "down", down)
			}

			// Post-FFN norm (Gemma3)
			if layer.PostFFNNorm != nil {
				if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
					rmsNormBF16(down, layer.PostFFNNorm.Data(), float32(cfg.RMSNormEps))
				} else {
					rmsNormInPlace(down, layer.PostFFNNorm.Data(), float32(cfg.RMSNormEps))
				}
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "down_postffn", down)
			}

			// Residual
			simd.VecAdd(hidden, residual, down)
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "hidden_post_ffn", hidden)
			}

			// Per-layer input gating (Gemma4)
			if layer.PLIGate != nil && perLayerInputs != nil && l < len(perLayerInputs) {
				hpl := cfg.HiddenPerLayer
				pli := perLayerInputs[l]
				// gate = gelu(per_layer_input_gate(h)) * per_layer_input → [hiddenPerLayer]
				gate2 := make([]float32, hpl)
				gemvNT(gate2, hidden, layer.PLIGate, h, hpl)
				simd.GELUTanhMul(gate2, gate2, pli)
				// proj = per_layer_projection(gate) → [hidden]
				proj2 := make([]float32, h)
				gemvNT(proj2, gate2, layer.PLIProj, hpl, h)
				// norm
				rmsNormInPlace(proj2, layer.PLIPostNorm, float32(cfg.RMSNormEps))
				// residual add
				simd.VecAdd(hidden, hidden, proj2)
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "hidden_post_pli", hidden)
			}
			// Layer scalar (Gemma4)
			if layer.LayerScalar != 1.0 {
				simd.VecScale(hidden, hidden, layer.LayerScalar)
			}
			if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
				simd.ToBF16(hidden)
			}
			if debugLayerHook != nil {
				debugLayerHook("cpu", step, l, hidden)
			}

		}

		// Final norm (BF16 for Gemma3)
		if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
			simd.RMSNormBF16(hidden, m.Norm.Data(), float32(cfg.RMSNormEps))
		} else {
			rmsNormInPlace(hidden, m.Norm.Data(), float32(cfg.RMSNormEps))
		}

		// LM head: logits = hidden @ lm_head^T (greedy: take argmax)
		if step >= len(tokenIDs)-1 {
			logits := make([]float32, cfg.VocabSize)
			if err := m.LMHeadLogitsInto(logits, hidden); err != nil {
				panic(err)
			}
			maxIdx, _, err := ArgmaxLogits(logits)
			if err != nil {
				panic(err)
			}
			if debugLogitsHook != nil {
				debugLogitsHook("cpu", step, hidden, logits)
			}
			output = append(output, maxIdx)
		}
	}
	return output
}

// --- Low-level helpers ---

func rmsNormInPlace(x, weight []float32, eps float32) {
	simd.RMSNorm(x, weight, eps)
}

// gemv: out = x @ w where w is either:
//
//	pre-transposed [inDim, outDim] (use NN), or
//	original [outDim, inDim] (use NT via dot products)
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

func geluTanh(x float32) float32 {
	// GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	x3 := x * x * x
	inner := float32(0.7978845608) * (x + 0.044715*x3) // sqrt(2/pi) ≈ 0.7978845608
	return 0.5 * x * (1.0 + float32(math.Tanh(float64(inner))))
}

func applyRoPE(x, freqs []float32, pos, numHeads, headDim int) {
	applyRoPEPartial(x, freqs, pos, numHeads, headDim, headDim/2)
}

// applyRoPEPartial applies RoPE with partial rotation.
// Only the first rotHalf pairs are rotated; remaining dims are untouched.
func applyRoPEPartial(x, freqs []float32, pos, numHeads, headDim, rotHalf int) {
	for h := 0; h < numHeads; h++ {
		for i := 0; i < rotHalf; i++ {
			freqOff := (pos*rotHalf + i) * 2
			if freqOff+1 >= len(freqs) {
				break
			}
			cos := freqs[freqOff]
			sin := freqs[freqOff+1]
			idx0 := h*headDim + i
			idx1 := h*headDim + i + rotHalf
			x0 := x[idx0]
			x1 := x[idx1]
			x[idx0] = x0*cos - x1*sin
			x[idx1] = x0*sin + x1*cos
		}
	}
}

func gqaAttention(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int) []float32 {
	return gqaAttentionScale(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, float32(1.0/math.Sqrt(float64(headDim))))
}

func gqaAttentionScale(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	out := make([]float32, numHeads*headDim)
	scores := make([]float32, seqLen)
	gqaAttentionScaleInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
	return out
}

func gqaAttentionScaleInto(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) {
	h := numHeads * headDim
	kvDim := numKVHeads * headDim
	headsPerKV := numHeads / numKVHeads
	out = out[:h]
	clear(out)
	if seqLen == 0 {
		return
	}
	scores = scores[:seqLen]

	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV

		// Compute attention scores for this head against all cached K.
		// Reuse one caller-owned score buffer across heads.
		qHead := q[head*headDim : (head+1)*headDim]
		for t := 0; t < seqLen; t++ {
			kHead := kCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
			scores[t] = simd.Sdot(qHead, kHead) * scale
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

		// Weighted sum of V. Iterate by cached token and use SIMD SAXPY on the
		// contiguous head slice instead of scalar strided accumulation per dim.
		outHead := out[head*headDim : (head+1)*headDim]
		for t := 0; t < seqLen; t++ {
			vHead := vCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
			simd.Saxpy(scores[t], vHead, outHead)
		}
	}
}

// gemvNTParallel is like gemvNT but parallelized across CPU cores.
func gemvNTParallel(out, x []float32, w []float32, inDim, outDim int) {
	nCPU := runtime.NumCPU()
	if nCPU > 8 {
		nCPU = 8
	} // cap at 8 for cache efficiency
	chunkSize := (outDim + nCPU - 1) / nCPU

	var wg sync.WaitGroup
	wg.Add(nCPU)
	for c := 0; c < nCPU; c++ {
		start := c * chunkSize
		end := start + chunkSize
		if end > outDim {
			end = outDim
		}
		go func(s, e int) {
			defer wg.Done()
			for j := s; j < e; j++ {
				row := w[j*inDim : (j+1)*inDim]
				if inDim >= 8 {
					out[j] = simd.Sdot(x, row)
				} else {
					sum := float32(0)
					for p := 0; p < inDim; p++ {
						sum += x[p] * row[p]
					}
					out[j] = sum
				}
			}
		}(start, end)
	}
	wg.Wait()
}
