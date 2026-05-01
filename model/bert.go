package model

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-tinygrad/safetensors"
	"github.com/rcarmo/go-tinygrad/tensor"
)

// BertConfig holds model hyperparameters.
type BertConfig struct {
	VocabSize    int
	HiddenSize   int
	NumLayers    int
	NumHeads     int
	Intermediate int
	MaxSeqLen    int
}

// GTESmallConfig is the config for thenlper/gte-small.
var GTESmallConfig = BertConfig{
	VocabSize: 30522, HiddenSize: 384, NumLayers: 12,
	NumHeads: 12, Intermediate: 1536, MaxSeqLen: 512,
}

// BertModel holds loaded weights for a BERT-style encoder.
type BertModel struct {
	Config BertConfig

	// Embeddings
	WordEmb    *tensor.Tensor
	PosEmb     *tensor.Tensor
	TypeEmb    *tensor.Tensor
	EmbLnW     *tensor.Tensor
	EmbLnB     *tensor.Tensor

	// Layers
	Layers []BertLayer

	// Pooler
	PoolerW *tensor.Tensor
	PoolerB *tensor.Tensor
}

// BertLayer holds weights for one transformer layer.
// Weights are stored pre-transposed for SgemmNN (faster than SgemmNT).
// QKV weights are fused into a single [hidden, 3*hidden] matrix.
type BertLayer struct {
	QKVW, QKVB                   *tensor.Tensor // fused [hidden, 3*hidden]
	AttnOutW, AttnOutB            *tensor.Tensor
	AttnLnW, AttnLnB             *tensor.Tensor
	FfnInterW, FfnInterB         *tensor.Tensor
	FfnOutW, FfnOutB             *tensor.Tensor
	FfnLnW, FfnLnB               *tensor.Tensor
}

// LoadGTESmall loads the GTE-small model from a safetensors file.
func LoadGTESmall(path string) (*BertModel, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}

	cfg := GTESmallConfig
	m := &BertModel{Config: cfg}

	load := func(name string, shape []int) *tensor.Tensor {
		data, _, err := f.GetFloat32(name)
		if err != nil {
			panic(fmt.Sprintf("load %s: %v", name, err))
		}
		return tensor.FromFloat32(data, shape)
	}

	// loadT loads a weight matrix and pre-transposes it for SgemmNN.
	// Input shape [outDim, inDim] → stored as [inDim, outDim].
	loadT := func(name string, shape []int) *tensor.Tensor {
		t := load(name, shape)
		return t.Transpose2D()
	}

	h := cfg.HiddenSize
	inter := cfg.Intermediate

	m.WordEmb = load("embeddings.word_embeddings.weight", []int{cfg.VocabSize, h})
	m.PosEmb = load("embeddings.position_embeddings.weight", []int{cfg.MaxSeqLen, h})
	m.TypeEmb = load("embeddings.token_type_embeddings.weight", []int{2, h})
	m.EmbLnW = load("embeddings.LayerNorm.weight", []int{h})
	m.EmbLnB = load("embeddings.LayerNorm.bias", []int{h})

	m.Layers = make([]BertLayer, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		p := fmt.Sprintf("encoder.layer.%d", l)
		// Fuse Q, K, V weights into single [h, 3h] matrix (pre-transposed)
		qwOrig := load(p+".attention.self.query.weight", []int{h, h})
		kwOrig := load(p+".attention.self.key.weight", []int{h, h})
		vwOrig := load(p+".attention.self.value.weight", []int{h, h})
		qkvW := fuseQKVWeights(qwOrig, kwOrig, vwOrig, h)
		qb := load(p+".attention.self.query.bias", []int{h})
		kb := load(p+".attention.self.key.bias", []int{h})
		vb := load(p+".attention.self.value.bias", []int{h})
		qkvB := fuseQKVBias(qb, kb, vb, h)
		m.Layers[l] = BertLayer{
			QKVW: qkvW,
			QKVB: qkvB,
			AttnOutW: loadT(p+".attention.output.dense.weight", []int{h, h}),
			AttnOutB: load(p+".attention.output.dense.bias", []int{h}),
			AttnLnW:  load(p+".attention.output.LayerNorm.weight", []int{h}),
			AttnLnB:  load(p+".attention.output.LayerNorm.bias", []int{h}),
			FfnInterW: loadT(p+".intermediate.dense.weight", []int{inter, h}),
			FfnInterB: load(p+".intermediate.dense.bias", []int{inter}),
			FfnOutW:   loadT(p+".output.dense.weight", []int{h, inter}),
			FfnOutB:   load(p+".output.dense.bias", []int{h}),
			FfnLnW:    load(p+".output.LayerNorm.weight", []int{h}),
			FfnLnB:    load(p+".output.LayerNorm.bias", []int{h}),
		}
	}

	m.PoolerW = loadT("pooler.dense.weight", []int{h, h})
	m.PoolerB = load("pooler.dense.bias", []int{h})

	return m, nil
}

// Forward runs the model on tokenized input. Returns [seqLen, hidden].
func (m *BertModel) Forward(tokenIDs []int) *tensor.Tensor {
	cfg := m.Config
	seqLen := len(tokenIDs)
	h := cfg.HiddenSize
	heads := cfg.NumHeads
	headDim := h / heads

	// Embeddings: word + position + token_type
	wordEmb := tensor.Embedding(m.WordEmb, tokenIDs)
	posIDs := make([]int, seqLen)
	for i := range posIDs { posIDs[i] = i }
	posEmb := tensor.Embedding(m.PosEmb, posIDs)
	typeIDs := make([]int, seqLen) // all zeros
	typeEmb := tensor.Embedding(m.TypeEmb, typeIDs)

	hidden := wordEmb.Add(posEmb).Add(typeEmb)
	hidden = hidden.LayerNorm(m.EmbLnW, m.EmbLnB, 1e-12)

	// Transformer layers
	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]

		// Fused QKV: [seqLen, h] @ [3h, h]^T → [seqLen, 3h] with contiguous Q,K,V blocks
		qkv := hidden.MatMulTransposed(layer.QKVW)
		qkv.Realize()
		qkvData := qkv.Data()
		// Add bias
		qkvBias := layer.QKVB.Data()
		for s := 0; s < seqLen; s++ {
			for d := 0; d < 3*h; d++ {
				qkvData[s*3*h+d] += qkvBias[d]
			}
		}
		// Split: Q=[0:seqLen*h], K=[seqLen*h:2*seqLen*h], V=[2*seqLen*h:]
		// Each row s has [q_s(h), k_s(h), v_s(h)] — need to gather into contiguous blocks
		qData := make([]float32, seqLen*h)
		kData := make([]float32, seqLen*h)
		vData := make([]float32, seqLen*h)
		for s := 0; s < seqLen; s++ {
			copy(qData[s*h:(s+1)*h], qkvData[s*3*h:s*3*h+h])
			copy(kData[s*h:(s+1)*h], qkvData[s*3*h+h:s*3*h+2*h])
			copy(vData[s*h:(s+1)*h], qkvData[s*3*h+2*h:s*3*h+3*h])
		}
		q := tensor.FromFloat32(qData, []int{seqLen, h})
		k := tensor.FromFloat32(kData, []int{seqLen, h})
		v := tensor.FromFloat32(vData, []int{seqLen, h})

		// Multi-head: reshape to [seqLen, heads, headDim], then score
		attnOut := multiHeadAttention(q, k, v, seqLen, heads, headDim)

		// Output projection + residual + layernorm
		attnProj := attnOut.LinearPreT(layer.AttnOutW, layer.AttnOutB)
		hidden = hidden.Add(attnProj)
		hidden = hidden.LayerNorm(layer.AttnLnW, layer.AttnLnB, 1e-12)

		// FFN
		ffnHidden := hidden.LinearPreT(layer.FfnInterW, layer.FfnInterB)
		ffnHidden = ffnHidden.GELU()
		ffnOut := ffnHidden.LinearPreT(layer.FfnOutW, layer.FfnOutB)
		hidden = hidden.Add(ffnOut)
		hidden = hidden.LayerNorm(layer.FfnLnW, layer.FfnLnB, 1e-12)
	}

	return hidden
}

// Embed produces an L2-normalized embedding via mean pooling.
func (m *BertModel) Embed(tokenIDs []int, attnMask []bool) []float32 {
	hidden := m.Forward(tokenIDs)
	hidden.Realize()
	data := hidden.Data()
	h := m.Config.HiddenSize
	seqLen := len(tokenIDs)

	// Mean pooling over attended tokens
	out := make([]float32, h)
	count := 0
	for s := 0; s < seqLen; s++ {
		if attnMask[s] {
			for d := 0; d < h; d++ {
				out[d] += data[s*h+d]
			}
			count++
		}
	}
	if count > 0 {
		inv := 1.0 / float32(count)
		for d := range out { out[d] *= inv }
	}

	// L2 normalize
	norm := float32(0)
	for _, v := range out { norm += v * v }
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		inv := 1.0 / norm
		for i := range out { out[i] *= inv }
	}
	return out
}

// multiHeadAttention computes multi-head self-attention.
// q, k, v are [seqLen, hidden]. Returns [seqLen, hidden].
func multiHeadAttention(q, k, v *tensor.Tensor, seqLen, heads, headDim int) *tensor.Tensor {
	q.Realize(); k.Realize(); v.Realize()
	hidden := heads * headDim
	qData, kData, vData := q.Data(), k.Data(), v.Data()
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	out := make([]float32, seqLen*hidden)

	for h := 0; h < heads; h++ {
		// Compute attention scores for this head
		scores := make([]float32, seqLen*seqLen)
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				sum := float32(0)
				for d := 0; d < headDim; d++ {
					sum += qData[i*hidden+h*headDim+d] * kData[j*hidden+h*headDim+d]
				}
				scores[i*seqLen+j] = sum * scale
			}
		}

		// Softmax per row
		for i := 0; i < seqLen; i++ {
			row := scores[i*seqLen : (i+1)*seqLen]
			mx := row[0]
			for _, v := range row[1:] { if v > mx { mx = v } }
			sum := float32(0)
			for j := range row {
				row[j] = float32(math.Exp(float64(row[j] - mx)))
				sum += row[j]
			}
			inv := 1.0 / sum
			for j := range row { row[j] *= inv }
		}

		// Context: scores @ V (per head)
		for i := 0; i < seqLen; i++ {
			for d := 0; d < headDim; d++ {
				sum := float32(0)
				for j := 0; j < seqLen; j++ {
					sum += scores[i*seqLen+j] * vData[j*hidden+h*headDim+d]
				}
				out[i*hidden+h*headDim+d] = sum
			}
		}
	}

	return tensor.FromFloat32(out, []int{seqLen, hidden})
}

// fuseQKVWeights concatenates 3 weight matrices [h, h] into [3h, h] (row-major).
// Used with MatMulTransposed for contiguous Q,K,V output blocks.
func fuseQKVWeights(qOrig, kOrig, vOrig *tensor.Tensor, h int) *tensor.Tensor {
	qD, kD, vD := qOrig.Data(), kOrig.Data(), vOrig.Data()
	fused := make([]float32, 3*h*h)
	copy(fused[0:h*h], qD)
	copy(fused[h*h:2*h*h], kD)
	copy(fused[2*h*h:3*h*h], vD)
	return tensor.FromFloat32(fused, []int{3 * h, h})
}

// fuseQKVBias concatenates 3 bias vectors [h] into [3h].
func fuseQKVBias(q, k, v *tensor.Tensor, h int) *tensor.Tensor {
	fused := make([]float32, 3*h)
	copy(fused[0:h], q.Data())
	copy(fused[h:2*h], k.Data())
	copy(fused[2*h:3*h], v.Data())
	return tensor.FromFloat32(fused, []int{3 * h})
}
