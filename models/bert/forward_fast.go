package bert

import (
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/simd"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	blasGonum "gonum.org/v1/gonum/blas/gonum"
)

// ForwardFast runs the model with zero allocations in the hot path.
// Uses pre-allocated workspace buffers and SIMD kernels directly.
func init() {
	blas32.Use(blasGonum.Implementation{})
}

func (m *BertModel) ForwardFast(tokenIDs []int, ws *Workspace) []float32 {
	cfg := m.Config
	seqLen := len(tokenIDs)
	h := cfg.HiddenSize
	heads := cfg.NumHeads
	headDim := h / heads

	hidden := ws.buf0[:seqLen*h]

	// Embeddings: word + position + token_type
	wData := m.WordEmb.Data()
	pData := m.PosEmb.Data()
	tData := m.TypeEmb.Data()
	for s := 0; s < seqLen; s++ {
		off := s * h
		wOff := tokenIDs[s] * h
		pOff := s * h
		for d := 0; d < h; d++ {
			hidden[off+d] = wData[wOff+d] + pData[pOff+d] + tData[d] // type 0
		}
	}

	// Embedding LayerNorm
	layerNormInPlace(hidden, m.EmbLnW.Data(), m.EmbLnB.Data(), seqLen, h, 1e-12)

	// Transformer layers
	for l := 0; l < cfg.NumLayers; l++ {
		layer := &m.Layers[l]
		temp := ws.buf1[:seqLen*h]

		// Fused QKV: [seqLen, h] @ [3h, h]^T → [seqLen, 3h]
		// Uses NT matmul so output has contiguous Q,K,V per row
		qkv := ws.qkvBuf[:seqLen*h*3]
		for i := range qkv {
			qkv[i] = 0
		}
		qkvW := layer.QKVW.Data()
		qkvB := layer.QKVB.Data()
		if simd.HasSgemmAsm {
			simd.SgemmNT(seqLen, 3*h, h, 1.0,
				unsafe.Pointer(&hidden[0]), unsafe.Pointer(&qkvW[0]), unsafe.Pointer(&qkv[0]),
				h, h, 3*h)
		} else {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < 3*h; j++ {
					sum := float32(0)
					for p := 0; p < h; p++ {
						sum += hidden[i*h+p] * qkvW[j*h+p]
					}
					qkv[i*3*h+j] = sum
				}
			}
		}
		// Add bias
		for s := 0; s < seqLen; s++ {
			for d := 0; d < 3*h; d++ {
				qkv[s*3*h+d] += qkvB[d]
			}
		}

		// Split Q,K,V: each row has [q(h), k(h), v(h)]
		q := ws.buf1[:seqLen*h]
		kBuf := ws.tempHidden[:seqLen*h]
		vBuf := ws.attnOut[:seqLen*h]
		for s := 0; s < seqLen; s++ {
			copy(q[s*h:(s+1)*h], qkv[s*3*h:s*3*h+h])
			copy(kBuf[s*h:(s+1)*h], qkv[s*3*h+h:s*3*h+2*h])
			copy(vBuf[s*h:(s+1)*h], qkv[s*3*h+2*h:s*3*h+3*h])
		}

		// Multi-head attention
		attnOut := ws.qkvBuf[:seqLen*h] // reuse qkvBuf since QKV split is done
		mhaInPlace(attnOut, q, kBuf, vBuf, ws.scores, seqLen, heads, headDim)

		// Output projection + residual + layernorm
		linearInPlace(temp, attnOut, layer.AttnOutW.Data(), layer.AttnOutB.Data(), seqLen, h, h)
		residualLayerNormInPlace(hidden, temp, layer.AttnLnW.Data(), layer.AttnLnB.Data(), seqLen, h, 1e-12)

		// FFN up: [seqLen, h] @ [h, inter] → [seqLen, inter]
		ffn := ws.ffnBuf[:seqLen*cfg.Intermediate]
		linearInPlace(ffn, hidden, layer.FfnInterW.Data(), layer.FfnInterB.Data(), seqLen, h, cfg.Intermediate)

		// GELU in-place
		geluInPlace(ffn)

		// FFN down: [seqLen, inter] @ [inter, h] → [seqLen, h]
		linearInPlace(temp, ffn, layer.FfnOutW.Data(), layer.FfnOutB.Data(), seqLen, cfg.Intermediate, h)
		residualLayerNormInPlace(hidden, temp, layer.FfnLnW.Data(), layer.FfnLnB.Data(), seqLen, h, 1e-12)
	}

	return hidden
}

// EmbedFast produces an L2-normalized embedding using pre-allocated workspace.
// Call InitWorkspace(maxSeqLen) once before first use.
func (m *BertModel) EmbedFast(tokenIDs []int, attnMask []bool) []float32 {
	ws := m.ws
	if ws == nil || ws.seqLen < len(tokenIDs) {
		ws = newWorkspace(len(tokenIDs), m.Config)
		m.ws = ws
	}
	hidden := m.ForwardFast(tokenIDs, ws)
	h := m.Config.HiddenSize
	seqLen := len(tokenIDs)

	out := ws.outEmb[:h]
	for i := range out {
		out[i] = 0
	}
	count := 0
	for s := 0; s < seqLen; s++ {
		if attnMask[s] {
			for d := 0; d < h; d++ {
				out[d] += hidden[s*h+d]
			}
			count++
		}
	}
	if count > 0 {
		inv := 1.0 / float32(count)
		for d := range out {
			out[d] *= inv
		}
	}
	norm := float32(0)
	for _, v := range out {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		inv := 1.0 / norm
		for i := range out {
			out[i] *= inv
		}
	}
	return out
}

// --- In-place kernels ---

// linearInPlace: out = x @ wT + bias (wT is pre-transposed [inDim, outDim])
func linearInPlace(out, x, wT, bias []float32, m, inDim, outDim int) {
	for i := range out[:m*outDim] {
		out[i] = 0
	}
	// Use gonum BLAS for NN matmul (matches gte-go's gonum path for best cache behavior)
	blas32.Implementation().Sgemm(
		blas.NoTrans, blas.NoTrans,
		m, outDim, inDim,
		1.0,
		x, inDim,
		wT, outDim,
		0.0,
		out, outDim,
	)
	if bias != nil {
		for i := 0; i < m; i++ {
			for j := 0; j < outDim; j++ {
				out[i*outDim+j] += bias[j]
			}
		}
	}
}

func layerNormInPlace(x, gamma, beta []float32, seqLen, h int, eps float32) {
	for s := 0; s < seqLen; s++ {
		off := s * h
		mean := float32(0)
		for d := 0; d < h; d++ {
			mean += x[off+d]
		}
		mean /= float32(h)
		variance := float32(0)
		for d := 0; d < h; d++ {
			diff := x[off+d] - mean
			variance += diff * diff
		}
		variance /= float32(h)
		stdInv := float32(1.0 / math.Sqrt(float64(variance+eps)))
		for d := 0; d < h; d++ {
			x[off+d] = gamma[d]*(x[off+d]-mean)*stdInv + beta[d]
		}
	}
}

func residualLayerNormInPlace(residual, x, gamma, beta []float32, seqLen, h int, eps float32) {
	for s := 0; s < seqLen; s++ {
		off := s * h
		mean := float32(0)
		for d := 0; d < h; d++ {
			v := x[off+d] + residual[off+d]
			x[off+d] = v
			mean += v
		}
		mean /= float32(h)
		variance := float32(0)
		for d := 0; d < h; d++ {
			diff := x[off+d] - mean
			variance += diff * diff
		}
		variance /= float32(h)
		stdInv := float32(1.0 / math.Sqrt(float64(variance+eps)))
		for d := 0; d < h; d++ {
			residual[off+d] = gamma[d]*(x[off+d]-mean)*stdInv + beta[d]
		}
	}
}

func geluInPlace(x []float32) {
	const c = float32(0.7978845608)
	for i, v := range x {
		arg := c * (v + 0.044715*v*v*v)
		var tanh float32
		if arg < -5 {
			tanh = -1
		} else if arg > 5 {
			tanh = 1
		} else {
			a2 := arg * arg
			tanh = arg * (135135 + a2*(17325+a2*(378+a2))) / (135135 + a2*(62370+a2*(3150+a2*28)))
		}
		x[i] = 0.5 * v * (1 + tanh)
	}
}

func mhaInPlace(out, q, k, v, scoresBuf []float32, seqLen, heads, headDim int) {
	hidden := heads * headDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for h := 0; h < heads; h++ {
		scores := scoresBuf[h*seqLen*seqLen : (h+1)*seqLen*seqLen]

		// Q·K^T per head
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				sum := float32(0)
				for d := 0; d < headDim; d++ {
					sum += q[i*hidden+h*headDim+d] * k[j*hidden+h*headDim+d]
				}
				scores[i*seqLen+j] = sum * scale
			}
		}

		// Softmax per row
		for i := 0; i < seqLen; i++ {
			row := scores[i*seqLen : (i+1)*seqLen]
			mx := row[0]
			for _, v := range row[1:] {
				if v > mx {
					mx = v
				}
			}
			sum := float32(0)
			for j := range row {
				row[j] = float32(math.Exp(float64(row[j] - mx)))
				sum += row[j]
			}
			inv := 1.0 / sum
			for j := range row {
				row[j] *= inv
			}
		}

		// Context: scores @ V per head
		for i := 0; i < seqLen; i++ {
			for d := 0; d < headDim; d++ {
				sum := float32(0)
				for j := 0; j < seqLen; j++ {
					sum += scores[i*seqLen+j] * v[j*hidden+h*headDim+d]
				}
				out[i*hidden+h*headDim+d] = sum
			}
		}
	}
}
