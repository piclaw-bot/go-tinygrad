package tensor

// Module interface for neural network layers.
type Module interface {
	Forward(x *Tensor) *Tensor
}

// LinearModule is a fully-connected layer: Y = X @ W^T + bias.
type LinearModule struct {
	Weight *Tensor // [outDim, inDim]
	Bias   *Tensor // [outDim] or nil
}

func NewLinear(inDim, outDim int) *LinearModule {
	// Xavier initialization
	scale := float32(1.0) / float32(inDim)
	w := Rand([]int{outDim, inDim})
	w.Realize()
	for i, v := range w.Data() {
		w.Data()[i] = (v - 0.5) * 2 * scale
	}
	bias := Zeros([]int{outDim})
	return &LinearModule{Weight: w, Bias: bias}
}

func (l *LinearModule) Forward(x *Tensor) *Tensor {
	return x.Linear(l.Weight, l.Bias)
}

// LayerNormModule applies layer normalization.
type LayerNormModule struct {
	Weight *Tensor // [dim]
	Bias   *Tensor // [dim]
	Eps    float32
}

func NewLayerNorm(dim int) *LayerNormModule {
	return &LayerNormModule{
		Weight: Ones([]int{dim}),
		Bias:   Zeros([]int{dim}),
		Eps:    1e-5,
	}
}

func (ln *LayerNormModule) Forward(x *Tensor) *Tensor {
	return x.LayerNorm(ln.Weight, ln.Bias, ln.Eps)
}

// EmbeddingModule is a lookup table.
type EmbeddingModule struct {
	Weight *Tensor // [vocabSize, dim]
}

func NewEmbedding(vocabSize, dim int) *EmbeddingModule {
	return &EmbeddingModule{Weight: Rand([]int{vocabSize, dim})}
}

func (e *EmbeddingModule) Forward(ids []int) *Tensor {
	return Embedding(e.Weight, ids)
}
