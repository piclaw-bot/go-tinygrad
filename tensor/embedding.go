package tensor

// Embedding performs token ID → vector lookup.
// weight is [vocabSize, dim], ids is a flat slice of token IDs.
// Returns [len(ids), dim].
func Embedding(weight *Tensor, ids []int) *Tensor {
	if weight == nil {
		panic("embedding: nil weight")
	}
	dims := weight.Shape()
	if len(dims) != 2 || dims[0] < 0 || dims[1] < 0 {
		panic("embedding: weight must be 2D")
	}
	weight.Realize()
	wData := weight.Data()
	vocabSize, dim := dims[0], dims[1]
	n := len(ids)
	out := make([]float32, n*dim)
	for i, id := range ids {
		if id < 0 || id >= vocabSize {
			panic("embedding: token id out of range")
		}
		copy(out[i*dim:(i+1)*dim], wData[id*dim:(id+1)*dim])
	}
	return FromFloat32(out, []int{n, dim})
}
