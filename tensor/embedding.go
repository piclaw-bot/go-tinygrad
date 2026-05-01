package tensor

// Embedding performs token ID → vector lookup.
// weight is [vocabSize, dim], ids is a flat slice of token IDs.
// Returns [len(ids), dim].
func Embedding(weight *Tensor, ids []int) *Tensor {
	weight.Realize()
	wData := weight.Data()
	dim := weight.Shape()[1]
	n := len(ids)
	out := make([]float32, n*dim)
	for i, id := range ids {
		copy(out[i*dim:(i+1)*dim], wData[id*dim:(id+1)*dim])
	}
	return FromFloat32(out, []int{n, dim})
}
