package tensor

import "math"

// Softmax computes softmax along the last axis.
func (t *Tensor) Softmax() *Tensor {
	t.Realize()
	data := t.Data()
	shape := t.Shape()
	ndim := len(shape)
	lastDim := shape[ndim-1]
	outerSize := t.Numel() / lastDim

	out := make([]float32, len(data))
	for i := 0; i < outerSize; i++ {
		off := i * lastDim
		row := data[off : off+lastDim]

		// Max for numerical stability
		mx := row[0]
		for _, v := range row[1:] {
			if v > mx {
				mx = v
			}
		}
		// Exp and sum
		sum := float32(0)
		for j, v := range row {
			e := float32(math.Exp(float64(v - mx)))
			out[off+j] = e
			sum += e
		}
		inv := 1.0 / sum
		for j := 0; j < lastDim; j++ {
			out[off+j] *= inv
		}
	}
	return FromFloat32(out, shape)
}

// LayerNorm computes layer normalization along the last axis.
func (t *Tensor) LayerNorm(gamma, beta *Tensor, eps float32) *Tensor {
	t.Realize()
	data := t.Data()
	shape := t.Shape()
	lastDim := shape[len(shape)-1]
	outerSize := t.Numel() / lastDim

	var g, b []float32
	if gamma != nil {
		g = gamma.Data()
	}
	if beta != nil {
		b = beta.Data()
	}

	out := make([]float32, len(data))
	for i := 0; i < outerSize; i++ {
		off := i * lastDim
		row := data[off : off+lastDim]

		// Mean
		mean := float32(0)
		for _, v := range row {
			mean += v
		}
		mean /= float32(lastDim)

		// Variance
		variance := float32(0)
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(lastDim)
		stdInv := float32(1.0 / math.Sqrt(float64(variance+eps)))

		for j := 0; j < lastDim; j++ {
			v := (row[j] - mean) * stdInv
			if g != nil {
				v = g[j]*v + b[j]
			}
			out[off+j] = v
		}
	}
	return FromFloat32(out, shape)
}

// GELU computes the GELU activation (tanh approximation).
func (t *Tensor) GELU() *Tensor {
	t.Realize()
	data := t.Data()
	out := make([]float32, len(data))
	const c = float32(0.7978845608) // sqrt(2/pi)
	for i, v := range data {
		out[i] = 0.5 * v * (1 + float32(math.Tanh(float64(c*(v+0.044715*v*v*v)))))
	}
	return FromFloat32(out, t.Shape())
}
