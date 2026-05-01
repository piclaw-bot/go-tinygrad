package tensor

import (
	"fmt"

	"github.com/rcarmo/go-tinygrad/simd"
)

// MatMul computes matrix multiplication: C = A @ B.
// A is [M, K], B is [K, N], result is [M, N].
// For batched: A is [..., M, K], B is [..., K, N].
func (t *Tensor) MatMul(other *Tensor) *Tensor {
	a, b := t, other
	aDims := a.Shape()
	bDims := b.Shape()

	if len(aDims) < 2 || len(bDims) < 2 {
		panic("matmul requires at least 2D tensors")
	}

	m := aDims[len(aDims)-2]
	k1 := aDims[len(aDims)-1]
	k2 := bDims[len(bDims)-2]
	n := bDims[len(bDims)-1]

	if k1 != k2 {
		panic(fmt.Sprintf("matmul: inner dims mismatch: %d vs %d", k1, k2))
	}
	k := k1

	// For now: eager 2D matmul using SIMD from gte-go
	a.Realize()
	b.Realize()

	aData := a.Data()
	bData := b.Data()
	cData := make([]float32, m*n)

	// Use SIMD dot product for each output element
	for i := 0; i < m; i++ {
		aRow := aData[i*k : i*k+k]
		for j := 0; j < n; j++ {
			// Gather B column j
			sum := float32(0)
			if k >= 8 {
				// Use SIMD for the dot product along k
				// B is row-major [K, N], so B[:,j] is strided
				bCol := make([]float32, k)
				for p := 0; p < k; p++ {
					bCol[p] = bData[p*n+j]
				}
				sum = simd.Sdot(aRow, bCol)
			} else {
				for p := 0; p < k; p++ {
					sum += aRow[p] * bData[p*n+j]
				}
			}
			cData[i*n+j] = sum
		}
	}

	outShape := []int{m, n}
	return FromFloat32(cData, outShape)
}

// Linear computes Y = X @ W^T + bias.
// X is [M, K], W is [N, K] (transposed), bias is [N].
func (t *Tensor) Linear(weight, bias *Tensor) *Tensor {
	// W^T: transpose last two dims
	wT := weight.Permute([]int{1, 0})
	result := t.MatMul(wT)
	if bias != nil {
		result = result.Add(bias)
	}
	return result
}
