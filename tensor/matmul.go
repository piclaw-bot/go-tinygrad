package tensor

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-tinygrad/simd"
)

// MatMul computes matrix multiplication: C = A @ B.
// A is [M, K], B is [K, N], result is [M, N].
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

	a.Realize()
	b.Realize()

	aData := a.Data()
	bData := b.Data()
	cData := make([]float32, m*n)

	// Use SIMD GEMM: C = A @ B is sgemm(NoTrans, NoTrans, m, n, k, 1, A, k, B, n, 0, C, n)
	if simd.HasSgemmAsm {
		simd.SgemmNN(m, n, k, 1.0,
			unsafe.Pointer(&aData[0]), unsafe.Pointer(&bData[0]), unsafe.Pointer(&cData[0]),
			k, n, n)
	} else {
		// Scalar fallback
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for p := 0; p < k; p++ {
					sum += aData[i*k+p] * bData[p*n+j]
				}
				cData[i*n+j] = sum
			}
		}
	}

	return FromFloat32(cData, []int{m, n})
}

// MatMulTransposed computes C = A @ B^T.
// A is [M, K], B is [N, K] (transposed), result is [M, N].
// This is the common pattern for linear layers: Y = X @ W^T.
func (t *Tensor) MatMulTransposed(other *Tensor) *Tensor {
	a, b := t, other
	aDims := a.Shape()
	bDims := b.Shape()

	m := aDims[len(aDims)-2]
	k := aDims[len(aDims)-1]
	n := bDims[len(bDims)-2] // B is [N, K], so N is dim 0

	if k != bDims[len(bDims)-1] {
		panic(fmt.Sprintf("matmul_t: inner dims mismatch: %d vs %d", k, bDims[len(bDims)-1]))
	}

	a.Realize()
	b.Realize()

	aData := a.Data()
	bData := b.Data()
	cData := make([]float32, m*n)

	// C = A @ B^T is sgemm(NoTrans, Trans)
	if simd.HasSgemmAsm {
		// Use gather on amd64, GEBP on arm64
		simd.SgemmNT(m, n, k, 1.0,
			unsafe.Pointer(&aData[0]), unsafe.Pointer(&bData[0]), unsafe.Pointer(&cData[0]),
			k, k, n)
	} else {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				sum := float32(0)
				for p := 0; p < k; p++ {
					sum += aData[i*k+p] * bData[j*k+p]
				}
				cData[i*n+j] = sum
			}
		}
	}

	return FromFloat32(cData, []int{m, n})
}

// Linear computes Y = X @ W^T + bias.
// X is [M, K], W is [N, K] (transposed), bias is [N] or nil.
func (t *Tensor) Linear(weight, bias *Tensor) *Tensor {
	result := t.MatMulTransposed(weight)
	if bias != nil {
		// Broadcast add: [M, N] + [N] → need to broadcast bias
		result.Realize()
		bData := bias.Data()
		rData := result.Data()
		n := result.Shape()[1]
		m := result.Shape()[0]
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				rData[i*n+j] += bData[j]
			}
		}
	}
	return result
}
