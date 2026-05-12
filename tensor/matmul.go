package tensor

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/simd"
)

// MatMul computes matrix multiplication: C = A @ B.
// A is [M, K], B is [K, N], result is [M, N].
func (t *Tensor) MatMul(other *Tensor) *Tensor {
	if t == nil || other == nil {
		panic("matmul: nil tensor")
	}
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

	if m < 0 || n < 0 || k1 < 0 || k2 < 0 {
		panic("matmul: invalid dimensions")
	}
	if k1 != k2 {
		panic(fmt.Sprintf("matmul: inner dims mismatch: %d vs %d", k1, k2))
	}
	outSize, ok := checkedMulInt(m, n)
	if !ok {
		panic("matmul: output shape overflows")
	}
	k := k1

	a.Realize()
	b.Realize()

	aData := a.Data()
	bData := b.Data()
	if len(aData) < shapeSize(aDims) || len(bData) < shapeSize(bDims) {
		panic("matmul: invalid backing data")
	}
	cData := make([]float32, outSize)

	// Use SIMD GEMM: C = A @ B is sgemm(NoTrans, NoTrans, m, n, k, 1, A, k, B, n, 0, C, n)
	if simd.HasSgemmAsm && len(aData) > 0 && len(bData) > 0 && len(cData) > 0 {
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
	if t == nil || other == nil {
		panic("matmul_t: nil tensor")
	}
	a, b := t, other
	aDims := a.Shape()
	bDims := b.Shape()
	if len(aDims) < 2 || len(bDims) < 2 {
		panic("matmul_t requires at least 2D tensors")
	}

	m := aDims[len(aDims)-2]
	k := aDims[len(aDims)-1]
	n := bDims[len(bDims)-2] // B is [N, K], so N is dim 0

	if m < 0 || n < 0 || k < 0 || bDims[len(bDims)-1] < 0 {
		panic("matmul_t: invalid dimensions")
	}
	if k != bDims[len(bDims)-1] {
		panic(fmt.Sprintf("matmul_t: inner dims mismatch: %d vs %d", k, bDims[len(bDims)-1]))
	}
	outSize, ok := checkedMulInt(m, n)
	if !ok {
		panic("matmul_t: output shape overflows")
	}

	a.Realize()
	b.Realize()

	aData := a.Data()
	bData := b.Data()
	if len(aData) < shapeSize(aDims) || len(bData) < shapeSize(bDims) {
		panic("matmul_t: invalid backing data")
	}
	cData := make([]float32, outSize)

	// C = A @ B^T is sgemm(NoTrans, Trans)
	if simd.HasSgemmAsm && len(aData) > 0 && len(bData) > 0 && len(cData) > 0 {
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

// Linear computes Y = X @ W^T + bias (standard convention).
// X is [M, K], W is [N, K], bias is [N] or nil.
func (t *Tensor) Linear(weight, bias *Tensor) *Tensor {
	result := t.MatMulTransposed(weight)
	if bias != nil {
		// Broadcast add: [M, N] + [N] → need to broadcast bias
		result.Realize()
		bDims := bias.Shape()
		n := result.Shape()[1]
		if len(bDims) != 1 || bDims[0] != n {
			panic("linear: bias shape mismatch")
		}
		bData := bias.Data()
		rData := result.Data()
		m := result.Shape()[0]
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				rData[i*n+j] += bData[j]
			}
		}
	}
	return result
}

// LinearPreT computes Y = X @ W_T + bias where W_T is already transposed [K, N].
// Used by models that pre-transpose weights at load time for SgemmNN performance.
func (t *Tensor) LinearPreT(weightT, bias *Tensor) *Tensor {
	result := t.MatMul(weightT)
	if bias != nil {
		result.Realize()
		bDims := bias.Shape()
		n := result.Shape()[1]
		if len(bDims) != 1 || bDims[0] != n {
			panic("linear: bias shape mismatch")
		}
		bData := bias.Data()
		rData := result.Data()
		m := result.Shape()[0]
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				rData[i*n+j] += bData[j]
			}
		}
	}
	return result
}
