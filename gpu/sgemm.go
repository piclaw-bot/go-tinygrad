package gpu

import (
	"fmt"
	"sync"
	"unsafe"
)

var (
	sgemmOnce sync.Once
	sgemmFn   CUfunction
	sgemmOK   bool
)


// SgemmReady returns true if GPU SGEMM is available.

func Sgemm(M, N, K int, alpha float32, A, B, C *Buffer) error {
	if !SgemmReady() {
		return fmt.Errorf("GPU SGEMM not available")
	}

	m := uint32(M)
	n := uint32(N)
	k := uint32(K)

	args := []unsafe.Pointer{
		unsafe.Pointer(&A.Ptr),
		unsafe.Pointer(&B.Ptr),
		unsafe.Pointer(&C.Ptr),
		unsafe.Pointer(&m),
		unsafe.Pointer(&n),
		unsafe.Pointer(&k),
		unsafe.Pointer(&alpha),
	}

	gridX := (uint32(N) + 15) / 16
	gridY := (uint32(M) + 15) / 16

	return LaunchKernel(sgemmFn, gridX, gridY, 1, 16, 16, 1, 0, args...)
}

// SgemmHost computes C = alpha * A * B on GPU from host data.
// Handles upload, compute, download, and cleanup.
// A is [M,K], B is [K,N], C (output) is [M,N].
func SgemmHost(M, N, K int, alpha float32, A, B []float32) ([]float32, error) {
	if !SgemmReady() {
		return nil, fmt.Errorf("GPU SGEMM not available")
	}

	dA, err := Malloc(M * K)
	if err != nil {
		return nil, err
	}
	defer dA.Free()

	dB, err := Malloc(K * N)
	if err != nil {
		return nil, err
	}
	defer dB.Free()

	dC, err := Malloc(M * N)
	if err != nil {
		return nil, err
	}
	defer dC.Free()

	if err := dA.Upload(A); err != nil {
		return nil, err
	}
	if err := dB.Upload(B); err != nil {
		return nil, err
	}

	if err := Sgemm(M, N, K, alpha, dA, dB, dC); err != nil {
		return nil, err
	}
	Sync()

	out := make([]float32, M*N)
	if err := dC.Download(out); err != nil {
		return nil, err
	}
	return out, nil
}
