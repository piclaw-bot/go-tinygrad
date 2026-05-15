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
	if M <= 0 || N <= 0 || K <= 0 || A == nil || B == nil || C == nil || A.Ptr == 0 || B.Ptr == 0 || C.Ptr == 0 {
		return fmt.Errorf("invalid SGEMM inputs M=%d N=%d K=%d", M, N, K)
	}
	mk, okMK := checkedMulInt(M, K)
	kn, okKN := checkedMulInt(K, N)
	mn, okMN := checkedMulInt(M, N)
	mkBytes, errMK := checkedByteSize(mk, -1)
	knBytes, errKN := checkedByteSize(kn, -1)
	mnBytes, errMN := checkedByteSize(mn, -1)
	if !okMK || !okKN || !okMN || errMK != nil || errKN != nil || errMN != nil || A.Size < int(mkBytes) || B.Size < int(knBytes) || C.Size < int(mnBytes) {
		return fmt.Errorf("invalid SGEMM buffer sizes")
	}
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
	mk, okMK := checkedMulInt(M, K)
	kn, okKN := checkedMulInt(K, N)
	mn, okMN := checkedMulInt(M, N)
	if M <= 0 || N <= 0 || K <= 0 || !okMK || !okKN || !okMN || len(A) < mk || len(B) < kn {
		return nil, fmt.Errorf("invalid SGEMM host inputs M=%d N=%d K=%d", M, N, K)
	}
	if !SgemmReady() {
		return nil, fmt.Errorf("GPU SGEMM not available")
	}

	dA, err := Malloc(mk)
	if err != nil {
		return nil, err
	}
	defer dA.Free()

	dB, err := Malloc(kn)
	if err != nil {
		return nil, err
	}
	defer dB.Free()

	dC, err := Malloc(mn)
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

	out := make([]float32, mn)
	if err := dC.Download(out); err != nil {
		return nil, err
	}
	return out, nil
}
