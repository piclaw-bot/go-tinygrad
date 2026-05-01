//go:build !cgo

package gpu

// Stub for non-CGo builds — GPU not available.

func Available() bool { return false }

type GpuBuffer struct{}

func Alloc(n int) *GpuBuffer                                             { return nil }
func (b *GpuBuffer) Free()                                                {}
func (b *GpuBuffer) Upload(data []float32)                                {}
func (b *GpuBuffer) Download(data []float32)                              {}
func SgemmNN(m, n, k int, alpha float32, A, B, C *GpuBuffer, lda, ldb, ldc int) {}
func SgemmNT(m, n, k int, alpha float32, A, B, C *GpuBuffer, lda, ldb, ldc int) {}
func Sync()                                                                {}
