//go:build cgo

package gpu

// #cgo LDFLAGS: -L/usr/local/cuda-12.3/lib64 -L/usr/local/cuda-12.3/targets/x86_64-linux/lib -L/usr/local/cuda-12.3/targets/x86_64-linux/lib/stubs -lcublas -lcudart -lcuda
// #cgo CFLAGS: -I/usr/local/cuda-12.3/include -I/usr/local/cuda-12.3/targets/x86_64-linux/include
// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include <stdlib.h>
//
// // Wrapper functions to avoid Go handling CUDA types directly
// static int gpu_init(cublasHandle_t* handle) {
//     cudaError_t cerr = cudaSetDevice(0);
//     if (cerr != cudaSuccess) return -1;
//     cublasStatus_t status = cublasCreate(handle);
//     return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -2;
// }
//
// static int gpu_sgemm(cublasHandle_t handle, int m, int n, int k,
//                      float alpha, float* A, int lda, float* B, int ldb,
//                      float beta, float* C, int ldc) {
//     // cuBLAS uses column-major, so we swap A and B to get row-major result
//     cublasStatus_t s = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//         n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc);
//     return (s == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
// }
//
// static int gpu_sgemm_nt(cublasHandle_t handle, int m, int n, int k,
//                         float alpha, float* A, int lda, float* B, int ldb,
//                         float beta, float* C, int ldc) {
//     cublasStatus_t s = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
//         n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc);
//     return (s == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
// }
//
// static float* gpu_alloc(size_t bytes) {
//     float* ptr;
//     if (cudaMalloc((void**)&ptr, bytes) != cudaSuccess) return NULL;
//     return ptr;
// }
//
// static void gpu_free(float* ptr) { cudaFree(ptr); }
// static int gpu_upload(float* dst, float* src, size_t bytes) { return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice); }
// static int gpu_download(float* dst, float* src, size_t bytes) { return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost); }
// static void gpu_sync() { cudaDeviceSynchronize(); }
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

var (
	handle  C.cublasHandle_t
	gpuOnce sync.Once
	gpuOK   bool
)

// Available returns true if CUDA GPU is accessible.
func Available() bool {
	gpuOnce.Do(func() {
		ret := C.gpu_init(&handle)
		gpuOK = (ret == 0)
		if gpuOK {
			fmt.Println("[gpu] cuBLAS initialized on GPU 0")
		}
	})
	return gpuOK
}

// GpuBuffer holds device memory.
type GpuBuffer struct {
	Ptr  *C.float
	Size int // bytes
}

// Alloc allocates GPU memory.
func Alloc(n int) *GpuBuffer {
	bytes := n * 4
	ptr := C.gpu_alloc(C.size_t(bytes))
	if ptr == nil {
		return nil
	}
	return &GpuBuffer{Ptr: ptr, Size: bytes}
}

// Free releases GPU memory.
func (b *GpuBuffer) Free() {
	if b.Ptr != nil {
		C.gpu_free(b.Ptr)
		b.Ptr = nil
	}
}

// Upload copies host data to GPU.
func (b *GpuBuffer) Upload(data []float32) {
	C.gpu_upload(b.Ptr, (*C.float)(unsafe.Pointer(&data[0])), C.size_t(len(data)*4))
}

// Download copies GPU data to host.
func (b *GpuBuffer) Download(data []float32) {
	C.gpu_download((*C.float)(unsafe.Pointer(&data[0])), b.Ptr, C.size_t(len(data)*4))
}

// SgemmNN computes C = alpha*A*B + beta*C on GPU (row-major).
func SgemmNN(m, n, k int, alpha float32, A, B, C_buf *GpuBuffer, lda, ldb, ldc int) {
	beta := float32(0)
	C.gpu_sgemm(handle, C.int(m), C.int(n), C.int(k),
		C.float(alpha), A.Ptr, C.int(lda), B.Ptr, C.int(ldb),
		C.float(beta), C_buf.Ptr, C.int(ldc))
}

// SgemmNT computes C = alpha*A*B^T + beta*C on GPU (row-major).
func SgemmNT(m, n, k int, alpha float32, A, B, C_buf *GpuBuffer, lda, ldb, ldc int) {
	beta := float32(0)
	C.gpu_sgemm_nt(handle, C.int(m), C.int(n), C.int(k),
		C.float(alpha), A.Ptr, C.int(lda), B.Ptr, C.int(ldb),
		C.float(beta), C_buf.Ptr, C.int(ldc))
}

// Sync waits for all GPU operations to complete.
func Sync() {
	C.gpu_sync()
}
