package gpu

// Pure Go CUDA bindings via dlopen — no CGo required.
// Loads libcuda.so.1 at runtime; falls back gracefully if not present.
//
// This wraps just the minimal CUDA Driver API needed for GEMM:
//   cuInit, cuDeviceGet, cuDeviceGetName, cuDeviceGetAttribute,
//   cuCtxCreate, cuMemAlloc, cuMemcpyHtoD, cuMemcpyDtoH, cuMemFree,
//   cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel, cuCtxSynchronize

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// CUDA types
type CUdevice int32
type CUcontext uintptr
type CUmodule uintptr
type CUfunction uintptr
type CUdeviceptr uintptr
type CUresult int32

const (
	CUDA_SUCCESS CUresult = 0
)

// Function pointers (populated by dlopen)
var (
	cuInit               func(uint32) CUresult
	cuDeviceGet          func(*CUdevice, int32) CUresult
	cuDeviceGetName      func(unsafe.Pointer, int32, CUdevice) CUresult
	cuDeviceGetAttribute func(*int32, int32, CUdevice) CUresult
	cuCtxCreate          func(*CUcontext, uint32, CUdevice) CUresult
	cuMemAlloc           func(*CUdeviceptr, uint64) CUresult
	cuMemFree            func(CUdeviceptr) CUresult
	cuMemcpyHtoD         func(CUdeviceptr, unsafe.Pointer, uint64) CUresult
	cuMemcpyDtoH         func(unsafe.Pointer, CUdeviceptr, uint64) CUresult
	cuModuleLoadData     func(*CUmodule, unsafe.Pointer) CUresult
	cuModuleGetFunction  func(*CUfunction, CUmodule, unsafe.Pointer) CUresult
	cuLaunchKernel       func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult
	cuCtxSynchronize     func() CUresult
)

var (
	gpuOnce sync.Once
	gpuOK   bool
	gpuDev  CUdevice
	gpuCtx  CUcontext
	gpuName string
	gpuSMs  int32
)

// Init attempts to load CUDA and initialize the GPU.
func Init() bool {
	gpuOnce.Do(func() {
		lib, err := purego.Dlopen("libcuda.so.1", purego.RTLD_LAZY)
		if err != nil {
			// Try versioned names
			lib, err = purego.Dlopen("libcuda.so", purego.RTLD_LAZY)
			if err != nil {
				return // No CUDA driver
			}
		}

		// Register all function pointers
		// Use helper to try versioned then non-versioned names
		regFn := func(fptr interface{}, lib uintptr, names ...string) bool {
			for _, name := range names {
				func() {
					defer func() { recover() }() // purego panics on missing symbol
					purego.RegisterLibFunc(fptr, lib, name)
				}()
			}
			return true
		}

		purego.RegisterLibFunc(&cuInit, lib, "cuInit")
		regFn(&cuDeviceGet, lib, "cuDeviceGet")
		regFn(&cuDeviceGetName, lib, "cuDeviceGetName_v2", "cuDeviceGetName")
		regFn(&cuDeviceGetAttribute, lib, "cuDeviceGetAttribute")
		regFn(&cuCtxCreate, lib, "cuCtxCreate_v2", "cuCtxCreate")
		regFn(&cuMemAlloc, lib, "cuMemAlloc_v2", "cuMemAlloc")
		regFn(&cuMemFree, lib, "cuMemFree_v2", "cuMemFree")
		regFn(&cuMemcpyHtoD, lib, "cuMemcpyHtoD_v2", "cuMemcpyHtoD")
		regFn(&cuMemcpyDtoH, lib, "cuMemcpyDtoH_v2", "cuMemcpyDtoH")
		regFn(&cuModuleLoadData, lib, "cuModuleLoadData")
		regFn(&cuModuleGetFunction, lib, "cuModuleGetFunction")
		regFn(&cuLaunchKernel, lib, "cuLaunchKernel")
		regFn(&cuCtxSynchronize, lib, "cuCtxSynchronize")

		// Initialize CUDA
		if r := cuInit(0); r != CUDA_SUCCESS {
			fmt.Printf("[gpu] cuInit failed: %d\n", r)
			return
		}

		// Get device 0
		if r := cuDeviceGet(&gpuDev, 0); r != CUDA_SUCCESS {
			fmt.Printf("[gpu] cuDeviceGet failed: %d\n", r)
			return
		}

		// Get device name
		nameBuf := make([]byte, 256)
		if r := cuDeviceGetName(unsafe.Pointer(&nameBuf[0]), 256, gpuDev); r == CUDA_SUCCESS {
			for i, b := range nameBuf {
				if b == 0 {
					gpuName = string(nameBuf[:i])
					break
				}
			}
		}

		// Get SM count (attribute 16 = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
		cuDeviceGetAttribute(&gpuSMs, 16, gpuDev)

		// Create context
		if r := cuCtxCreate(&gpuCtx, 0, gpuDev); r != CUDA_SUCCESS {
			fmt.Printf("[gpu] cuCtxCreate failed: %d\n", r)
			return
		}

		gpuOK = true
		fmt.Printf("[gpu] %s (%d SMs) — pure Go, no CGo\n", gpuName, gpuSMs)
	})
	return gpuOK
}

// Available returns true if CUDA GPU is accessible.
func Available() bool {
	return Init()
}

// DeviceName returns the GPU name.
func DeviceName() string { return gpuName }

// SMCount returns the number of streaming multiprocessors.
func SMCount() int { return int(gpuSMs) }

// Buffer holds GPU device memory.
type Buffer struct {
	Ptr  CUdeviceptr
	Size int
}

// Malloc allocates GPU memory for n float32s.
func Malloc(n int) (*Buffer, error) {
	// Ensure context is fully initialized (PTX load finalizes context setup)
	SgemmReady()
	var ptr CUdeviceptr
	size := uint64(n * 4)
	if r := cuMemAlloc(&ptr, size); r != CUDA_SUCCESS {
		return nil, fmt.Errorf("cuMemAlloc(%d): error %d", size, r)
	}
	return &Buffer{Ptr: ptr, Size: n * 4}, nil
}

// Free releases GPU memory.
func (b *Buffer) Free() {
	if b.Ptr != 0 {
		cuMemFree(b.Ptr)
		b.Ptr = 0
	}
}

// Upload copies host data to GPU.
func (b *Buffer) Upload(data []float32) error {
	if r := cuMemcpyHtoD(b.Ptr, unsafe.Pointer(&data[0]), uint64(len(data)*4)); r != CUDA_SUCCESS {
		return fmt.Errorf("cuMemcpyHtoD: error %d", r)
	}
	return nil
}

// Download copies GPU data to host.
func (b *Buffer) Download(data []float32) error {
	if r := cuMemcpyDtoH(unsafe.Pointer(&data[0]), b.Ptr, uint64(len(data)*4)); r != CUDA_SUCCESS {
		return fmt.Errorf("cuMemcpyDtoH: error %d", r)
	}
	return nil
}

// Sync waits for all GPU operations to complete.
func Sync() {
	cuCtxSynchronize()
}

// LoadPTX loads a PTX module and returns a kernel function by name.
func LoadPTX(ptx string, kernelName string) (CUfunction, error) {
	ptxBytes := append([]byte(ptx), 0) // null-terminate
	var mod CUmodule
	if r := cuModuleLoadData(&mod, unsafe.Pointer(&ptxBytes[0])); r != CUDA_SUCCESS {
		return 0, fmt.Errorf("cuModuleLoadData: error %d", r)
	}
	nameBytes := append([]byte(kernelName), 0)
	var fn CUfunction
	if r := cuModuleGetFunction(&fn, mod, unsafe.Pointer(&nameBytes[0])); r != CUDA_SUCCESS {
		return 0, fmt.Errorf("cuModuleGetFunction(%s): error %d", kernelName, r)
	}
	return fn, nil
}

// LaunchKernel launches a CUDA kernel.
func LaunchKernel(fn CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ uint32, sharedMem uint32, args ...unsafe.Pointer) error {
	var argPtrs unsafe.Pointer
	if len(args) > 0 {
		argPtrs = unsafe.Pointer(&args[0])
	}
	if r := cuLaunchKernel(fn, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, 0, argPtrs, nil); r != CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel: error %d", r)
	}
	return nil
}
