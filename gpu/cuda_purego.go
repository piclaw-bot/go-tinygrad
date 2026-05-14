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
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// CUDA types
type CUdevice int32
type CUcontext uintptr
type CUmodule uintptr
type CUfunction uintptr
type CUdeviceptr uint64
type CUresult int32

const (
	CUDA_SUCCESS CUresult = 0
)

// Function pointers (populated by dlopen)
var (
	cuInit                   func(uint32) CUresult
	cuDeviceGet              func(*CUdevice, int32) CUresult
	cuDeviceGetName          func(unsafe.Pointer, int32, CUdevice) CUresult
	cuDeviceGetAttribute     func(*int32, int32, CUdevice) CUresult
	cuCtxCreate              func(*CUcontext, uint32, CUdevice) CUresult
	cuCtxDestroy             func(CUcontext) CUresult
	cuDevicePrimaryCtxRetain func(*CUcontext, CUdevice) CUresult
	cuMemAlloc               func(*CUdeviceptr, uint64) CUresult
	cuMemFree                func(CUdeviceptr) CUresult
	cuMemcpyHtoD             func(CUdeviceptr, unsafe.Pointer, uint64) CUresult
	cuMemcpyDtoH             func(unsafe.Pointer, CUdeviceptr, uint64) CUresult
	cuMemcpyHtoDAsync        func(CUdeviceptr, unsafe.Pointer, uint64, uintptr) CUresult
	cuMemcpyDtoHAsync        func(unsafe.Pointer, CUdeviceptr, uint64, uintptr) CUresult
	cuModuleLoadData         func(*CUmodule, unsafe.Pointer) CUresult
	cuModuleUnload           func(CUmodule) CUresult
	cuModuleGetFunction      func(*CUfunction, CUmodule, unsafe.Pointer) CUresult
	cuLaunchKernel           func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult
	cuCtxSynchronize         func() CUresult
	cuCtxSetCurrent          func(CUcontext) CUresult
	cuMemcpyDtoD             func(CUdeviceptr, CUdeviceptr, uint64) CUresult
	cuMemcpyDtoDAsync        func(CUdeviceptr, CUdeviceptr, uint64, uintptr) CUresult
)

var (
	gpuOnce    sync.Once
	gpuOK      bool
	gpuDev     CUdevice
	gpuCtx     CUcontext
	gpuName    string
	gpuSMs     int32
	gpuCCMajor int32
	gpuCCMinor int32
)

// Init attempts to load CUDA and initialize the GPU.
func Init() bool {
	gpuOnce.Do(func() {
		runtime.LockOSThread() // CUDA context is thread-local
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
				ok := false
				func() {
					defer func() { recover() }()
					purego.RegisterLibFunc(fptr, lib, name)
					ok = true
				}()
				if ok {
					return true // stop at first successful registration
				}
			}
			return false
		}

		purego.RegisterLibFunc(&cuInit, lib, "cuInit")
		regFn(&cuDeviceGet, lib, "cuDeviceGet")
		regFn(&cuDeviceGetName, lib, "cuDeviceGetName_v2", "cuDeviceGetName")
		regFn(&cuDeviceGetAttribute, lib, "cuDeviceGetAttribute")
		regFn(&cuCtxCreate, lib, "cuCtxCreate_v2", "cuCtxCreate")
		regFn(&cuCtxDestroy, lib, "cuCtxDestroy_v2", "cuCtxDestroy")
		regFn(&cuDevicePrimaryCtxRetain, lib, "cuDevicePrimaryCtxRetain")
		regFn(&cuMemAlloc, lib, "cuMemAlloc_v2", "cuMemAlloc")
		regFn(&cuMemFree, lib, "cuMemFree_v2", "cuMemFree")
		regFn(&cuMemcpyHtoD, lib, "cuMemcpyHtoD_v2", "cuMemcpyHtoD")
		regFn(&cuMemcpyDtoH, lib, "cuMemcpyDtoH_v2", "cuMemcpyDtoH")
		regFn(&cuMemcpyHtoDAsync, lib, "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync")
		regFn(&cuMemcpyDtoHAsync, lib, "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync")
		regFn(&cuModuleLoadData, lib, "cuModuleLoadData")
		regFn(&cuModuleUnload, lib, "cuModuleUnload")
		regFn(&cuModuleGetFunction, lib, "cuModuleGetFunction")
		regFn(&cuLaunchKernel, lib, "cuLaunchKernel")
		regFn(&cuCtxSynchronize, lib, "cuCtxSynchronize")
		regFn(&cuCtxSetCurrent, lib, "cuCtxSetCurrent")
		regFn(&cuMemcpyDtoD, lib, "cuMemcpyDtoD_v2", "cuMemcpyDtoD")
		regFn(&cuMemcpyDtoDAsync, lib, "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoDAsync")

		// Streams, events, graphs
		regFn(&cuMemGetInfo, lib, "cuMemGetInfo_v2", "cuMemGetInfo")
		regFn(&cuStreamCreate, lib, "cuStreamCreate")
		regFn(&cuStreamDestroy, lib, "cuStreamDestroy_v2", "cuStreamDestroy")
		regFn(&cuStreamSynchronize, lib, "cuStreamSynchronize")
		regFn(&cuEventCreate, lib, "cuEventCreate")
		regFn(&cuEventRecord, lib, "cuEventRecord")
		regFn(&cuEventSynchronize, lib, "cuEventSynchronize")
		regFn(&cuStreamWaitEvent, lib, "cuStreamWaitEvent")
		regFn(&cuEventDestroy, lib, "cuEventDestroy_v2", "cuEventDestroy")
		regFn(&cuStreamBeginCapture, lib, "cuStreamBeginCapture")
		regFn(&cuStreamEndCapture, lib, "cuStreamEndCapture")
		regFn(&cuGraphInstantiate, lib, "cuGraphInstantiate_v2", "cuGraphInstantiate")
		regFn(&cuGraphLaunch, lib, "cuGraphLaunch")
		regFn(&cuGraphDestroy, lib, "cuGraphDestroy")
		regFn(&cuGraphExecDestroy, lib, "cuGraphExecDestroy")

		// Initialize CUDA
		if r := cuInit(0); r != CUDA_SUCCESS {
			debugf("[gpu] cuInit failed: %d\n", r)
			return
		}

		// Get device 0
		if r := cuDeviceGet(&gpuDev, 0); r != CUDA_SUCCESS {
			debugf("[gpu] cuDeviceGet failed: %d\n", r)
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
		// CUDA compute capability attributes.
		cuDeviceGetAttribute(&gpuCCMajor, 75, gpuDev)
		cuDeviceGetAttribute(&gpuCCMinor, 76, gpuDev)

		// Create context
		if r := cuCtxCreate(&gpuCtx, 0, gpuDev); r != CUDA_SUCCESS {
			debugf("[gpu] cuCtxCreate failed: %d\n", r)
			return
		}

		gpuOK = true
		debugf("[gpu] %s (%d SMs) — pure Go, no CGo\n", gpuName, gpuSMs)
	})
	return gpuOK
}

// EnsureContext sets the CUDA context on the calling thread.
func EnsureContext() {
	if gpuOK && gpuCtx != 0 && cuCtxSetCurrent != nil {
		cuCtxSetCurrent(gpuCtx)
	}
}

// Available returns true if CUDA GPU is accessible.
func Available() bool {
	return Init()
}

// DeviceName returns the GPU name.
func DeviceName() string { return gpuName }

// SMCount returns the number of streaming multiprocessors.
func SMCount() int { return int(gpuSMs) }

// ComputeCapability returns the CUDA compute capability for device 0.
func ComputeCapability() (major, minor int) { return int(gpuCCMajor), int(gpuCCMinor) }

// Buffer holds GPU device memory.
type Buffer struct {
	Ptr  CUdeviceptr
	Size int
}

// Malloc allocates GPU memory for n float32s.
func Malloc(n int) (*Buffer, error) {
	if n <= 0 {
		return &Buffer{}, nil
	}
	maxInt := int(^uint(0) >> 1)
	if n > maxInt/4 {
		return nil, fmt.Errorf("cuMemAlloc size overflow for %d float32s", n)
	}
	EnsureContext()
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
		EnsureContext()
		cuMemFree(b.Ptr)
		b.Ptr = 0
	}
}

// Upload copies host data to GPU.
func (b *Buffer) Upload(data []float32) error {
	if b == nil {
		return fmt.Errorf("nil GPU buffer")
	}
	if len(data) == 0 {
		return nil
	}
	EnsureContext()
	r := cuMemcpyHtoD(b.Ptr, unsafe.Pointer(&data[0]), uint64(len(data)*4))
	runtime.KeepAlive(data) // prevent GC from moving data during CUDA memcpy
	if r != CUDA_SUCCESS {
		return fmt.Errorf("cuMemcpyHtoD: error %d", r)
	}
	return nil
}

// Download copies GPU data to host.
func (b *Buffer) Download(data []float32) error {
	if b == nil {
		return fmt.Errorf("nil GPU buffer")
	}
	if len(data) == 0 {
		return nil
	}
	EnsureContext()
	r := cuMemcpyDtoH(unsafe.Pointer(&data[0]), b.Ptr, uint64(len(data)*4))
	runtime.KeepAlive(data)
	if r != CUDA_SUCCESS {
		return fmt.Errorf("cuMemcpyDtoH: error %d", r)
	}
	return nil
}

// Sync waits for all GPU operations to complete.
func Sync() {
	EnsureContext()
	cuCtxSynchronize()
}

// SyncErr waits for all GPU operations and reports CUDA driver errors.
func SyncErr() error {
	EnsureContext()
	if r := cuCtxSynchronize(); r != CUDA_SUCCESS {
		return fmt.Errorf("cuCtxSynchronize: error %d", r)
	}
	return nil
}

var extraModules []CUmodule

func loadPTXModule(ptx string, kernelName string) (CUmodule, CUfunction, error) {
	ptxBytes := append([]byte(ptx), 0) // null-terminate
	var mod CUmodule
	if r := cuModuleLoadData(&mod, unsafe.Pointer(&ptxBytes[0])); r != CUDA_SUCCESS {
		return 0, 0, fmt.Errorf("cuModuleLoadData: error %d", r)
	}
	nameBytes := append([]byte(kernelName), 0)
	var fn CUfunction
	if r := cuModuleGetFunction(&fn, mod, unsafe.Pointer(&nameBytes[0])); r != CUDA_SUCCESS {
		if cuModuleUnload != nil {
			cuModuleUnload(mod)
		}
		return 0, 0, fmt.Errorf("cuModuleGetFunction(%s): error %d", kernelName, r)
	}
	return mod, fn, nil
}

// LoadPTX loads a PTX module and returns a kernel function by name.
// The backing module is retained until Shutdown() so the function pointer stays valid.
func LoadPTX(ptx string, kernelName string) (CUfunction, error) {
	mod, fn, err := loadPTXModule(ptx, kernelName)
	if err != nil {
		return 0, err
	}
	extraModules = append(extraModules, mod)
	return fn, nil
}

// LaunchKernel launches a CUDA kernel.
func LaunchKernel(fn CUfunction, gridX, gridY, gridZ, blockX, blockY, blockZ uint32, sharedMem uint32, args ...unsafe.Pointer) error {
	if cuLaunchKernel == nil {
		return fmt.Errorf("cuLaunchKernel unavailable")
	}
	if fn == 0 {
		return fmt.Errorf("invalid CUDA function")
	}
	if gridX == 0 || gridY == 0 || gridZ == 0 || blockX == 0 || blockY == 0 || blockZ == 0 {
		return fmt.Errorf("invalid CUDA launch dimensions grid=(%d,%d,%d) block=(%d,%d,%d)", gridX, gridY, gridZ, blockX, blockY, blockZ)
	}
	var argPtrs unsafe.Pointer
	if len(args) > 0 {
		argPtrs = unsafe.Pointer(&args[0])
	}
	if r := cuLaunchKernel(fn, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, 0, argPtrs, nil); r != CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel: error %d", r)
	}
	return nil
}

var cuMemGetInfo func(*uint64, *uint64) CUresult

func init() {
	// Will be registered in Init()
}

// MemInfo returns (free, total) GPU memory in bytes.
func MemInfo() (uint64, uint64) {
	EnsureContext()
	var free, total uint64
	if cuMemGetInfo == nil {
		return 0, 0
	}
	cuMemGetInfo(&free, &total)
	return free, total
}

// Shutdown releases global CUDA-side resources so a fresh context can be created.
// Intended primarily for tests and one-shot diagnostic processes.
func Shutdown() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if gpuOK {
		EnsureContext()
		SyncAll()
	}
	shutdownCompiledKernels()
	shutdownNativeBF16()
	shutdownMegaModule()
	shutdownStreams()
	for _, mod := range extraModules {
		if mod != 0 && cuModuleUnload != nil {
			cuModuleUnload(mod)
		}
	}
	extraModules = nil
	if gpuCtx != 0 && cuCtxDestroy != nil {
		cuCtxDestroy(gpuCtx)
	}
	gpuCtx = 0
	gpuDev = 0
	gpuOK = false
	gpuName = ""
	gpuSMs = 0
	gpuOnce = sync.Once{}
}
