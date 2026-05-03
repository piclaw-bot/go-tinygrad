# GPU Compute Options for go-pherence

## Available Hardware
- **NVIDIA GeForce RTX 3060** (12GB VRAM, Ampere, 3584 CUDA cores, 12.7 TFLOPS FP32)
- Driver 580.126.09
- Device nodes: `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/dri/renderD128`

## Options Ranked by Feasibility

### 1. Direct NVIDIA ioctl (pure Go) — THE TINYGRAD WAY ⭐
**No C. No CGo. No libcuda. Pure Go + syscall.**

tinygrad itself does this: talks to `/dev/nvidiactl` + `/dev/nvidia-uvm` via raw
`ioctl()` syscalls. Go's `syscall.Syscall` / `unix.IoctlSetPointerInt` can do this.

How it works:
1. Open `/dev/nvidiactl`, `/dev/nvidia0` — allocate GPU context
2. Open `/dev/nvidia-uvm` — unified virtual memory (GPU malloc)
3. Compile PTX → CUBIN (need `nvptxas` or ship pre-compiled)
4. Submit compute commands via ioctl command buffers
5. Synchronize and read results

**Pros:** Zero dependencies. Static binary. Full GPU control.
**Cons:** Must reverse-engineer / adapt ~2000 lines of ioctl structs from
NVIDIA's open-source kernel module headers. Driver-version sensitive.
**Effort:** High (2-3 days), but tinygrad's `runtime/ops_nv.py` is the blueprint.
**Reference:** https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_nv.py

### 2. Vulkan Compute via Go syscall (pure Go)
Vulkan's Linux interface is `/dev/dri/renderD128` (DRM). In theory, pure Go:
1. Open render node
2. DRM ioctls for GPU memory alloc
3. Submit SPIR-V compute shaders via command buffers

But Vulkan has a massive API surface (hundreds of entry points via `libvulkan.so`).
The DRM layer is simpler but NVIDIA's DRM support for compute is limited vs their
proprietary path.

**Pros:** Cross-vendor (AMD, Intel, NVIDIA). Standard API.
**Cons:** Still large; Vulkan loader (`libvulkan.so`) is practically required.
NVIDIA Vulkan compute perf ~90% of CUDA for GEMM.
**Effort:** Very high. Better to use CGo Vulkan bindings.

### 3. OpenCL via /dev/nvidia* (needs libOpenCL)
Similar to cuBLAS but more portable. Still needs a C library though.

### 4. CUDA via dlopen (Go + syscall, no CGo)
Load `libcuda.so` at runtime using `dlopen`/`dlsym` from pure Go:
```go
lib, _ := purego.Dlopen("libcuda.so.1", purego.RTLD_LAZY)
purego.RegisterLibFunc(&cuInit, lib, "cuInit")
```
Using [purego](https://github.com/nicholasgasior/purego) or Go's `plugin` mechanism.

**Pros:** Full CUDA API. No CGo. Static binary + runtime GPU support.
**Cons:** Still needs `libcuda.so.1` on the host (driver library).
**Effort:** Medium (1 day). Clean, practical.

### 5. Pre-compiled PTX kernels + minimal driver shim (pure Go)
Write SGEMM kernels in PTX assembly (NVIDIA's GPU ISA), ship them as
embedded strings, and use approach #1 or #4 to launch them.

```ptx
.version 8.0
.target sm_86  // Ampere
.address_size 64

.visible .entry sgemm_128x128(
    .param .u64 A, .param .u64 B, .param .u64 C,
    .param .u32 M, .param .u32 N, .param .u32 K
) {
    // Tiled SGEMM using shared memory
    ...
}
```

### 6. CPU-only: wider SIMD (AVX-512)
Our current AVX2+FMA kernels process 8 floats/cycle. AVX-512 does 16.
The RTX 3060 CPU in this system might support AVX-512.

```
$ cat /proc/cpuinfo | grep avx512
```

## Recommendation

**Phase 1 (immediate):** Option #4 — `purego` dlopen of `libcuda.so.1`
- No CGo, static Go binary, GPU at runtime if driver present
- Wrap just 10 functions: cuInit, cuDeviceGet, cuCtxCreate, cuMemAlloc,
  cuMemcpyHtoD, cuMemcpyDtoH, cuModuleLoadData, cuModuleGetFunction,
  cuLaunchKernel, cuCtxSynchronize
- Ship PTX SGEMM kernel as embedded string

**Phase 2 (ambitious):** Option #1 — direct ioctl
- Port tinygrad's `ops_nv.py` to Go
- True zero-dependency GPU compute
- ~2000 lines of ioctl structs + command buffer construction
