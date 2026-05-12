# GPU Compute Options

go-pherence currently has a production CUDA backend plus Vulkan backend scaffolding. CUDA/Vulkan use `purego` dlopen (no CGo):

## CUDA PTX (NVIDIA)

Primary GPU backend. 27 hand-written PTX kernels. Source strings are owned by `backends/cuda/ptx`; runtime loading, launch helpers, `DevBuf`, and GPU-resident resources remain in the transitional `gpu` package:

| Category | Kernels | Notes |
|---|---|---|
| **Core GEMV** | sgemm_nn, gemv_q4sym, gemm_q4sym | GPTQ tiled + shared mem |
| **MLX** | mlx_gemv, mlx_gemm, mlx_correct | Transposed layout + bias |
| **Element-wise** | vec_add, vec_mul, vec_scale, vec_silu | threshold-free |
| **Fused** | fused_silu_mul, rms_norm, gelu_tanh_mul | reduced launch count |
| **Norms** | rms_norm_no_scale, to_bf16_f32 | Gemma4 V-norm, BF16 trunc |
| **Attention** | rope_apply, rope_partial, gqa_attention | precomputed cos/sin, scale param |
| **Utility** | lm_head_gemv, prefetch_l2, vec_scale | 2D grid, L2 warming |
| **BF16** | bf16_rms_norm, bf16_vec_add, bf16_silu_mul, bf16_gelu_tanh_mul | emulated (sm_80) |
| **BF16 native** | native_bf16_rms_norm/vec_add/gemv | ld.b16+cvt (sm_86+) |

Loaded as one mega module + optional native BF16 module.

### Memory Management

- **DevBuf**: device-agnostic buffers with lazy CPU↔GPU transfer; vector/norm/dense GEMV/LM-head fast paths preflight kernel operands before launching and fall back or no-op safely if upload/allocation fails
- **Quantized dispatch**: Q4/MLX upload paths validate dimensions, packed-weight sizes, scale layouts, and group indices before allocating GPU buffers
- **ExpertPool**: LRU cache for MoE expert weights with auto-sized VRAM budget; disabled and replacement cases return GPU resources for explicit release
- **BudgetManager**: 4-tier memory tracking (resident/layer/stream/expert), now owned by `backends/placement`
- **MmapAdvisor**: `runtime/memory` page-level madvise tracking for eager loading and future weight streaming
- **Layer placement**: `backends/placement` auto-fit/manual policy (`--gpu-layers N`) with caller-supplied device memory availability

## Vulkan Compute (any GPU)

Portable backend for non-NVIDIA hardware. Vulkan code and shaders now live under `backends/vulkan`:

- **Targets**: Intel iGPU (UHD/Iris/Arc), AMD RDNA, ARM Mali, Qualcomm Adreno, MoltenVK
- **API**: 35 Vulkan functions, device auto-selection, compute queue + command pool
- **Shaders**: GLSL/SPIR-V coverage for vector add, RMSNorm, GEMV, SiLU, attention score, RMSNormNoScale, RoPEPartial, and GELU paths
- **BF16**: emulated via uint16 bitshift (no extensions needed)
- **Status**: init + buffer path and embedded SPIR-V are present; Vulkan op dispatch wiring is still pending in Phase 3.6

## CPU SIMD Assembly

AVX2+FMA (amd64) and NEON (arm64):

- Runtime-gated AVX2/FMA and NEON wrappers with scalar fallback
- Covered hot paths include vector add/mul/scale, dot/Saxpy, RMSNorm variants, BF16 widen/narrow, and SGEMM wrappers
- Remaining CPU SIMD gaps include fused GELU, RoPEPartial, and MLX/GPTQ Q4 GEMV kernels

## Backend Selection

```
if NVIDIA GPU available:
    → CUDA PTX (fastest, 27 kernels)
elif Vulkan model dispatch is enabled and a non-software Vulkan device is available:
    → backends/vulkan SPIR-V (portable shader path; still being wired)
else:
    → CPU SIMD (AVX2 or NEON assembly)
    → Go scalar (universal fallback)
```

The current production model path chooses CUDA when requested/available, otherwise CPU SIMD/scalar. Vulkan device/shader scaffolding is present but full forward dispatch remains a Phase 3.6 item.


## DevBuf/dispatch guard status

During the Phase 6.5 refactor audit, the transitional `gpu` package has been hardened before the eventual CUDA runtime split:

- `DevBuf` receiver helpers are nil-safe and `ToGPU`/`GPUPtr` propagate upload failures instead of marking stale GPU state authoritative.
- CUDA allocation rejects host-side byte-size overflow before driver calls.
- Stream/graph helpers validate nil graph executables, nil kernel arguments, and invalid launch dimensions before CUDA calls.
- Q4/MLX quantized weight upload/dispatch validates packed-weight and scale product arithmetic, buffer byte sizes, group consistency/indices, batched dimensions, and download errors in CPU fallback.
- Expert-pool helpers reject nil pools and invalid expert IDs without leaking caller-owned GPU resources.
- Experimental direct-NVIDIA ioctl/memory/query/GPFIFO helpers validate nil receivers, size arithmetic, fd/argument state, class-list sizes, and release partially allocated resources on setup failure.
- Dense SGEMM/LM-head dispatch validates dimensions, buffer byte sizes, and product overflow before kernel launch.
- CUDA JIT helpers validate kernel specs and launch buffers before PTX generation or dispatch.
- BF16 CUDA wrappers validate nil/undersized buffers and length overflow before emulated/native dispatch.

These guards are part of the current backend baseline and should move with the CUDA runtime when `gpu` is split into `backends/cuda`.
