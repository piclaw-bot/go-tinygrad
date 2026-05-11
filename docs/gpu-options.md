# GPU Compute Options

go-pherence currently has a production CUDA backend plus Vulkan backend scaffolding. CUDA/Vulkan use `purego` dlopen (no CGo):

## CUDA PTX (NVIDIA)

Primary GPU backend. 27 hand-written PTX kernels:

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

- **DevBuf**: device-agnostic buffers with lazy CPU↔GPU transfer
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
