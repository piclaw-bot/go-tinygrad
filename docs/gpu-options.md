# GPU Compute Options

go-pherence supports three GPU backends, all via `purego` dlopen (no CGo):

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
- **ExpertPool**: LRU cache for MoE expert weights with auto-sized VRAM budget
- **BudgetManager**: 4-tier memory tracking (resident/layer/stream/expert)
- **MmapAdvisor**: page-level madvise tracking for weight streaming
- **Layer placement**: auto-fit or manual `--gpu-layers N` for hybrid forward

## Vulkan Compute (any GPU)

Portable backend for non-NVIDIA hardware:

- **Targets**: Intel iGPU (UHD/Iris/Arc), AMD RDNA, ARM Mali, Qualcomm Adreno, MoltenVK
- **API**: 35 Vulkan functions, device auto-selection, compute queue + command pool
- **Shaders**: 8 GLSL compute shaders (F32 + BF16 variants)
- **BF16**: emulated via uint16 bitshift (no extensions needed)
- **Status**: init + buffer + dispatch pipeline working; SPIR-V needs glslangValidator

## CPU SIMD Assembly

AVX2+FMA (amd64) and NEON (arm64):

- 13 F32 operations + 5 BF16 operations per architecture
- All hot paths covered: RMSNorm, VecAdd, Dot, SiLU, BF16 widen/narrow
- Scalar Go fallback for other architectures

## Backend Selection

```
if NVIDIA GPU available:
    → CUDA PTX (fastest, 21+ kernels)
elif Vulkan device available:
    → Vulkan SPIR-V (portable, 8 shaders)
else:
    → CPU SIMD (AVX2 or NEON assembly)
    → Go scalar (universal fallback)
```

The model forward pass auto-dispatches: each operation checks GPU availability and falls back to SIMD/scalar transparently.
