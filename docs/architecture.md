# Architecture

go-pherence is a multi-backend inference engine that runs MLX, GPTQ, and BF16 model weights on any hardware.

## Design Goals

1. **Run MLX weights everywhere** — Apple's MLX ecosystem has the best quantized models, but only runs on Apple Silicon. go-pherence makes them portable.
2. **Pure Go, zero CGo** — single static binary, GPU activates at runtime via `purego` dlopen.
3. **Tiered acceleration** — production CUDA PTX path, Vulkan SPIR-V scaffolding, SIMD assembly, and Go scalar fallback.
4. **Native BF16 scaffolding** — half-bandwidth helpers for BF16-trained models, with F32-compatible paths still used where required.

## Backend Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Model Forward Pass                    │
│  (llama, qwen2, qwen3, gemma3, gemma4)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ CUDA PTX │  │  Vulkan  │  │   SIMD   │  ┌────────┐ │
│  │ 21 kernels│ │ SPIR-V   │  │AVX2+NEON │  │Go scalar│ │
│  │ sm_80+   │  │ any GPU  │  │asm       │  │fallback │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘ │
│       │              │             │             │       │
│  ┌────┴────┐   ┌─────┴────┐  ┌────┴────┐   ┌───┴───┐  │
│  │NVIDIA   │   │Intel iGPU│  │x86_64   │   │any    │  │
│  │RTX/GTX  │   │AMD RDNA  │  │ARM64    │   │GOARCH │  │
│  │Jetson   │   │Mali/Adr. │  │         │   │       │  │
│  └─────────┘   └──────────┘  └─────────┘   └───────┘  │
└─────────────────────────────────────────────────────────┘
```

## Source Layout

Phase 6.5 is moving the repository toward explicit ownership boundaries:

| Area | Current package | Notes |
|---|---|---|
| CLI front-ends | `cmd/llmgen`, `cmd/llmchat`, `cmd/llmserver` | Flags and user/server I/O only |
| Loader helpers | `loader/config`, `loader/tokenizer`, `loader/safetensors`, `loader/weights` | Config JSON, tokenizer JSON, mmap safetensors, sharded/single-file weight sources; safetensors metadata and tokenizer helpers are guarded |
| Placement policy | `backends/placement` | Backend-neutral budget manager and layer placement estimator; device memory availability is caller-supplied |
| SIMD backend | `backends/simd` | Package name remains `simd`; import path is backend-owned; scalar fallbacks and SGEMM/GEBP wrappers are bounds-guarded |
| Vulkan backend | `backends/vulkan` | Vulkan loader/device/buffer/shader dispatch scaffolding and embedded SPIR-V assets |
| BERT/GTE | `models/bert` | Encoder path split out of the decoder package |
| KV runtime | `runtime/kv` | TurboQuant state, compressed KV cache, and staging/rollback helpers with layout and overflow guards |
| Memory runtime | `runtime/memory` | mmap residency advice/range tracking used by safetensors eager loading and future streaming; nil advisors are inert |
| Quant runtime | `runtime/quant` | MLX/GPTQ CPU quant formats, dtype/shape validation, dequantization, and guarded on-the-fly Q4 GEMV helpers |
| Decoder transition package | `model` | LLaMA-family loader/forward, Gemma/Qwen/MoE/MTP, model-specific KV sizing; still being split |
| GPU transition package | `gpu`, `backends/cuda/ptx` | CUDA runtime dispatch and GPU-resident expert cache remain in `gpu`; embedded PTX source assets now live in backend ownership under `backends/cuda/ptx` |
| Tensor graph | `tensor` | Lazy tensor DAG/runtime; transitional direct import of `backends/simd`; malformed-input validation across shapes, realization, rewrite/fusion, NN helpers, and modules |


## Shared Runtime Hardening Baseline

The Phase 6.5 audit now treats guard behavior in shared packages as part of the architecture, not incidental cleanup:

- `tensor` constructors and shape helpers reject negative or overflowing dimensions before allocation.
- Tensor entrypoints are nil-safe or explicitly panic with domain errors before dereferencing internal fields.
- Realization, broadcast, reduction, rewrite, and fusion paths validate malformed UOps, source lists, buffer lengths, and reduction metadata before indexing.
- Embedding, matmul, linear, softmax, layernorm, GELU, and module wrappers validate dimensions and optional parameters before slicing or dispatching SIMD kernels.
- `runtime/quant`, `runtime/kv`, `runtime/memory`, loader helpers, SIMD wrappers, and CUDA dispatch wrappers follow the same policy: validate dimensions/pointers/layouts at API boundaries, then either return an error/nil/no-op or panic with a local diagnostic rather than relying on incidental index panics.
- SIMD scalar fallbacks bound all participating slices; SGEMM/GEBP wrappers validate dimensions, pointers, strides, and overflow before unsafe slicing or pointer arithmetic.
- Safetensors validates known dtype byte sizes against declared shapes and data offsets at open time; tokenizer byte maps use one-time initialization for concurrent callers.

Later package moves should preserve this policy and keep focused regression tests close to the package that owns the guard.

## Weight Format Pipeline

```
HuggingFace (mlx-community, GPTQ, BF16)
    │
    ▼
loader/safetensors + loader/weights (GetFloat32, GetBF16, GetInt32, GetRaw)
    │
    ├─── MLX 4-bit: runtime/quant.LoadMLXWeight validates packed shape + F32/F16/BF16 scales/biases
    │    └─── GPU: transpose → GPTQ kernel + bias correction
    │
    ├─── GPTQ 4-bit: loader reads qweight/g_idx/scales/qzeros → runtime/quant validates before dequant or GemvQ4Sym
    │    └─── GPU: direct tiled GEMV
    │
    └─── BF16/F16/F32: load → tensor (optional BF16 native path)
         └─── GPU: DevBuf upload
```

## Model Architecture Support

| Feature | llama | qwen2 | qwen3 | gemma3 | gemma4 |
|---|---|---|---|---|---|
| RoPE | ✅ | ✅ | ✅ | ✅ dual | ✅ dual |
| GQA | ✅ | ✅ | ✅ | ✅ | ✅ |
| QK-Norm | — | — | ✅ | ✅ | ✅ |
| 4-norm residual | — | — | — | ✅ | ✅ |
| Sliding window | — | — | — | ✅ | ✅ |
| Embed scaling | — | — | — | ✅ ×√h | ✅ ×√h |
| Norm +1 offset | — | — | — | ✅ | ✅ |
| GELU activation | — | — | — | ✅ | ✅ |
| BOS token | — | — | — | ✅ | ✅ |
| Tensor prefix | — | — | — | — | ✅ language_model. |
| Q/K/V bias | — | ✅ | — | — | — |
| head_dim ≠ h/heads | — | — | ✅ | ✅ | ✅ |
| MTP drafter assets | — | — | research | — | scaffold |

## Speculative Decoding / MTP

Gemma4 MTP support is currently scaffolded but not wired into public generation paths. Implemented pieces:

- `LoadGemma4MTPDrafter` for `gemma4_assistant` safetensors assets with q-only attention blocks.
- Assistant projection helpers: token embedding row copy, masked ordering lookup, `PreProjectInto`, and `PostProjectInto`.
- Main-model verifier primitives: raw/scaled token embeddings, Gemma4 per-layer input preparation, LM-head logits, and greedy argmax.
- Acceptance helpers: `AcceptMTPDraft`, `AcceptMTPDraftFromLogits`, and LiteRT-style bonus-token accounting.
- `runtime/kv` staging helpers for candidate rollback/commit in both uncompressed and TurboQuant-backed caches.

Remaining architecture work is the batched verifier forward path and q-only drafter forward loop with external/main-model KV state.

## BF16 Pipeline

```
loader/safetensors BF16 → GetBF16() → []uint16 (zero conversion)
    │
    ├─── CPU: `backends/simd` package: BF16DotAsm (AVX2 445ns / NEON 8-wide)
    │         BF16RMSNormAsm (AVX2 1.4µs)
    │         BF16VecAddAsm (AVX2/NEON 8-wide)
    │
    ├─── GPU CUDA: ld.global.b16 + cvt.f32.bf16 (native Ampere+)
    │              ld.global.u16 + shl (emulated sm_80)
    │
    └─── GPU Vulkan: uint16 load + bitshift (universal)
```

## Kernel / Shader Inventory

| Backend | Current status |
|---|---|
| CUDA PTX | 27 hand-written kernels across GEMV/GEMM, attention/RoPE, norms, activations, BF16, and utility paths; source assets live in `backends/cuda/ptx` while dispatch/resource ownership remains in `gpu` |
| Vulkan SPIR-V | `backends/vulkan` owns shader assets for vector add, RMSNorm, GEMV, SiLU, attention score, RMSNormNoScale, RoPEPartial, and GELU paths; full forward dispatch is still pending |
| AVX2 asm | Runtime-gated vector, norm, dot/Saxpy, BF16, and SGEMM helpers with scalar fallback |
| NEON asm | Runtime-gated vector, norm, dot/Saxpy, BF16, and SGEMM helpers with scalar fallback; hardware verification still pending |
| Go scalar | Universal fallback for unsupported architectures or uncovered kernels |
