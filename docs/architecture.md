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
| Loader helpers | `loader/config`, `loader/tokenizer`, `loader/safetensors`, `loader/weights` | Config JSON, tokenizer JSON, mmap safetensors, sharded/single-file weight sources |
| Placement policy | `backends/placement` | Backend-neutral budget manager and layer placement estimator; device memory availability is caller-supplied |
| SIMD backend | `backends/simd` | Package name remains `simd`; import path is now backend-owned |
| Vulkan backend | `backends/vulkan` | Vulkan loader/device/buffer/shader dispatch scaffolding and embedded SPIR-V assets |
| BERT/GTE | `models/bert` | Encoder path split out of the decoder package |
| KV runtime | `runtime/kv` | TurboQuant state, compressed KV cache, and staging/rollback helpers |
| Memory runtime | `runtime/memory` | mmap residency advice/range tracking used by safetensors eager loading and future streaming |
| Quant runtime | `runtime/quant` | MLX/GPTQ CPU quant formats, dtype/shape validation, dequantization, and guarded on-the-fly Q4 GEMV helpers |
| Decoder transition package | `model` | LLaMA-family loader/forward, Gemma/Qwen/MoE/MTP, model-specific KV sizing; still being split |
| GPU transition package | `gpu`, `backends/cuda/ptx` | CUDA runtime dispatch and GPU-resident expert cache remain in `gpu`; pure PTX source assets have started moving to backend ownership in `backends/cuda/ptx` |
| Tensor graph | `tensor` | Lazy tensor DAG/runtime; transitional direct import of `backends/simd` |

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
| CUDA PTX | 27 hand-written kernels across GEMV/GEMM, attention/RoPE, norms, activations, BF16, and utility paths; pure PTX source assets are being separated under `backends/cuda/ptx` while dispatch remains in `gpu` |
| Vulkan SPIR-V | `backends/vulkan` owns shader assets for vector add, RMSNorm, GEMV, SiLU, attention score, RMSNormNoScale, RoPEPartial, and GELU paths; full forward dispatch is still pending |
| AVX2 asm | Runtime-gated vector, norm, dot/Saxpy, BF16, and SGEMM helpers with scalar fallback |
| NEON asm | Runtime-gated vector, norm, dot/Saxpy, BF16, and SGEMM helpers with scalar fallback; hardware verification still pending |
| Go scalar | Universal fallback for unsupported architectures or uncovered kernels |
