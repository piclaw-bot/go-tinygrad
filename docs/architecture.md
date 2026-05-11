# Architecture

go-pherence is a multi-backend inference engine that runs MLX, GPTQ, and BF16 model weights on any hardware.

## Design Goals

1. **Run MLX weights everywhere** — Apple's MLX ecosystem has the best quantized models, but only runs on Apple Silicon. go-pherence makes them portable.
2. **Pure Go, zero CGo** — single static binary, GPU activates at runtime via `purego` dlopen.
3. **Three-tier acceleration** — CUDA PTX → Vulkan SPIR-V → SIMD assembly → Go scalar.
4. **Native BF16** — half-bandwidth pipeline for models trained in BF16.

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

## Weight Format Pipeline

```
HuggingFace (mlx-community, GPTQ, BF16)
    │
    ▼
safetensors loader (GetFloat32, GetBF16, GetInt32, GetRaw)
    │
    ├─── MLX 4-bit: loadMLXWeight → [outDim, inDim/8] uint32 + scales + biases
    │    └─── GPU: transpose → GPTQ kernel + bias correction
    │
    ├─── GPTQ 4-bit: loadQW → [inDim/8, outDim] int32 + g_idx + scales
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
- KV staging helpers for candidate rollback/commit in both uncompressed and TurboQuant-backed caches.

Remaining architecture work is the batched verifier forward path and q-only drafter forward loop with external/main-model KV state.

## BF16 Pipeline

```
safetensors BF16 → GetBF16() → []uint16 (zero conversion)
    │
    ├─── CPU: simd.BF16DotAsm (AVX2 445ns / NEON 8-wide)
    │         simd.BF16RMSNormAsm (AVX2 1.4µs)
    │         simd.BF16VecAddAsm (AVX2/NEON 8-wide)
    │
    ├─── GPU CUDA: ld.global.b16 + cvt.f32.bf16 (native Ampere+)
    │              ld.global.u16 + shl (emulated sm_80)
    │
    └─── GPU Vulkan: uint16 load + bitshift (universal)
```

## Kernel Inventory

| Backend | F32 | BF16 emulated | BF16 native | Total |
|---|---|---|---|---|
| CUDA PTX | 16 | 2 | 3 | **21** |
| Vulkan SPIR-V | 4 | 4 | — | **8** |
| AVX2 asm | 8 | 5 | — | **13** |
| NEON asm | 8 | 5 | — | **13** |
| Go scalar | all | all | — | fallback |
