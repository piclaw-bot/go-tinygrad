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

Phase 6.5 has moved the repository toward explicit ownership boundaries. Remaining large splits are documented follow-ups rather than Phase 6.5 blockers:

| Area | Current package | Notes |
|---|---|---|
| CLI front-ends | `cmd/llmgen`, `cmd/llmchat`, `cmd/llmserver`, `cmd/specbench` | Flags and user/server I/O only; `specbench` emits normal-vs-speculative CSV benchmark rows |
| Loader helpers | `loader/config`, `loader/tokenizer`, `loader/safetensors`, `loader/weights` | Config JSON, tokenizer JSON, mmap safetensors, sharded/single-file weight sources; safetensors metadata, nil helpers, deterministic names, checked eager totals, partial sharded-open cleanup, and tokenizer merge helpers are guarded |
| Placement policy | `backends/placement` | Backend-neutral budget manager and layer placement estimator; device memory availability is caller-supplied; accounting rejects invalid categories and estimator math is saturating |
| SIMD backend | `backends/simd` | Package name remains `simd`; import path is backend-owned; `backends/simd` is the facade for future CPU-family subpackages; scalar fallbacks, BF16 GEMV, empty-slice dispatch, per-call GEBP scratch, and SGEMM/GEBP/gather byte offsets are bounds/overflow-guarded |
| Vulkan backend | `backends/vulkan` | Vulkan loader/device/buffer/shader dispatch scaffolding and embedded SPIR-V assets; diagnostics are opt-in via `GO_PHERENCE_VULKAN_DEBUG` |
| BERT/GTE | `models/bert` | Encoder path split out of the decoder package |
| KV runtime | `runtime/kv` | TurboQuant state, compressed KV cache, and staging/rollback helpers with layout, accessor, memory-accounting, protected-layer input, and overflow guards |
| Memory runtime | `runtime/memory` | mmap residency advice/range tracking used by safetensors eager loading and future streaming; nil advisors are inert and malformed tracked ranges are sanitized with saturating accounting |
| Quant runtime | `runtime/quant` | MLX/GPTQ CPU quant formats, dtype/shape validation, checked expected-size/dequant output arithmetic, dequantization, and guarded on-the-fly Q4 GEMV helpers |
| Decoder transition package | `model` | LLaMA-family loader/forward, Gemma/Qwen/MoE/MTP, stock-weight speculative scaffold, model-specific KV sizing; package split deferred to Phase 6.8 and generation extraction to Phase 6.9; helper guards cover MTP drafter/verifier/acceptance/KV commit edges, speculative proposer/config/stats/checkpoint paths, CPU decode finish/final norm, generation allocation setup, MoE, inference helpers, CPU forward-layer entrypoints, KV sizing, GPU prefill/LM-head, GEMV, GQA arithmetic, and opt-in loader/prefill logging |
| GPU transition package | `gpu`, `backends/cuda/ptx` | CUDA runtime dispatch and GPU-resident expert cache remain in `gpu` until the Phase 6.7 CUDA runtime split; embedded PTX source assets live under `backends/cuda/ptx`; DevBuf/stream/Q4/MLX/expert/NV/dense/JIT/BF16 validation and opt-in GPU diagnostics are hardened before that split |
| Tensor graph | `tensor` | Lazy tensor DAG/runtime; transitional direct import of `backends/simd`; malformed-input validation across shapes, unsafe views, broadcasting, realization, rewrite/fusion, NN/convenience helpers, matmul/linear, and modules |


## Shared Runtime Hardening Baseline

The Phase 6.5 audit now treats guard behavior in shared packages as part of the architecture, not incidental cleanup:

- `tensor` constructors and shape helpers reject negative or overflowing dimensions before allocation; malformed shape sizing/contiguity/broadcast checks fail before indexing or allocating.
- Tensor entrypoints are nil-safe or explicitly panic with domain errors before dereferencing internal fields.
- Realization, broadcast, reduction, rewrite, and fusion paths validate malformed UOps, source lists, buffer lengths, and reduction metadata before indexing.
- Unsafe float32 views, embedding, matmul, linear, softmax, layernorm, GELU, convenience ops, and module wrappers validate dimensions, backing-data lengths, and optional parameters before slicing or dispatching SIMD kernels.
- `runtime/quant`, `runtime/kv`, `runtime/memory`, loader helpers, SIMD wrappers, and CUDA dispatch wrappers follow the same policy: validate dimensions/pointers/layouts and checked arithmetic at API boundaries, then either return an error/nil/no-op or panic with a local diagnostic rather than relying on incidental index panics.
- SIMD scalar fallbacks bound all participating slices; scalar RMSNorm uses precise `math.Sqrt`; BF16 GEMV validates shape-product overflow; empty vector/BF16 calls route through scalar fallbacks; GEBP packing scratch is per-call; SGEMM/GEBP/gather wrappers validate dimensions, pointers, strides, CPU capability gates, checked float32 byte offsets, and overflow before unsafe slicing or pointer arithmetic.
- Safetensors validates known dtype byte sizes against declared shapes and data offsets at open time; nil file/sharded helpers are safe, tensor names are sorted deterministically, sharded eager-load totals are checked, partial sharded opens clean up already-open shards, tokenizer byte maps use one-time initialization for concurrent callers, and malformed BPE merges are rejected.
- Transitional model helpers validate staged MTP acceptance consistency, model-aware verifier token/position/logit/activation dimensions, shared-KV verifier sources, alias-safe MTP drafter projection products, q-only external-KV/layer dimensions, bounded multi-draft counts, speculative stats/rollback paths, CPU decode final norm/LM-head dimensions, generation allocation setup, MoE loader/forward edge cases, model-specific KV dimensions, embedding/LM-head/per-layer-input backing slices, CPU forward-layer entrypoints, GPU prefill/chunked-LM-head entrypoints, and low-level GEMV/GQA product arithmetic before slicing or dispatch.
- Transitional CUDA helpers validate `DevBuf` receiver/upload state, graph launches, stream kernel arguments, allocation sizes, Q4/MLX packed-weight layouts, expert-pool IDs, experimental NV helper inputs, dense SGEMM/LM-head buffers, JIT specs, BF16 buffers, and RoPE/attention tensor shapes before driver calls or kernel dispatch; CUDA/NV progress logs are opt-in under `GO_PHERENCE_GPU_DEBUG`.

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
| MTP drafter assets | — | — | research | — | internal |

## Speculative Decoding / MTP

There are two distinct speculative tracks:

1. **Gemma4/Qwen MTP internals** — custom drafter/checkpoint assets, still disabled in public generation.
2. **Stock-weight speculative scaffold** — Orthrus-inspired verifier mechanics without custom weights, opt-in on the CPU backend via `--speculative`.

### MTP internals

Gemma4 MTP support now has internal verifier/drafter integration pieces, but it remains deliberately disabled in public generation/CLI paths. Implemented pieces:

- `LoadGemma4MTPDrafter` for `gemma4_assistant` safetensors assets with q-only attention blocks.
- Assistant projection helpers: token embedding row copy, masked ordering lookup, `PreProjectInto`, and `PostProjectInto`.
- Main-model verifier primitives: raw/scaled token embeddings, Gemma4 per-layer input preparation, CPU decode finish helper with copied final activation, LM-head logits, and greedy argmax.
- Initial CPU verifier loop: `RunMTPVerifierForward` validates plan/KV contracts, rejects unsupported Gemma4 PLI/batched semantics, runs real CPU layers through `ForwardLayer`, stages float KV, and returns per-position logits plus final activation.
- Acceptance helpers: `AcceptMTPDraft`, `AcceptMTPDraftFromLogits`, LiteRT-style bonus-token accounting, and `MTPAcceptance.Validate` before staged KV commits.
- `runtime/kv` staging helpers for candidate rollback/commit in both uncompressed and TurboQuant-backed caches; model-aware verifier plans/results validate vocab/token/position/logit/activation dimensions before deriving acceptance.
- Internal drafter/verifier seams: projection-only, synthetic q-only, and local real-asset contract tests can run against an explicit external-KV view; bounded multi-step drafter and multi-draft speculative helpers record LiteRT-style stats and restore staged verifier KV on verifier/stat errors.

Remaining MTP architecture work is full Gemma4 PLI/batched verifier semantics, production q-only drafter parity against real assistant assets, adaptive draft-count policy, GPU/hybrid support, and public generation wiring after CPU/GPU smokes.

### Stock-weight speculative scaffold

The stock-weight path deliberately avoids Orthrus custom `*_diff` tensors and instead provides a reusable verification/proposer scaffold for normal Qwen/Gemma/LLaMA weights:

- `GenerateSpeculative` / `GenerateSpeculativeWithStats` are opt-in CPU entrypoints used by `llmgen`, `llmchat`, and `llmserver` when `--speculative` is set.
- `SpeculativeProposer` is pluggable. Current proposers are `prompt` (prompt/suffix lookup), `repeat-last` (cheap verifier stress), and `none` (fallback overhead baseline).
- `CPUDecodeState` owns output/KV checkpoint, restore, `GenerateGreedy`/`DecodeOneGreedy`, accepted-prefix commit, and `VerifyGreedyBlock` contracts.
- Current verifier backend is `replay`: exact greedy verification by replaying the prepared CPU prompt. It is a correctness/measurement scaffold and can be slower.
- The `kv` backend selector is accepted but falls back to `replay` until a stateful KV-reusing verifier replaces the replay body.
- `SpeculativeStats` records backend, proposer, proposal/acceptance/fallback counters, emitted-token counts, tokens/step, average proposal length, plus reusable add/average helpers for benchmarks.
- `cmd/specbench` compares normal vs speculative output, validates parity, supports prompt-file workloads and repeat averaging, and emits CSV suitable for tracking `backend=replay` to future `backend=kv` improvements.

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
| CUDA PTX | 29 hand-written kernels across GEMV/GEMM, attention/RoPE, norms, activations, BF16, NVFP4 dequant fallback, fused add-scaled accumulation, and utility paths; source assets live in `backends/cuda/ptx` while dispatch/resource ownership remains in `gpu` |
| Vulkan SPIR-V | `backends/vulkan` owns shader assets for vector add, RMSNorm, GEMV, SiLU, attention score, RMSNormNoScale, RoPEPartial, and GELU paths; full forward dispatch is still pending |
| AVX2 asm | Runtime-gated vector, norm, dot/Saxpy, BF16, and SGEMM helpers with scalar fallback |
| NEON asm | Runtime-gated vector, norm, dot/Saxpy, BF16, and SGEMM helpers with scalar fallback; hardware verification still pending |
| Go scalar | Universal fallback for unsupported architectures or uncovered kernels |
