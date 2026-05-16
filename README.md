# go-pherence

![go-pherence](docs/icon-256.png)

**Run MLX models on any hardware.** A pure Go inference engine for Apple MLX, GPTQ and BF16 model weights — with a production CUDA backend, Vulkan scaffolding, and SIMD assembly CPU paths. Single static binary, no Python, no CGo, no external dependencies.

> Active development goal: Qwen3.6 27B native MTP support. See [docs/qwen36-mtp.md](docs/qwen36-mtp.md) for checkpoint findings, blockers, and the implementation roadmap.

## Why

Apple's [MLX](https://github.com/ml-explore/mlx) ecosystem has the best quantized model collection on HuggingFace (mlx-community). But MLX only runs on Apple Silicon. **go-pherence runs those same MLX weights on NVIDIA GPUs, Intel/AMD CPUs, ARM SBCs, and (soon) any Vulkan device.**

## Performance

| Model | Arch | Format | GPU tok/s | CPU tok/s |
|---|---|---|---|---|
| **Qwen2.5-7B** | qwen2 | MLX 4-bit | **~120–158** | 1.1 |
| **SmolLM2-135M** | llama | BF16 | **86** | 35.5 |
| **Gemma3-1B** | gemma3 | MLX 4-bit | **~72** | 4.9 |
| **Qwen2.5-7B** | qwen2 | GPTQ 4-bit | **51** | 0.9 |
| **Qwen2.5-0.5B** | qwen2 | MLX 4-bit | **31** | 7.2 |
| **Qwen3-0.6B** | qwen3 | MLX 4-bit | **25** | 7.2 |
| **Gemma4-E2B** | gemma4 | MLX 4-bit | **~21–22** | — |
| **Qwen3-30B MoE** | qwen3_moe | MLX 4-bit | **~5.2 cold / ~5.5 warm** | 0.6 |

*RTX 3060 12GB + i7-12700 6-core. Pure Go, zero CGo. Short-run decode rates vary with prompt length, route-set warmth, and VRAM headroom.*
*MoE: 128 experts/layer, 8 active/token. CUDA runs attention, router, and selected experts via a GPU-resident expert cache; cold route sets pay one-time expert upload cost.*

## Supported Models

| Architecture | Models | Formats | Status |
|---|---|---|---|
| **llama** | SmolLM2, LLaMA 3.x | BF16, F16, F32 | ✅ |
| **qwen2** | Qwen2.5 0.5B–7B | MLX 4-bit, GPTQ 4-bit | ✅ |
| **qwen3** | Qwen3 0.6B+ | MLX 4-bit, BF16 | ✅ |
| **qwen3_moe** | Qwen3-30B-A3B MoE | MLX 4-bit | ✅ |
| **gemma3** | Gemma 3 1B+ | MLX 4-bit, BF16 | ✅ |
| **gemma4** | Gemma 4 E2B+ | MLX 4-bit | ✅ |

Any model from [mlx-community](https://huggingface.co/mlx-community) using these architectures works out of the box.

## Quick Start

```bash
# Download any MLX model from HuggingFace
mkdir -p models/qwen3-0.6b
for f in config.json model.safetensors tokenizer.json; do
  curl -L "https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/resolve/main/$f" \
    -o "models/qwen3-0.6b/$f"
done

# Run on CPU (AVX2/NEON SIMD)
go run ./cmd/llmgen -model models/qwen3-0.6b -tokens 50 -prompt "The meaning of life is"

# Run on GPU (auto-detects NVIDIA at runtime, zero CGo)
go run ./cmd/llmgen -gpu -model models/qwen3-0.6b -tokens 50 -prompt "The meaning of life is"
```

## Backend Stack

### GPU: CUDA PTX (NVIDIA)

29 hand-written PTX kernels compiled by the driver at runtime via `purego` dlopen. Runtime dispatch/resource ownership remains in `gpu`; embedded PTX source assets live in `backends/cuda/ptx`:

- **Quantized GEMV**: INT4 dequant+multiply with shared memory tiling (GPTQ + MLX)
- **Batched GEMM**: multi-token prefill, reads weights once for all tokens
- **LM Head**: dedicated large-vocab GEMV with 2D grid
- **RMSNorm, RoPE, GQA Attention, SiLU, VecAdd/Mul/Scale/AddScaled**
- **BF16 kernels**: native `cvt.f32.bf16` on Ampere+ (sm_86), emulated on sm_80
- **MLX bias correction**: post-GEMV fixup for MLX affine quantization
- **NVFP4 fallback**: packed FP4/F8-scale upload, dequant-to-F32 kernel, and correctness-first dense GEMV fallback; public loading/generation remains disabled for NVFP4 checkpoints until real logits/tokens agree

### GPU: Vulkan Compute (any GPU)

Portable compute backend scaffolding for Intel iGPU, AMD, ARM Mali, Adreno:

- `backends/vulkan` owns the Vulkan loader, device/buffer helpers, dispatch scaffolding, and embedded SPIR-V assets
- 35 Vulkan API functions via `purego` (no SDK required)
- GLSL/SPIR-V shader coverage for vector add, RMSNorm, GEMV, SiLU, attention score, RMSNormNoScale, RoPEPartial, and GELU paths
- Device auto-selection rejects software/CPU devices by default; set `GO_PHERENCE_VULKAN_ALLOW_CPU=1` for debugging
- Current status: init/buffer/shader assets are in place; full model dispatch wiring is still pending

### CPU: SIMD Assembly

AVX2+FMA (amd64) and NEON (arm64) assembly for the core CPU hot paths:

| Operation | AVX2 | NEON | Notes |
|---|---|---|---|
| **Sdot** | 16-wide FMA | 8-wide VFMLA | dot product |
| **RMSNorm** | fused sum-sq + scale | 4-wide | 677ns / 3584 elements |
| **VecAdd** | 16-wide VADDPS | 8-wide FADD | 217ns / 3584 elements |
| **ToBF16** | 16-wide VANDPS | 8-wide AND | 179ns / 3584 elements |
| **BF16 Dot** | widen+FMA+narrow | USHLL+VFMLA | 445ns / 3584 elements |
| **BF16 RMSNorm** | fused BF16 | USHLL+XTN | 1.4µs / 3584 elements |
| **BF16 Widen** | VPMOVZXWD+VPSLLD | USHLL+SHL | 292ns / 3584 elements |
| **SiLU×Mul** | Go (exp not SIMD) | Go fallback | |

Public SIMD entrypoints are runtime-gated with scalar fallback. Recent Phase 6.6 cleanup keeps `backends/simd` as the public facade, splits scalar dot/SAXPY fallbacks into `scalar.go`, uses precise scalar sqrt/dimension guards, avoids assembly dispatch for empty vector/BF16 slices, keeps GEBP packing scratch per call, and checks SGEMM/GEBP/gather byte offsets before unsafe pointer arithmetic. Remaining CPU gaps include fused GELU, RoPEPartial, and MLX/GPTQ Q4 GEMV kernels.

### Native BF16

End-to-end BF16 pipeline for models trained in BF16 (Gemma3/4):

- **Safetensors**: `GetBF16()` returns `[]uint16` without F32 conversion
- **SIMD**: AVX2 and NEON assembly for BF16 dot/norm/add/widen/narrow
- **CUDA**: native `ld.global.b16` / `cvt.f32.bf16` on Ampere+
- **Vulkan**: BF16 emulated via uint16 bitshift (universal)
- **Model**: BF16 helper/scaffolding code for native hidden-state paths; public generation still primarily uses the F32-compatible path where required

## Weight Format Support

| Format | Detection | Dequant | GPU | Notes |
|---|---|---|---|---|
| **MLX affine 4-bit** | `config.json` quantization block | `val × scale + bias` | Transpose → GPTQ kernel | Primary format; runtime validates packed shape and F32/F16/BF16 scale/bias dtypes |
| **GPTQ INT4** | `quantize_config.json` | `(val - 8) × scale` | Native tiled GEMV | Symmetric; runtime validates qweight/g_idx/scales/qzeros and Q4 GEMV inputs |
| **BF16** | safetensors dtype | Direct load | F32 on GPU | Half bandwidth |
| **F16** | safetensors dtype | F16→F32 at load | F32 on GPU | |
| **F32** | safetensors dtype | Direct load | Native | |
| **NVFP4 / FP4** | `quantization_config` ModelOpt/compressed-tensors metadata, including mixed `config_groups` and group/weight format fields | FP4 E2M1 + F8_E4M3FN scale reference path | Upload + dequant-to-F32 fallback kernel | Experimental/internal only; synthetic CPU/CUDA dequant agrees, but public loading rejects NVFP4 until real checkpoint logits/tokens agree |

## Commands

### llmgen — one-shot text generation

```bash
go run ./cmd/llmgen -model models/qwen3-0.6b-mlx4 -gpu -tokens 50 -prompt "The meaning of life is"
```

CPU generation has an opt-in stock-weight speculative path for experimentation:

```bash
go run ./cmd/llmgen -model models/smollm2-135m -tokens 32 \
  -prompt "abc abc abc abc" \
  -speculative -speculative-proposer prompt -speculative-debug
```

Current speculative backend is `replay`, a correctness scaffold that reuses the CPU generator and can be slower. It is useful for measuring proposer acceptance before the planned KV-reusing verifier backend lands. Available proposer choices are `prompt`, `repeat-last`, and `none`; `-speculative-min-proposal` gates tiny proposals.

### qwenmtpmeta / qwenmtpsynth — Qwen3.6 native MTP triage

```bash
go run ./cmd/qwenmtpmeta -model /path/to/qwen3.6-27b-mtp

go run ./cmd/qwenmtpsynth -steps 2
```

`qwenmtpmeta` inspects Qwen3.5/Qwen3.6 native-MTP config/tensor metadata without entering the full model loader. `qwenmtpsynth` runs a tiny deterministic native-MTP synthetic correctness path while real Qwen3.6 loading remains gated.

### specbench / speccheck — speculative benchmark and correctness harness

```bash
go run ./cmd/specbench -model models/smollm2-135m \
  -prompt-file prompts.txt -tokens 16 -repeat 3 \
  -speculative-proposer prompt -csv specbench.csv

go run ./cmd/speccheck -model models/smollm2-135m \
  -prompt-file prompts.txt -tokens 16 \
  -proposers prompt,repeat-last,none
```

`specbench` emits normal/speculative rows with output parity, speedup vs normal, verifier backend, proposer, acceptance/fallback counters, emitted tokens, tokens/step, average proposal length, and aggregate total rows for multi-prompt workloads. `speccheck` emits JSON with total/failed speculative and golden check counts (including golden metadata checks), and exits non-zero on any mismatch; use `-write-golden` / `-golden` to save and compare normal-output token baselines.

### llmchat — interactive chat

```bash
go run ./cmd/llmchat -model models/gemma4-e2b-mlx4 -gpu -n 256
> Hello
Hello! How can I help you today?
[8 tok, 13.5 tok/s, 592ms]
```

### llmserver — OpenAI-compatible API server

```bash
go run ./cmd/llmserver -model models/gemma4-e2b-mlx4 -gpu -listen :8080
# Test with curl:
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-e2b-mlx4","messages":[{"role":"user","content":"Hello"}]}'
```

All generation/chat/server commands support `--gpu-layers N` for hybrid CPU/GPU inference (0=all on GPU).
Use `--eager-load` to pre-fault mmap'd safetensors weights at startup for more
predictable first-token latency. CPU generation also supports `--turbo-quant` to
enable TurboQuant KV-cache compression (4-bit keys, 2-bit values, protected
first/last layers, 128-token residual window). TurboQuant is currently CPU-backend
only; GPU KV compression is a future step. `llmgen`, `llmchat`, and `llmserver`
also support `--speculative` for the stock-weight speculative scaffold on the CPU
backend; GPU speculative verification is not enabled yet.

## Architecture Details

Current package ownership is now organized around explicit loader/runtime/backend boundaries, with the remaining large CUDA/model/generation splits explicitly deferred to follow-up phases:

- **`loader/`** — `config`, `tokenizer`, `safetensors`, and shared `weights` source opening; tokenizer merge validation and deterministic safetensors name ordering are hardened
- **`backends/placement/`** — backend-neutral memory budget and layer placement policy with guarded budget accounting and saturating estimator math
- **`backends/simd/`** — AVX2/FMA and NEON dispatch/kernels with guarded scalar fallbacks and SGEMM/GEBP preflights
- **`backends/cuda/ptx/`** — pure CUDA PTX source assets used by the transitional `gpu` mega-module loader
- **`backends/vulkan/`** — Vulkan loader/device/buffer/shader dispatch scaffolding and embedded SPIR-V assets; diagnostics are opt-in via `GO_PHERENCE_VULKAN_DEBUG`
- **`models/bert/`** — GTE/BERT encoder path
- **`runtime/kv/`** — TurboQuant state, compressed KV cache, and KV staging/rollback primitives with layout/overflow, accessor, and memory-accounting guards
- **`runtime/memory/`** — mmap residency advice and range tracking for eager/streamed weights; nil/invalid/malformed ranges are inert or sanitized with saturating accounting
- **`runtime/quant/`** — MLX/GPTQ CPU quant formats plus experimental NVFP4 layout/decode helpers, dtype/shape validation, checked expected-size arithmetic, dequantization, and guarded on-the-fly Q4/GEMV helpers
- **`model/`** — transitional LLaMA-family decoder package; Gemma/Qwen/MoE/MTP package splits are deferred to Phase 6.8 and generation extraction to Phase 6.9; MTP, MoE, inference/forward, KV, prefill, LM-head, logging gates, and low-level helper guards are hardened
- **`gpu/`** — transitional CUDA package plus GPU-resident expert cache pending the Phase 6.7 CUDA backend split; DevBuf, stream/graph, Q4/MLX/NVFP4 fallback dispatch, expert-pool, NV ioctl/memory/query/GPFIFO, dense SGEMM/LM-head, JIT, BF16, RoPE, softmax, attention dispatch guards, and opt-in `GO_PHERENCE_GPU_DEBUG` diagnostics are hardened

- **Lazy tensor DAG** with elementwise fusion, graph rewrites, and explicit malformed-input validation
- **Pattern matcher + graph rewrite** (tinygrad-style, 16 rules), nil-safe for malformed rule graphs
- **Safetensors loader** — `loader/safetensors`, mmap'd, sharded, F16/BF16/F32, GPTQ/MLX quantized, with bounded offset/shape/dtype-byte checks
- **Tokenizer** — `loader/tokenizer`, BPE with auto-detect SentencePiece `▁` vs GPT-2 `Ġ` prefix and race-safe byte-level maps
- **LLaMA decoder** — RoPE (global + local + partial), GQA, KV cache, SiLU/GELU MLP
- **Mixture of Experts** — router top-k, parallel expert MLP, ExpertPool with LRU GPU caching
- **QK-Norm** — per-head RMSNorm (Qwen3, Gemma3/4)
- **4-norm residual** — pre/post FFN norms (Gemma3/4)
- **Sliding window attention** — alternating local/global with window masking (Gemma3/4)
- **Per-layer input gating** — PLI with GELU gate, projection, norm (Gemma4)
- **KV sharing** — shared layers reuse source-layer KV cache (Gemma4)
- **MTP speculative decoding internals** — Gemma4 assistant drafter loader, alias-safe projection helpers, verifier plan/result validation, initial CPU verifier-forward loop, projection-only/synthetic/real-asset q-only drafter contract tests, bounded multi-step drafter loop, multi-draft drafter→verifier seam, LiteRT-style stats, and staged KV commit/rollback; public MTP generation remains disabled until full Gemma4 PLI/batched verifier and CPU/GPU smokes are complete
- **Stock-weight speculative scaffold** — opt-in CPU path inspired by Orthrus-style verification but without custom weights; pluggable prompt/repeat/no-op proposers, replay verifier backend, decode-state checkpoint/commit contracts, structured stats, and `cmd/specbench` benchmarking. Current `backend=replay` is exact but can be slower; speedup depends on a future KV-reusing verifier backend and proposer quality.
- **TurboQuant KV compression** — `runtime/kv`, optional CPU KV cache compression via `--turbo-quant`
- **Layer scalar** — per-layer output scaling (Gemma4)
- **Embedding scaling** — `× √hidden_size` (Gemma3/4)
- **Batched prefill** — GEMM for multi-token prompt processing
- **Hybrid forward** — GPU layers + CPU layers with `--gpu-layers N`
- **Weight budget** — tiered memory: GPU VRAM, pinned CPU, mmap with madvise
- **Eager mmap loading** — optional `--eager-load` startup pre-faulting for stable latency
- **LM-head placement policy** — F32 when moderate and resident, compact MLX when very large or VRAM-constrained
- **GPU DevBuf** — device-agnostic buffers, lazy CPU↔GPU transfer
- **Chat templates** — Gemma4 (`<|turn>`), Qwen3 (`<|im_start|>`)


### Validation / Hardening Status

Recent Phase 6.5 audit passes made malformed-input behavior explicit across the shared runtime layers:

- `tensor/` validates shapes, reductions, broadcasting, unsafe float32 views, realization internals, rewrite/fusion graphs, pooled allocations, NN helpers, convenience ops, embeddings, matmul/linear helpers, and module wrappers.
- `runtime/quant` validates MLX/GPTQ/Q4 tensor layouts, checked shape/expected-size/dequant output arithmetic, NVFP4 unpack/dequant bounds without overflow-prone packed-count multiplication, and no-ops or returns nil on malformed in-memory weights.
- `runtime/kv` and `runtime/memory` guard cache dimensions/layouts, compressed-cache accessor/memory accounting, staging rollback arithmetic, TurboQuant sizing/packed-byte calculations, protected-layer helper inputs, mmap range overflow, malformed tracked ranges, and nil advisor receivers.
- `gpu/` CUDA helpers preflight dimensions, upload/download state, device pointers, stream launches, graph executables, copy wrappers, allocation and byte-size arithmetic, Q4/MLX/NVFP4 weight layouts, expert IDs, experimental NV ioctl/memory/query setup, dense SGEMM/LM-head buffers, JIT/NVFP4 kernel specs, and BF16 buffers before dispatch; failed `DevBuf` transfers preserve authoritative state or fall back safely, and CUDA/NV progress diagnostics are quiet unless `GO_PHERENCE_GPU_DEBUG` is set.
- `backends/simd` scalar fallbacks bound all input/output slices, BF16 GEMV checks shape-product overflow, scalar RMSNorm uses precise `math.Sqrt`, empty vector/BF16 calls avoid assembly stubs, GEBP scratch is per-call, and SGEMM/GEBP/gather helpers preflight dimensions, pointers, strides, CPU capability gates, checked byte offsets, and overflow before unsafe pointer arithmetic.
- `loader/safetensors` validates dtype byte sizes against shapes/offsets at open time; file/sharded helpers are nil-safe, names are sorted deterministically, partial sharded opens clean up already-open shards, sharded eager-load totals are checked, tokenizer byte maps are initialized with `sync.Once`, and malformed tokenizer BPE merges are rejected.
- Transitional `model` helpers validate MTP token/KV keep counts, model-aware MTP verifier plan/logit/activation dimensions, shared-KV verifier sources, MTP acceptance consistency before KV commit, alias-safe MTP drafter projection sizing, q-only drafter external-KV/layer dimensions, bounded multi-draft counts, speculative stats overflow/rollback paths, zero-count state copy semantics, CPU decode final norm/LM-head dimensions, CPU generation allocation setup, MoE edge cases, embedding/LM-head/per-layer input backing data, chunked LM-head and batched-prefill dimensions, CPU forward-layer entrypoints, model-specific KV width overflow, and low-level GEMV/GQA product arithmetic; loader/prefill/GPU placement diagnostics are quiet unless `GO_PHERENCE_LOAD_DEBUG` or `GO_PHERENCE_PREFILL_DEBUG` is set.

Fast refactor validation remains focused to avoid accidentally loading large local model fixtures:

```bash
go test ./tensor -count=1
go test ./gpu ./loader/... ./backends/cuda/ptx ./backends/placement ./backends/simd ./backends/vulkan ./runtime/... ./models/bert ./tensor ./cmd/... -run '^$'
go vet ./...
```

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — UOp graph, fusion, SIMD dispatch
- **[docs/gemma4-precision.md](docs/gemma4-precision.md)** — Gemma4 GPU correctness & precision
- **[docs/weight-budget.md](docs/weight-budget.md)** — tiered weight budget manager (ds4-inspired)
- **[docs/nvfp4.md](docs/nvfp4.md)** — NVFP4/FP4 support track and relevant Gemma/Qwen checkpoint findings
- **[docs/mtp-speculative.md](docs/mtp-speculative.md)** — Gemma4/Qwen3.6 MTP research plus current internal implementation status
- **[docs/orthrus.md](docs/orthrus.md)** — Orthrus analysis and stock-weight speculative decoding scaffold/benchmark notes
- **[docs/qwen36-mtp.md](docs/qwen36-mtp.md)** — Qwen3.6 27B native MTP checkpoint findings and shortest implementation path
- **[docs/performance.md](docs/performance.md)** — benchmarks, kernel timings
- **[docs/gpu-options.md](docs/gpu-options.md)** — GPU compute paths (CUDA, Vulkan)
- **[docs/development-log.md](docs/development-log.md)** — build process
- **[docs/refactor-plan.md](docs/refactor-plan.md)** — Phase 6.5 source-tree refactor plan

## License

MIT
