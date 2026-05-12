# Source-tree refactor plan

Status: in progress, blocking new MTP/backend functionality until the validation gate is complete.

This document captures the Phase 6.5 assessment and target package design. The refactor should be mostly mechanical at first: move code into ownership boundaries, keep CLI behavior stable where practical, but allow intentional package/API breaking changes instead of preserving old wrappers.

## Why this is now blocking

The current repository has reached the point where adding more MTP, backend, and quantization behavior into the existing package layout will make later extraction harder. In particular, Gemma4 MTP touches loading, tokenizer/config handling, architecture-specific forward logic, KV caches, CPU/GPU backends, and generation loops. Those concerns currently meet inside `model/`, so new code would further entangle them.

The immediate goal is not to make the architecture perfect. It is to create enough package boundaries that future work has obvious homes.

## Current map

Current Go packages from `go list ./...` after the loader/backend/runtime moves completed so far:

```text
backends/placement   -> backend-neutral memory budget and layer placement policy
backends/simd        -> AVX2/FMA/NEON dispatch and kernels
backends/vulkan      -> Vulkan loader/device/buffer/shader dispatch + SPIR-V assets
cmd/llmchat          -> imports gpu, loader/tokenizer, model
cmd/llmgen           -> imports loader/tokenizer, model
cmd/llmserver        -> imports gpu, loader/tokenizer, model
cmd/tinydemo         -> imports tensor
gpu                  -> CUDA/PTX path plus GPU-resident expert pool
loader/config        -> config.json and quantize_config JSON helpers
loader/safetensors   -> mmap safetensors reader and sharded reader with metadata validation
loader/tokenizer     -> tokenizer.json and BPE/SentencePiece-compatible encode/decode; race-safe byte maps
loader/weights       -> common safetensors source opener for sharded/single-file weights
model                -> LLaMA-family loader/types, CPU forward, GPU model wrapper,
                        MoE, MTP scaffold, model-specific KV sizing
models/bert          -> GTE/BERT encoder path
runtime/kv           -> TurboQuant state, compressed KV cache, float/compressed KV staging with overflow/layout guards
runtime/memory       -> mmap residency advice and range tracking; nil/invalid ranges are inert
runtime/quant        -> MLX/GPTQ CPU quant formats, dequantization, Q4 GEMV helpers
tensor               -> tensor graph/runtime; transitional direct import of backends/simd; hardened malformed-input guards
```

Current import direction:

```text
cmd -> loader/tokenizer, model, gpu
model -> backends/simd, gpu, loader/{config,tokenizer,weights}, runtime/{kv,quant}, tensor
models/bert -> backends/simd, loader/safetensors, tensor
loader/safetensors -> runtime/memory
loader/weights -> loader/safetensors
tensor -> backends/simd
gpu -> backends/placement, purego/unix only
```

Initial assessment before the first moves found root `safetensors/`, root `simd/`, tokenizer code inside `model/`, and GTE/BERT code inside `model/`. Those have now been moved to their target owners. Remaining hotspots:

```text
model/llama.go        ~1560 lines: config normalization, model structs, loader,
                       quant loading, architecture-specific Gemma/Qwen/MoE branches
model/gpu_forward.go  ~1150 lines: GPU-resident model, upload policy, Gemma4 GPU forward,
                       hybrid/layer-split behavior
gpu/mlx_cuda.go       runtime MLX CUDA upload/dispatch helpers
gpu/devbuf.go         ~620 lines: CUDA memory abstraction and vector ops
```

## Main problems found

1. **`model` remains a catch-all package.** Tokenizer, safetensors, config helpers, weight-source opening, SIMD, BERT/GTE, generic KV/TurboQuant, and MLX/GPTQ CPU quant helpers have moved out, but `model` still contains LLaMA-family model definitions, CPU kernels, GPU orchestration, MoE, model-specific KV sizing, and MTP scaffold.
2. **Backends are not cleanly separated.** `gpu/` still mixes CUDA/PTX, GPU expert pool, and memory abstractions, although backend-neutral budget/placement policy has moved to `backends/placement` and Vulkan scaffolding/assets have moved to `backends/vulkan`. `model/gpu_forward.go` imports and orchestrates GPU details directly.
3. **Architecture-specific behavior is mixed into generic names.** `LlamaModel` currently also carries Qwen, Gemma3, Gemma4, MoE, TurboQuant, and MTP concerns.
4. **Loading is still coupled to architecture structs.** `LoadLlama` now uses `loader/config`, `loader/weights`, and `runtime/quant`, and load-time panics are recovered as returned errors, but it still normalizes config, applies quant format choices, and fills architecture-specific weights in one flow.
5. **Generation APIs hide backend policy.** CLI code toggles global state such as `model.ForceOnTheFly`, then chooses CPU/GPU behavior after loading.
6. **Tensor/runtime/backend/loader guards are now part of the refactor baseline.** The tensor graph/runtime has explicit validation for malformed shapes, nil receivers/UOps, reduction axes, unsafe reinterpret helpers, realization internals, pooled allocation overflow, rewrite/fusion paths, embedding/matmul helpers, NN helpers, and module wrappers. SIMD fallbacks and SGEMM/GEBP wrappers are bounds/overflow guarded. Runtime KV/TurboQuant/mmap helpers validate layouts/arithmetic/nil receivers. Safetensors metadata and tokenizer byte maps are guarded. Keep these guards intact during later backend/model moves.
7. **Tests mix durable validation with diagnostics.** Gemma4 trace/sensitivity/generation diagnostics now carry a `diagnostic` build tag and still require `GEMMA4_TRACE_TEST=1`, but they still live beside normal unit tests until the model-package split.
8. **Local asset assumptions leak into tests.** Tests use paths such as `../models/gemma4-e2b-mlx4`, `../models/smollm2-135m`, and `../../gte-go/models/gte-small/model.safetensors`; these need explicit fixture policy during later moves. `.gitignore` now ignores downloaded model assets under `models/*` while allowing source package folders such as `models/bert`, `models/gemma4`, and `models/qwen3`.

## Target layout

This is the target shape after the mechanical move phase. Names may be adjusted during implementation, but the ownership rules should remain stable.

```text
cmd/
  llmgen/
  llmchat/
  llmserver/
  tinydemo/

loader/
  config/              # config.json and quantize_config parsing
  tokenizer/           # tokenizer.json, chat templates, special-token rules
  safetensors/         # moved current safetensors package
  weights/             # common tensor-name lookup/source interfaces
  detect/              # architecture/quant format detection

models/
  shared/              # transformer-level types used by multiple architecture packages
  llama/               # LLaMA-family baseline implementation
  qwen2/
  qwen3/
  gemma3/
  gemma4/
    mtp/               # Gemma4 assistant drafter/verifier scaffold
  bert/                # GTE/BERT embedding path
  moe/                 # MoE architecture components and routing policy

runtime/
  generation/          # Generate/streaming/speculative decode orchestration
  kv/                  # float KV, staged KV, compressed KV, paging/compression interfaces
  quant/               # GPTQ, MLX4, future NVFP4 shared quant formats
  memory/              # mmap advisory, eager-load, mlock/pinned-memory policy

backends/
  cpu/                 # scalar/SIMD-backed CPU inference operations
  simd/                # moved current SIMD package
  cuda/                # CUDA driver bindings, PTX kernels, device buffers
  vulkan/              # Vulkan device/shader/dispatch path
  placement/           # layer/expert/budget policy spanning backends
  shared/              # backend-neutral interfaces only

tensor/                # tensor graph/runtime; keep stable initially
scripts/
docs/
```

### Breaking-change migration policy

This refactor is allowed to break package-level APIs. Prefer updating call sites to the new package owner over adding compatibility wrappers.

- Do not keep `model.LoadTokenizer`; callers should import `loader/tokenizer` directly.
- Do not add new feature code to transitional old packages.
- CLI behavior and flags should remain stable unless explicitly changed, but Go import paths and internal APIs may change.
- If a temporary bridge is unavoidable for a large move, document it with a removal commit/phase before adding further functionality.

## Import rules

The desired dependency direction is:

```text
cmd -> runtime/generation -> loader -> models -> runtime/{kv,quant} -> backends -> tensor
# transitional: tensor may import backends/simd until CPU backend interfaces are split
```

More precise rules:

1. `cmd/*` contains flags, logging, process setup, and user I/O only.
2. `loader/*` may import `loader/safetensors`, `loader/tokenizer`, `loader/config`, `models/*`, `runtime/quant`, and `tensor`.
3. `models/*` must not import `cmd` or concrete CLI code.
4. `models/*` should depend on backend-neutral interfaces for matmul/attention/norm where practical; direct backend imports are transitional only.
5. `backends/*` must not import `cmd` or architecture-specific model packages.
6. `backends/cuda` and `backends/vulkan` must not import each other.
7. `runtime/kv`, `runtime/quant`, and `runtime/memory` should hold shared runtime state that is not architecture-specific.
8. `shared/` folders are local escape hatches only. Do not create a global dumping ground.
9. New feature code should not be added to transitional old packages or temporary bridges.

## Proposed ownership moves

### Loader/config/tokenizer

Move/update directly:

- `model/tokenizer.go` -> `loader/tokenizer`
- config parsing helpers for `model/llama.go` and `model/mtp_drafter.go` -> `loader/config` ✅; `models/bert` currently uses a fixed GTE-small config
- tensor source interface formerly local to `LoadLlama` -> `loader/weights` ✅
- `safetensors/` -> `loader/safetensors` ✅; call sites import the new owner directly, no old import-path wrapper

### Runtime KV/quant/memory

Move/update directly:

- `model/kv_cache.go` and the generic staging parts of `model/kv_staging.go` -> `runtime/kv` ✅
- `model/turboquant.go` -> `runtime/kv` ✅
- `model/gptq.go`, `model/mlx.go`, `model/gemv_q4.go` -> `runtime/quant` ✅; runtime now validates MLX scale/bias dtypes, GPTQ qweight/g_idx/scales/qzeros, and Q4 GEMV call inputs; `model/bf16.go` remains with BF16 model semantics for now
- `gpu/budget.go`, `gpu/placement.go` -> `backends/placement` ✅; `gpu/expert_pool.go` stays in `gpu` because it owns `GPUMLXWeight` device resources
- `loader/safetensors/mmap_advisor.go` -> `runtime/memory` ✅; safetensors keeps only file/eager-load integration tests

### Backends

Move/update directly:

- CUDA driver/PTX: embedded PTX source assets have moved to `backends/cuda/ptx` ✅, covering attention/RoPE, core vector/norm/activation kernels, LM head, Q4 GEMV/GEMM, SGEMM, prefetch, BF16, and MLX kernels. Runtime CUDA dispatch/types remain in transitional `gpu` until `DevBuf`, upload state, quantized GPU weights, expert resources, and model orchestration can be split without compatibility wrappers. Recent audit passes hardened `DevBuf`, stream/copy wrappers, Q4/MLX dispatch validation, and GPU pointer call sites before the larger runtime split.
- Vulkan: `gpu/vulkan*.go`, `gpu/shaders/` -> `backends/vulkan` ✅; dispatch wiring remains a Phase 3.6 implementation task
- `simd/` -> `backends/simd` ✅; tensor/model imports now point at the backend owner directly
- CPU backend loops now in `model/forward_layer.go`, `model/inference_helpers.go`, `model/moe.go` should move only after model packages can call backend interfaces cleanly

### Models

Move or split:

- `model/llama.go` into shared config/types plus architecture-specific loaders/weights
- `model/forward_layer.go`, `model/inference_helpers.go`, `model/chunked_lm_head.go` into `models/shared` or `runtime/generation`
- Gemma4-specific tests/helpers and PLI/KV-sharing logic into `models/gemma4`
- `model/mtp_*` into `models/gemma4/mtp`
- `model/moe.go`, `model/moe_gpu.go` into `models/moe` plus backend hooks
- `model/bert.go`, `model/forward_fast.go`, `model/workspace.go` -> `models/bert` ✅

## Test policy

Classify tests before moving them:

- **Durable unit tests**: pure helpers, tensor math, safetensors parsing, SIMD dispatch, KV staging, MTP acceptance, tokenizer, quant roundtrips.
- **Fixture smoke tests**: SmolLM2/Gemma4/GTE paths that require local model files but are small enough to run intentionally.
- **Heavy diagnostics**: Gemma4 GPU layerwalk/optrace/sensitivity/generation tests tagged with `//go:build diagnostic` and gated by `GEMMA4_TRACE_TEST=1`; move to package-specific diagnostic folders/files during the later model split.

Do not let the mechanical move phase accidentally ungate heavy diagnostics.

## Migration sequence

Each step should be one small commit with validation after it.

1. **Add docs and package stubs only.** Land this plan and any README notes. No behavior changes.
2. **Loader extraction.** Move tokenizer/config/safetensors-facing loader boundaries first and update call sites directly.
3. **Runtime KV/quant extraction.** Move KV staging/cache and quant formats to runtime packages, updating call sites in the same commit. ✅
4. **Backend split.** Vulkan scaffolding/assets have moved to `backends/vulkan` ✅. Embedded CUDA PTX source assets have moved to `backends/cuda/ptx` ✅. CUDA runtime dispatch still remains in `gpu` until the split can preserve model upload/DevBuf semantics cleanly.
5. **Model split.** Move BERT/GTE first ✅, then LLaMA-family shared code, Gemma4, MoE, and MTP scaffold into architecture packages.
6. **Generation runtime.** Move CPU/GPU/speculative generation loops into `runtime/generation` once backends/models have clean interfaces.
7. **Test quarantine.** Gemma4 diagnostic tests are build-tagged ✅; later model-package split should move them next to their architecture package.
8. **Remove temporary bridges.** Only if any were unavoidable during earlier large moves.

## Validation gate

Run after every non-trivial move. Include `runtime/kv` and `runtime/quant` in the fast package set because they now own shared KV and CPU quantization behavior:

```sh
go test ./tensor ./backends/simd ./runtime/... ./loader/... -count=1
go test ./gpu ./loader/... ./backends/cuda/ptx ./backends/placement ./backends/simd ./backends/vulkan ./runtime/... ./models/bert ./tensor ./cmd/...
go test ./model -run 'TestMTP|Test.*KV|TestTokenizer|TestGQAAttention|TestMLX|TestBF16' -count=1
go vet ./...
git diff --check
```

Final Phase 6.5 validation before resuming MTP/verifier/drafter work:

```sh
go test ./...
go vet ./...
```

If `go test ./...` is too memory-heavy with local fixtures, document the failure mode and run the focused package set plus explicit smoke tests:

```sh
go test ./tensor ./backends/simd ./runtime/... ./loader/... -count=1
go test ./gpu ./loader/... ./backends/cuda/ptx ./backends/placement ./backends/simd ./backends/vulkan ./runtime/... ./models/bert ./tensor ./cmd/...
go test ./model -run 'TestFloatKV|TestCompressedKV|TestMTP|TestLayerKVDim|TestGQAAttention|TestMLX|TestBF16|TestLoadLlama|TestGenerateSmolLM2' -count=1
go test ./models/bert -count=1
go run ./cmd/llmgen -model models/smollm2-135m -prompt 'Hello' -tokens 2
go run ./cmd/llmgen -model models/gemma4-e2b-mlx4 -prompt 'Hello' -tokens 2
```

GPU smoke tests should remain explicit and environment-gated:

```sh
GEMMA4_TRACE_TEST=1 go test -tags diagnostic ./model -run 'TestGemma4GPUGenerate|TestGemma4KVSharingGPU' -count=1
```

## Definition of done for Phase 6.5

- `docs/refactor-plan.md` exists and is current.
- Package ownership and import rules are enforced by review, and ideally by a small script later.
- Major files are moved out of `model/` catch-all into loader/runtime/backend/model packages.
- No compatibility wrappers remain unless explicitly documented as short-lived temporary bridges.
- Diagnostic tests are separated or clearly gated.
- CLI behavior and flags are unchanged.
- Validation gate passes and the refactor is committed/pushed before more MTP or backend functionality lands.
