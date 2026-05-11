# Source-tree refactor plan

Status: planned, blocking new MTP/backend functionality until the validation gate is complete.

This document captures the Phase 6.5 assessment and target package design. The refactor should be mostly mechanical at first: move code into ownership boundaries, keep public behavior stable, and defer semantic rewrites until imports are clean.

## Why this is now blocking

The current repository has reached the point where adding more MTP, backend, and quantization behavior into the existing package layout will make later extraction harder. In particular, Gemma4 MTP touches loading, tokenizer/config handling, architecture-specific forward logic, KV caches, CPU/GPU backends, and generation loops. Those concerns currently meet inside `model/`, so new code would further entangle them.

The immediate goal is not to make the architecture perfect. It is to create enough package boundaries that future work has obvious homes.

## Current map

Current Go packages from `go list ./...`:

```text
cmd/llmchat     -> imports gpu, model
cmd/llmgen      -> imports model
cmd/llmserver   -> imports gpu, model
cmd/tinydemo    -> imports tensor
gpu             -> CUDA, Vulkan, budgets, placement, expert pool, PTX/SPIR-V embedding
model           -> loader, configs, tokenizer, architectures, CPU forward, GPU model wrapper,
                   quant formats, MoE, MTP scaffold, KV cache, TurboQuant, GTE/BERT
safetensors     -> mmap safetensors reader, sharded reader, mmap advisor
simd            -> AVX2/FMA/NEON dispatch and kernels
tensor          -> tensor graph/runtime; imports simd
```

Import direction today:

```text
cmd -> model -> gpu
cmd -> gpu
model -> safetensors, simd, tensor
tensor -> simd
gpu -> purego/unix only
```

Notable file-size hotspots:

```text
model/llama.go        ~1560 lines: config, model structs, loader, tokenizer-adjacent policy,
                       quant loading, architecture-specific Gemma/Qwen/MoE branches
model/gpu_forward.go  ~1150 lines: GPU-resident model, upload policy, Gemma4 GPU forward,
                       hybrid/layer-split behavior
gpu/mlx_ptx.go        ~750 lines: MLX CUDA kernel source/dispatch helpers
gpu/devbuf.go         ~620 lines: CUDA memory abstraction and vector ops
gpu/attn_ptx.go       ~560 lines: attention kernels
```

## Main problems found

1. **`model` is a catch-all package.** It contains model definitions, loader code, tokenizer logic, CPU kernels, GPU orchestration, quantized formats, MoE, KV caches, TurboQuant, MTP scaffold, and GTE/BERT code.
2. **Backends are not cleanly separated.** `gpu/` mixes CUDA, Vulkan, budget/placement, expert pool, and memory abstractions. `model/gpu_forward.go` imports and orchestrates GPU details directly.
3. **Architecture-specific behavior is mixed into generic names.** `LlamaModel` currently also carries Qwen, Gemma3, Gemma4, MoE, TurboQuant, and MTP concerns.
4. **Loading is coupled to architecture structs.** `LoadLlama` parses config, opens safetensors, handles sharding/eager loading, applies quant format choices, and fills architecture-specific weights in one flow.
5. **Generation APIs hide backend policy.** CLI code toggles global state such as `model.ForceOnTheFly`, then chooses CPU/GPU behavior after loading.
6. **Tests mix durable validation with diagnostics.** Many Gemma4 trace/sensitivity tests are correctly gated with `GEMMA4_TRACE_TEST=1`, but they live beside normal unit tests and make the package look larger and riskier than it is.
7. **Local asset assumptions leak into tests.** Tests use paths such as `../models/gemma4-e2b-mlx4`, `../models/smollm2-135m`, and `../../gte-go/models/gte-small/model.safetensors`; these need explicit fixture policy during moves.

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
  safetensors/         # moved or wrapped current safetensors package
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
  simd/                # moved or wrapped current simd package
  cuda/                # CUDA driver bindings, PTX kernels, device buffers
  vulkan/              # Vulkan device/shader/dispatch path
  placement/           # layer/expert/budget policy spanning backends
  shared/              # backend-neutral interfaces only

tensor/                # tensor graph/runtime; keep stable initially
scripts/
docs/
```

### Compatibility during migration

To avoid breaking every call site in one commit, keep thin wrappers for one transition phase:

- `model.LoadLlama` delegates to the new loader/model constructor.
- `model.LoadTokenizer` delegates to `loader/tokenizer`.
- `model.GPUModel` can temporarily alias the new backend/runtime type.
- Existing `gpu` and `safetensors` packages can stay as compatibility packages until imports are moved.

Remove compatibility wrappers only after CLI, tests, docs, and examples no longer need them.

## Import rules

The desired dependency direction is:

```text
cmd -> runtime/generation -> loader -> models -> runtime/{kv,quant} -> backends -> tensor/simd
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
9. New feature code should not be added to compatibility wrappers.

## Proposed ownership moves

### Loader/config/tokenizer

Move or wrap:

- `model/tokenizer.go` -> `loader/tokenizer`
- config parsing from `model/llama.go`, `model/bert.go`, `model/mtp_drafter.go` -> `loader/config`
- tensor source interface currently local to `LoadLlama` -> `loader/weights`
- `safetensors/` -> either `loader/safetensors` or `formats/safetensors`; keep old `safetensors` import path as a wrapper if needed

### Runtime KV/quant/memory

Move or wrap:

- `model/kv_cache.go`, `model/kv_staging.go` -> `runtime/kv`
- `model/turboquant.go` -> `runtime/kv` or `runtime/quant` depending on interface split
- `model/gptq.go`, `model/mlx.go`, `model/gemv_q4.go`, `model/bf16.go` -> `runtime/quant` plus `backends/cpu` kernels where appropriate
- `gpu/budget.go`, `gpu/placement.go`, `gpu/expert_pool.go` -> `backends/placement` or `runtime/memory` depending on whether they own device resources
- `safetensors/mmap_advisor.go` -> `runtime/memory` if it becomes format-agnostic; otherwise keep under safetensors for now

### Backends

Move or wrap:

- CUDA driver/PTX: `gpu/cuda_purego.go`, `gpu/devbuf.go`, `gpu/*_ptx.go`, `gpu/mega_module.go`, `gpu/streams.go`, `gpu/q4_gpu.go`, `gpu/sgemm.go` -> `backends/cuda`
- Vulkan: `gpu/vulkan*.go`, `gpu/shaders/` -> `backends/vulkan`
- SIMD: `simd/` -> `backends/simd` eventually; keep root `simd` wrapper until tensor/model imports are migrated
- CPU backend loops now in `model/forward_layer.go`, `model/inference_helpers.go`, `model/moe.go` should move only after model packages can call backend interfaces cleanly

### Models

Move or split:

- `model/llama.go` into shared config/types plus architecture-specific loaders/weights
- `model/forward_layer.go`, `model/inference_helpers.go`, `model/chunked_lm_head.go` into `models/shared` or `runtime/generation`
- Gemma4-specific tests/helpers and PLI/KV-sharing logic into `models/gemma4`
- `model/mtp_*` into `models/gemma4/mtp`
- `model/moe.go`, `model/moe_gpu.go` into `models/moe` plus backend hooks
- `model/bert.go`, `model/forward_fast.go`, `model/workspace.go` into `models/bert`

## Test policy

Classify tests before moving them:

- **Durable unit tests**: pure helpers, tensor math, safetensors parsing, SIMD dispatch, KV staging, MTP acceptance, tokenizer, quant roundtrips.
- **Fixture smoke tests**: SmolLM2/Gemma4/GTE paths that require local model files but are small enough to run intentionally.
- **Heavy diagnostics**: Gemma4 GPU layerwalk/optrace/sensitivity/generation tests gated by `GEMMA4_TRACE_TEST=1`; move to package-specific diagnostic files or add clearer build tags later.

Do not let the mechanical move phase accidentally ungate heavy diagnostics.

## Migration sequence

Each step should be one small commit with validation after it.

1. **Add docs and package stubs only.** Land this plan and any README notes. No behavior changes.
2. **Loader extraction.** Move tokenizer/config/safetensors-facing loader boundaries first. Keep compatibility wrappers under `model`.
3. **Runtime KV/quant extraction.** Move KV staging/cache and quant formats to runtime packages. Keep `model` wrappers until all call sites move.
4. **Backend split.** Separate CUDA and Vulkan into `backends/cuda` and `backends/vulkan`; keep `gpu` compatibility wrapper if required.
5. **Model split.** Move BERT/GTE, LLaMA-family shared code, Gemma4, MoE, and MTP scaffold into architecture packages.
6. **Generation runtime.** Move CPU/GPU/speculative generation loops into `runtime/generation` once backends/models have clean interfaces.
7. **Test quarantine.** Move or tag diagnostic tests; keep focused unit tests close to packages.
8. **Remove wrappers.** Only after CLI/docs/tests use the new import paths.

## Validation gate

Run after every non-trivial move:

```sh
go test ./gpu ./safetensors ./simd ./tensor ./cmd/...
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
go test ./gpu ./safetensors ./simd ./tensor ./cmd/...
go test ./model -run 'TestFloatKV|TestCompressedKV|TestMTP|TestLayerKVDim|TestGQAAttention|TestMLX|TestBF16|TestLoadLlama|TestGenerateSmolLM2' -count=1
go run ./cmd/llmgen -model models/smollm2-135m -prompt 'Hello' -tokens 2
go run ./cmd/llmgen -model models/gemma4-e2b-mlx4 -prompt 'Hello' -tokens 2
```

GPU smoke tests should remain explicit and environment-gated:

```sh
GEMMA4_TRACE_TEST=1 go test ./model -run 'TestGemma4GPUGenerate|TestGemma4KVSharingGPU' -count=1
```

## Definition of done for Phase 6.5

- `docs/refactor-plan.md` exists and is current.
- Package ownership and import rules are enforced by review, and ideally by a small script later.
- Major files are moved out of `model/` catch-all into loader/runtime/backend/model packages.
- Compatibility wrappers are either removed or documented as temporary.
- Diagnostic tests are separated or clearly gated.
- CLI behavior and flags are unchanged.
- Validation gate passes and the refactor is committed/pushed before more MTP or backend functionality lands.
