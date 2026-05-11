# Development Log

Step-by-step record of building go-pherence from scratch.

## Session 1: Core framework + GTE-small inference

### Step 1 — Analyze tinygrad architecture

Studied tinygrad's Python source to identify the core abstractions:
- **UOp**: single IR node type (replaces older LazyBuffer)
- **Ops enum**: ~60 operations covering movement, ALU, reduce, memory
- **DType**: data type system with float16/32/64, int types
- Lazy DAG evaluation with `realize()` triggering execution
- Backend-agnostic: CPU, CUDA, Metal all use the same graph

Key insight for Go port: implement UOp interning + eager interpreter first,
add fusion and scheduling later.

### Step 2 — Core types (tensor/, 7 files)

Built the foundation:
- `dtype.go`: Float32, Int32, Bool, etc.
- `ops.go`: 30+ operations with category methods (IsUnary, IsBinary, IsReduce)
- `uop.go`: hash-consed DAG node with `sync.Map` interning
- `shape.go`: dimensions + strides, reshape, permute, expand
- `tensor.go`: user API — constructors, lazy binary/unary/reduce ops, Realize()
- `realize.go`: recursive eager interpreter for UOp graphs
- `unsafe.go`: zero-copy byte↔float32 conversions

**Bug found**: UOp interning caused buffer sharing — two tensors with same
shape got the same UOp node, overwriting each other's data. Fix: don't
intern Buffer UOps (they're unique by identity).

**Bug found**: reduce indexing used wrong strides (output shape strides vs
input shape strides). Fix: use `NewShape(outDims).Strides` for output index.

11 tests passing.

### Step 3 — SIMD kernels + MatMul

Copied the full SIMD assembly suite from gte-go:
- `Sdot`, `Saxpy`: AVX2+FMA / NEON vector ops
- `SgemmNN`, `SgemmNT`: matrix multiply with tiled assembly
- `GEBP`: General Block Panel micro-kernels
- `VGATHERDPS`: AVX2 gather for NT without packing

Rewrote `MatMul` to use `SgemmNN` instead of per-column `Sdot`.
Result: **14.7ms → 0.5ms (29× faster)** for 64×384 @ 384×384.

Added `MatMulTransposed` for the `Y = X @ W^T` pattern used by Linear layers.

### Step 4 — Broadcasting

Implemented shape broadcasting for binary ops:
- Automatic shape expansion ([2,3] + [3] → [2,3])
- `BroadcastArg` struct stores input shapes for realize indexing
- Stride-based broadcast index computation in `binaryBroadcastEval`

**Bug found**: `[3][]int` array type assertion failed silently in Go.
Fix: use named struct `BroadcastArg` instead of anonymous array type.

### Step 5 — NN operations

Added high-level ops:
- `Softmax`: numerically stable (max subtraction)
- `LayerNorm`: with gamma/beta affine transform
- `GELU`: tanh approximation matching the standard formula
- `Permute`: correct transpose via per-element index mapping

**Bug found**: Permute source index mapping was wrong (forward instead of
inverse permutation). Fix: `srcIdx[order[d]] = outIdx[d]`.

### Step 6 — Elementwise fusion

Built fusion engine (`fuse.go`):
- Walks UOp DAG, identifies chains of fusible elementwise ops
- Compiles to flat `fusedOp` list with buffer references
- Executes all ops per-element in one pass (no intermediate buffers)
- Skips broadcast ops (different buffer sizes)

Performance: **Add+Mul 888ns → 441ns (2× faster)**, 5-op chain 2.4× faster.

### Step 7 — Numpy reference tests

Generated ground-truth values from numpy (seed=42) for all ops.
20 reference tests verify bit-level reproducibility:
- Binary: add, sub, mul, div (atol=1e-6)
- Unary: neg, sqrt, exp2, log2, recip
- Reduce: sum/max over both axes
- MatMul: forward and transposed
- NN: softmax, layernorm, gelu, linear
- Movement: permute, broadcast

### Step 8 — Safetensors loader

Implemented HuggingFace safetensors format reader:
- JSON header parsing for tensor metadata
- F16 → F32 conversion (IEEE 754 half-precision with subnormals)
- BF16, I32, I64 support
- Tested against GTE-small: 200 tensors loaded successfully

### Step 9 — BERT encoder + GTE-small inference

Built complete BERT model (now owned by `models/bert/` after the Phase 6.5 refactor):
- `LoadGTESmall`: load all weights from safetensors
- `Forward`: word + position + type embeddings → 12 transformer layers
- `multiHeadAttention`: per-head Q·K^T with softmax
- `Embed`: mean pooling + L2 normalization

**Result**: embeddings match gte-go reference within F16 tolerance.
Forward pass: ~30ms for 5-token input.

### Step 10 — Performance comparison

| | go-pherence | gte-go |
|---|---|---|
| Latency | 30ms | 10ms |
| Allocs/embed | 1,672 | 1 |
| Memory/embed | 3.5 MB | 7 B |
| Correctness | ✅ | ✅ |
| Lines of code | 4,240 | ~8,000 |
| Model format | Safetensors | Custom .gtemodel |

3× gap from: per-op buffer allocation, scalar attention, no fused
residual+layernorm, tensor object overhead.

## Test inventory

| Package | Tests | Coverage |
|---|---|---|
| `tensor/` — unit tests | 22 | all ops, lazy eval, fusion |
| `tensor/` — numpy reference | 20 | bit-level reproducibility |
| `loader/safetensors/` | 3 | load, list, F16 conversion |
| `models/bert/` | 2 | load weights, end-to-end embed |
| **Total** | **47** | |


## Session 2: Gemma4 MTP speculative decoding scaffolding

Implemented the first native safetensors-based Gemma4 MTP building blocks:

- Documented LiteRT-LM's Gemma4 MTP flow and the local `gemma4-e2b-mtp-drafter` asset.
- Added `LoadGemma4MTPDrafter` for `gemma4_assistant` q-only drafter assets.
- Hardened drafter loading with exact tensor shape validation, malformed-config checks, and explicit `KVSourceLayer=-1` external-KV markers.
- Added assistant helper methods for token embedding row copy, masked ordering lookup, `PreProjectInto`, and `PostProjectInto`.
- Extracted main-model helper primitives for raw/scaled token embeddings, Gemma4 per-layer inputs, LM-head logits, and greedy argmax; `Generate` now uses the shared helpers.
- Added KV staging checkpoints for uncompressed and TurboQuant-backed caches, including accepted-prefix plus verifier bonus-token commit.
- Added LiteRT-style MTP acceptance accounting from verifier token IDs or verifier logits.

Current status: MTP is not yet exposed as a generation mode. The remaining work is the batched main-model verifier forward path and the q-only drafter forward loop that consumes external/main-model KV state and projected activations.

## Session 3: Phase 6.5 source-tree refactor start

Started the blocking source-tree refactor before adding more MTP/backend functionality:

- Added `docs/refactor-plan.md` with package ownership rules, target folder layout, migration sequence, and validation gate.
- Moved tokenizer code from `model` to `loader/tokenizer`; callers import the new owner directly.
- Moved root `safetensors` package to `loader/safetensors`.
- Added `loader/config` for config JSON helpers and `loader/weights` for shared sharded/single-file safetensors opening.
- Moved root `simd` package to `backends/simd` while keeping package name `simd`.
- Moved the GTE/BERT encoder path from `model` to `models/bert`.

Compatibility wrappers are intentionally avoided; package/API breaks are part of this internal refactor while CLI behavior remains stable.

## Session 4: Runtime KV/quant extraction and hardening

Continued the Phase 6.5 mechanical refactor by moving shared runtime concerns out of the transitional decoder package:

- Moved generic TurboQuant state, compressed KV cache, and float/compressed KV staging helpers from `model` to `runtime/kv`.
- Kept model-specific KV width derivation in `model` so Gemma4 variable/shared KV layout remains architecture-owned.
- Moved MLX/GPTQ CPU quantization helpers from `model` to `runtime/quant`, including MLX affine weights, GPTQ dequantization, and scalar Q4 GEMV helpers.
- Updated model loader/forward, MoE, GPU fallback, benchmarks, and diagnostics to import `runtime/kv` and `runtime/quant` directly.
- Hardened `runtime/quant.LoadMLXWeight` with packed-weight config validation, shape inference, and scale/bias length checks.
- Converted LLaMA and GTE load-time panics into returned errors, and stopped ignoring GPTQ scale/qzero load failures.

Validation covered the new runtime packages, focused model tests, backend/loader/tensor/cmd packages, `go test ./... -run '^$'`, `go vet ./...`, and `git diff --check`.

## Session 5: Placement policy extraction

Continued Phase 6.5 by separating backend-neutral placement policy from GPU device-resource ownership:

- Moved `gpu/budget.go` and `gpu/placement.go` to `backends/placement`.
- Made placement planning accept caller-supplied available device memory instead of calling CUDA `MemInfo()` directly.
- Kept `gpu/expert_pool.go` in `gpu` because `ExpertEntry` owns `GPUMLXWeight` resources that must be freed through the GPU backend.
- Updated expert-pool accounting to depend on `backends/placement.BudgetManager`.
- Updated Makefile and docs so the fast validation set includes `backends/placement`.

## Session 6: Runtime memory extraction

Moved mmap residency policy to a runtime owner:

- Moved `loader/safetensors/mmap_advisor.go` to `runtime/memory` because it only needs an mmap'd byte slice and is not safetensors-specific.
- Updated `loader/safetensors.File` to hold `*memory.MmapAdvisor` and create it via `memory.NewMmapAdvisor`.
- Split tests so generic advisor range/merge behavior lives in `runtime/memory`, while safetensors keeps file/eager-load integration tests.
- Updated docs to describe `runtime/memory` as the owner for mmap advice and future streaming policy.

## Session 7: Vulkan backend extraction

Started the backend split by moving Vulkan-only scaffolding out of the transitional `gpu` package:

- Moved `gpu/vulkan*.go` and `gpu/shaders/` to `backends/vulkan`.
- Changed the package name to `vulkan`, keeping CUDA/PTX files and GPU expert resources in `gpu`.
- Updated README, architecture, GPU options, refactor plan, and Makefile validation targets to include `backends/vulkan`.

## Session 8: Documentation/status audit after backend/runtime moves

Reviewed the public and internal Markdown after the placement, runtime memory, and Vulkan extractions:

- Corrected README/backend docs to avoid over-claiming Vulkan full-forward support; Vulkan is now documented as `backends/vulkan` scaffolding/assets with Phase 3.6 dispatch wiring still pending.
- Updated kernel inventory wording to avoid stale exact F32/BF16 tables and reflect the current CUDA/Vulkan/SIMD ownership split.
- Clarified CPU SIMD coverage as runtime-gated core hot paths with remaining GEMV/RoPE/GELU gaps, rather than claiming complete coverage.
- Clarified native BF16 as scaffolding/helpers where the F32-compatible path is still used as needed.

## Session 9: Runtime/backend hardening audit

Audited the newly split runtime/backend packages for concrete edge cases and stale assumptions:

- Hardened `runtime/memory.MmapAdvisor` so repeated prefetch/evict calls do not skew hot-byte accounting, invalid ranges are ignored safely, cold ranges are not merged into hot ranges, and `madvise` errors propagate to safetensors eager loading.
- Hardened `backends/placement.BudgetManager` and `PlanLayerPlacement` against negative budgets, huge device-memory values, negative model dimensions, and overflow-prone arithmetic.
- Hardened `gpu.ExpertPool` for disabled zero-slot pools, nil entries, replacement behavior, and replacement budget accounting.
- Hardened `runtime/quant` validation: MLX scale/bias tensors must be F32/F16/BF16, GPTQ qweight/g_idx/scales/qzeros are validated before use, and public Q4 GEMV calls validate their slices/dimensions instead of panicking.
- Updated focused regression tests for each fix and kept the fast validation gate green.

## Session 10: Diagnostic test quarantine

Reduced accidental test/compile load from the transitional `model` package:

- Added the `diagnostic` build tag to the Gemma4 trace/sensitivity/generation diagnostic test files under `model/`.
- Kept their existing `GEMMA4_TRACE_TEST=1` runtime guard, so heavy local fixture/GPU diagnostics now require both `-tags diagnostic` and the explicit environment opt-in.
- Updated Makefile/refactor-plan documentation examples to include `-tags diagnostic` for Gemma4 GPU diagnostics.

## Session 11: BF16 scaffolding cleanup

Removed dead model-local BF16 forward-path experiment scaffolding:

- Deleted the unused `model/BF16Hidden` wrapper and `UseBF16` helper, which had no non-self references and was not part of the active CPU/GPU BF16 paths.
- Kept the active BF16 conversion/math helpers in `model/bf16.go` and backend SIMD/CUDA BF16 kernels intact.
- Re-ran the focused model gate, fast package gate, no-test compile sweep, vet, and whitespace checks.

## Session 12: CUDA PTX asset extraction start

Started the CUDA backend split with a low-risk asset-only move:

- Moved pure PTX source definitions from `backends/cuda/ptx/attn.go` and `backends/cuda/ptx/kernels.go` into `backends/cuda/ptx`.
- Updated the CUDA mega-module loader to import those backend-owned PTX assets while keeping runtime dispatch, `DevBuf`, GPU quantized weights, and expert resources in the transitional `gpu` package.
- Left mixed dispatch/source files such as MLX and BF16 PTX in `gpu` for now because they still define CUDA function handles and runtime helpers.

## Session 13: LM head PTX asset extraction

Continued the CUDA PTX asset split:

- Moved the `LMHeadPTX` source string from the GPU dispatch file to `backends/cuda/ptx`.
- Left `gpu.DevLMHead` and its CUDA function handle in `gpu`, preserving runtime behavior while shrinking mixed source/dispatch files.

## Session 14: Q4 PTX asset extraction

Continued separating CUDA source assets from runtime dispatch:

- Moved the optimized Q4 GEMV PTX source to `backends/cuda/ptx`.
- Moved the batched Q4 GEMM PTX source to `backends/cuda/ptx`.
- Kept `gpu.GemmQ4`, `gpu.BatchGEMMReady`, and CUDA function handles in `gpu` because they still own runtime launch semantics.

## Session 15: SGEMM PTX asset extraction

Continued the asset-only CUDA backend split:

- Moved the standalone `SgemmPTX` source string into `backends/cuda/ptx`.
- Kept SGEMM launch/runtime state in `gpu`, matching the current `DevBuf` and mega-module ownership boundaries.

## Session 16: Prefetch PTX asset extraction

Continued CUDA source-asset extraction:

- Moved `PrefetchPTX` into `backends/cuda/ptx`.
- Kept CUDA stream/event/graph helpers and the prefetch function handle in `gpu`, since they are runtime orchestration rather than source assets.

## Session 17: BF16 PTX asset extraction

Continued CUDA source-asset extraction:

- Moved emulated BF16 PTX source strings into `backends/cuda/ptx`.
- Moved native SM86 BF16 PTX source strings into `backends/cuda/ptx`.
- Kept BF16 launch helpers, function handles, and native module loading in `gpu` because those still belong to CUDA runtime orchestration.

## Session 18: MLX PTX asset extraction

Finished the current CUDA PTX source-asset sweep:

- Moved MLX GEMV, batched MLX GEMM, and MLX correction PTX source strings into `backends/cuda/ptx`.
- Kept `GPUMLXWeight`, upload/transposition logic, launch helpers, and function handles in `gpu`, because they still own CUDA resource lifetimes and runtime dispatch.

## Session 19: CUDA helper filename cleanup

Cleaned up stale file naming after the PTX asset extraction:

- Renamed remaining `gpu/*_ptx.go` files to runtime-oriented names because they now contain launch helpers/function handles, not embedded PTX source strings.
- Updated stale comments and refactor-plan references so `gpu` is described as runtime dispatch/resource ownership and `backends/cuda/ptx` as PTX source ownership.

## Session 20: GPU vector-op upload guard audit

Fixed a GPU fast-path guard bug found during the post-PTX-split audit:

- `gpu.DevAdd` and `gpu.DevMul` now include both input buffers in the `tryGPU` preflight.
- Previously the fast path only checked `a` and `out`, then uploaded `b` while ignoring the error; if `b` failed to upload, the kernel argument setup could dereference a nil GPU pointer instead of falling back to CPU.

## Session 21: DevBuf bounds and fallback audit

Hardened GPU runtime helpers against malformed dimensions and failed uploads:

- Added nil-safe GPU preflight and common-length bounding for vector helpers.
- `DevRMSNorm` and `DevRMSNormNoScale` now require successful upload of all kernel operands before launching, instead of ignoring `ToGPU` errors.
- Bounded `DevToBF16`, `DevSoftmax`, `DevGELUTanhMul`, `DevCopy`, and `DevBuf.Slice` to avoid out-of-range slices or overlong GPU launches on malformed inputs.
- Added regression coverage for mismatched buffer lengths and overlong operation lengths.

## Session 22: Q4/MLX CUDA dispatch guard audit

Hardened CUDA quantized dispatch paths found during the GPU runtime audit:

- `UploadQuantWeight` now validates dimensions, packed-weight length, scale layout, and group-index ranges before allocating GPU buffers.
- Q4 GEMV/GEMM launch helpers now reject nil/malformed weights, undersized input/output buffers, and failed buffer uploads before touching CUDA kernel arguments.
- `UploadMLXWeight` now validates dimensions, packed MLX weight length, and scale/bias lengths before transposition/upload.
- MLX GEMV/GEMM launch helpers now preflight native/GPTQ weight availability and input/output dimensions before dispatch.
- Low-level CUDA `Buffer.Upload`/`Download` and integer reinterpret helpers now handle empty slices without indexing `data[0]`.

## Session 23: GEMV/LM-head dispatch guard audit

Hardened remaining dense CUDA dispatch helpers:

- `DevGemv`, `DevGemvNN`, and `DevLMHead` now validate nil inputs, dimensions, and backing-buffer lengths before GPU launch or CPU fallback.
- Dense GEMV and LM-head GPU paths now use the same `tryGPU` preflight as vector and norm helpers, avoiding ignored upload/allocation errors.
- Added malformed-call regression coverage for GEMV, pre-transposed GEMV, and LM-head dispatch.

## Session 24: CUDA stream/memcpy guard audit

Hardened stream and device-copy wrappers:

- `PrefetchWeights` now validates quantized weights before touching prefetch kernel arguments and stops if CUDA event setup fails.
- `LaunchKernelOnStream` now rejects nil functions and zero launch dimensions before calling CUDA, and handles zero-argument launches without indexing an empty slice.
- `CopyDtoD` now returns an error, treats zero pointers/zero bytes as no-op, and reports CUDA copy failures instead of silently ignoring them.
- Updated GPU forward call sites to explicitly ignore `CopyDtoD` errors where the existing generation path cannot yet surface them.

## Session 25: GPU pointer call-site audit

Reduced hidden upload/retry hazards around `DevBuf.GPUPtr()` call sites:

- Cached GPU pointers in batched prefill RoPE/KV-copy paths instead of repeatedly calling `GPUPtr()` inside one operation.
- Cached Gemma4 PLI and KV cache GPU pointers in the main GPU forward path before dispatch/copy decisions.
- KV copy paths now require source and destination GPU pointers to be non-nil before calling `CopyDtoD`, avoiding nil-pointer dereferences when lazy upload fails.

## Session 26: Runtime KV staging bounds audit

Hardened staged float KV rollback:

- `FloatKVCheckpoint.KeepAppended` now rejects negative per-layer KV dimensions instead of allowing negative slice targets.
- Added regression coverage for malformed negative `kvDims` input.

## Session 27: TurboQuant cache input audit

Hardened TurboQuant and compressed KV cache helpers:

- Sanitized negative cache dimensions and residual-window settings in constructors.
- `CompressedKVCache` methods now handle nil/zero-dimension caches without division-by-zero or negative-capacity panics.
- TurboQuant bit widths are clamped to the supported 1–8 bit range, and malformed/short packed inputs dequantize to bounded zero-filled vectors instead of indexing past the input.
- Added regression coverage for malformed compressed-cache and TurboQuant inputs.

## Session 28: Runtime MLX quant helper validation

Hardened in-memory MLX quant helper use:

- Added `ValidateMLXQuantWeight` for already-loaded MLX affine weights.
- `DequantMLX` now returns nil for malformed weights instead of panicking.
- `GemvMLQ` now no-ops on malformed weights or undersized input/output slices.
- Added regression coverage for malformed and valid in-memory MLX weights.

## Session 29: MmapAdvisor overflow audit

Hardened mmap residency range bounding:

- `MmapAdvisor.boundedRange` now clamps oversized byte counts before page alignment so huge caller ranges cannot overflow alignment arithmetic.
- Added regression coverage for huge range prefetch clamping to the mapped file size.

## Session 30: Safetensors malformed-file audit

Hardened safetensors loader edge cases:

- Header length is checked before converting to `int`, avoiding overflow on malformed files.
- Tensor data offsets are validated at open time and again before raw slicing.
- F32/F16/BF16/I32/I64 conversion paths now reject byte lengths that are not element-aligned instead of silently truncating.
- Sharded raw/I32 lookups now return errors for missing shard objects instead of nil-pointer panics.
- Added small synthetic safetensors regression tests for invalid offsets, misaligned raw lengths, and missing shards.

## Session 31: Tokenizer malformed-input audit

Hardened tokenizer edge cases:

- Loading a tokenizer with missing `model.vocab` no longer panics when added tokens are present.
- Missing/null merge lists are accepted as empty merges.
- `Encode`/`Decode` are nil-safe.
- `Decode` now preserves unknown non-byte-level Unicode runes as UTF-8 instead of truncating them to a single byte.
- Added focused tokenizer regression tests.
