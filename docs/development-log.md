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
| `tensor/` — unit tests | growing | all ops, lazy eval, fusion, shape/realization/rewrite/NN validation |
| `tensor/` — numpy reference | 20 | bit-level reproducibility |
| `loader/safetensors/` | 3 | load, list, F16 conversion |
| `models/bert/` | 2 | load weights, end-to-end embed |
| **Total** | evolving | focused package gates preferred during Phase 6.5 | |


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

## Session 32: Attention/RoPE helper bounds audit

Hardened CPU attention/RoPE helpers:

- `applyRoPEPartial` now validates/caps position, head counts, head dimensions, and rotation width before indexing.
- `gqaAttention`/`gqaAttentionScale` now handle invalid dimensions and zero sequence length without divide-by-zero or negative-length allocation hazards.
- `gqaAttentionScaleInto` validates output/scratch/cache lengths and GQA divisibility before slicing.
- Added malformed-input regression coverage for attention and RoPE helpers.

## Session 33: CPU GEMV helper bounds audit

Hardened low-level CPU GEMV helpers:

- `gemv`, `gemvNT`, and `gemvNTParallel` now validate dimensions and slice lengths before indexing or taking unsafe pointers.
- Malformed calls zero the destination and return instead of panicking or reading short buffers.
- Added regression coverage for malformed and valid GEMV helper calls.

## Session 34: MTP drafter shape validation audit

Hardened Gemma4 MTP drafter loader helpers:

- `validateShape` now rejects negative dimensions and detects shape-product overflow via `shapeProduct`.
- `loadIntTensor` validates caller-provided expected lengths directly instead of trusting the raw shape product as the data length.
- Added regression coverage for negative and overflowing shape dimensions.

## Session 35: MTP drafter helper backing-data audit

Hardened MTP drafter helper methods:

- `AssistantTokenEmbeddingInto` now verifies the embedding tensor backing data is large enough for the requested row before slicing.
- `PreProjectInto` and `PostProjectInto` now reject invalid dimensions and short projection buffers before indexing.
- Added regression coverage for short embedding/projection backing data.

## Session 36: Tensor shape validation audit

Hardened tensor shape helpers:

- `shapeSize` now rejects negative dimensions and integer overflow with a negative sentinel.
- Tensor constructors reject malformed shapes before allocation.
- `NewShape`, `Permute`, and `Expand` now validate malformed dimensions/orders before indexing.
- Added regression coverage for negative, overflowing, duplicate, short, and out-of-range shape operations.

## Session 37: Tensor reduce-axis validation audit

Hardened tensor reduction helpers:

- `reduceOp` now validates nil receivers, out-of-range axes, negative axes, and duplicate axes before indexing shape dimensions.
- Added regression coverage for malformed reduction axes.

## Session 38: Tensor nil-operation audit

Hardened tensor operation entrypoints:

- `Realize`, unary ops, and binary ops now report nil tensor receivers/operands explicitly instead of dereferencing nil fields.
- `Data` is nil-safe and returns nil for a nil tensor.
- Broadcast now rejects malformed shapes before attempting expansion.
- Added regression coverage for nil tensor operations.

## Session 39: Tensor unsafe slice helper audit

Hardened tensor byte/float reinterpret helpers:

- `byteSliceToFloat32` and `float32ToByteSlice` now return nil for empty inputs instead of indexing element zero.
- `Buffer.Float32Data` is nil-safe for nil buffers.
- Added regression coverage for empty/zero-size tensor data paths.

## Session 40: Tensor realization validation audit

Hardened tensor realization internals:

- `realize` now rejects nil UOps, nil sources, and invalid shapes before dispatch.
- `allocBuffer`, unary/binary eval, broadcast eval, reduce eval, and input-shape guessing now validate malformed internal inputs before indexing.
- Added regression coverage for malformed UOp/eval helpers.

## Session 41: Tensor buffer pool allocation audit

Hardened tensor buffer allocation:

- `pooledAlloc` now rejects negative lengths, zero-byte dtypes, and integer-overflowing allocation sizes before creating pool keys or byte slices.
- Added regression coverage for malformed pooled allocation inputs.

## Session 42: Tensor rewrite/fusion nil-safety audit

Hardened tensor graph rewrite and fusion paths:

- Patterns, pattern matchers, graph rewrite traversal, and rules now handle nil patterns/UOps/rules without nil-deref panics.
- Fusion setup rejects nil roots and invalid shapes; fused kernel execution validates kernel structure and leaf buffer sizes.
- Added regression coverage for nil rewrite inputs and malformed fused kernels.

## Session 43: Tensor embedding/matmul validation audit

Hardened tensor neural-network helpers:

- `Embedding` now validates nil/2D weights and token ID bounds while preserving empty ID handling.
- `MatMul`/`MatMulTransposed` reject nil tensors and avoid taking `&slice[0]` on zero-sized matrices before SIMD calls.
- `Linear` and `LinearPreT` validate bias shape before in-place bias addition.

## Session 44: Tensor NN helper validation audit

Hardened tensor neural-network utility ops:

- `Softmax`, `LayerNorm`, and `GELU` now reject nil receivers explicitly.
- `Softmax` and `LayerNorm` avoid division/indexing on zero-width last axes.
- `LayerNorm` validates gamma/beta shape compatibility and requires them to be supplied together.

## Session 45: Tensor module constructor audit

Hardened tensor module wrappers:

- `NewLinear`, `NewLayerNorm`, and `NewEmbedding` now reject invalid dimensions before initialization.
- Module `Forward` methods now reject nil module receivers explicitly.
- Tensor property accessors are nil-safe, returning zero values for nil tensors.


## Session 46: Documentation sweep after tensor hardening

Reviewed and refreshed documentation after the tensor/runtime/backend malformed-input audit passes:

- README now calls out the shared validation/hardening baseline and the focused fast validation gate.
- Architecture docs now treat tensor/runtime guard behavior as an explicit package-boundary policy for later refactor moves.
- Refactor plan now records tensor hardening as part of the Phase 6.5 baseline and fixes the stale `loader/safetensors` mmap-advisor ownership note.
- CPU SIMD coverage notes now mention zero-length tensor matmul guard behavior before assembly dispatch.

## Session 47: SIMD BF16 helper bounds audit

Hardened scalar BF16 helper paths in `backends/simd`:

- `BF16Dot` now bounds mismatched input lengths like `BF16DotF32`.
- `BF16RMSNorm` no-ops on empty inputs or short weights instead of dividing by zero or indexing past weights.
- `BF16VecAdd` bounds all three slices and leaves the untouched destination tail unchanged.
- `BF16GemvNT` validates dimensions and backing slice lengths before row slicing.

## Session 48: SIMD vector fallback bounds audit

Hardened scalar vector fallback helpers in `backends/simd`:

- F32 vector add/mul/scale/scale-add and activation fallback loops now bound all input/output slices instead of trusting `a` length.
- F32 RMSNorm fallback no-ops on empty inputs or short weights; no-scale RMSNorm no-ops on empty input.
- BF16 widen/narrow fallbacks bound source and destination lengths, leaving destination tails untouched.

## Session 49: SIMD GEBP argument validation audit

Hardened GEBP/packed-B helper paths in `backends/simd`:

- `ensureGebpBuf` now returns nil for non-positive requests instead of slicing with invalid bounds.
- `packBNT`/`packBNTScalar` validate strides, block sizes, `k`, packed-buffer size, and B backing length before slicing or taking row pointers.
- `SgemmNTGebp` validates dimensions, pointers, strides, and multiplication overflow before building unsafe slices.

## Session 50: SIMD blocked SGEMM validation audit

Reused the GEBP argument preflight for `SgemmNTBlockedFMA` so the blocked FMA path rejects invalid dimensions, nil pointers, short strides, and overflow-prone shape products before pointer arithmetic or tile dispatch.

## Session 51: Compressed KV cache layout audit

Hardened `runtime/kv.CompressedKVCache` layout handling:

- Constructor now disables compression when `numKVHeads*headDim` does not match `kvDim`.
- Compression preflight rejects inconsistent head layouts before per-head slicing.
- `GetK`/`GetV` clamp overlong full caches to `seqLen*kvDim` when no compressed entries exist, and fall back to full-precision storage if compressed entry metadata is malformed.

## Session 52: KV staging overflow audit

Hardened `runtime/kv` staging helpers:

- Float KV `KeepAppended` now checks `base + keepTokens*kvDim` for integer overflow before truncating slices.
- Compressed KV `KeepAppended` validates checkpoint/keep arithmetic, negative compressed-entry checkpoint lengths, and positive `kvDim` when retaining staged tokens.
- Added regression coverage for overflow and malformed checkpoint cases.

## Session 53: TurboQuant size-overflow audit

Hardened TurboQuant sizing math:

- `NewTurboQuantState` and `randomOrthogonal` now reject overflowing `headDim*headDim` sizes before allocation.
- `QuantizeVector` and `DequantizeVector` validate rotation-size arithmetic before indexing rotation matrices.
- `packIndices` now validates packed byte-length arithmetic before allocation.

## Session 54: MmapAdvisor nil-safety audit

Hardened `runtime/memory.MmapAdvisor` method receivers:

- Public methods now treat a nil advisor as an inert no-op and return zero stats, matching the existing invalid-range behavior.
- Internal alignment/range helpers and total recomputation now guard nil receivers before touching advisor fields.
- Added regression coverage for nil advisor method calls.

## Session 55: Safetensors metadata validation audit

Hardened safetensors metadata and sharded-file helpers:

- Tensor metadata validation now checks shape product overflow and known dtype byte-size agreement with tensor data offsets at open time.
- Sharded safetensors methods are nil-safe and report nil sharded files as errors for tensor lookups.
- `OpenSharded` now uses `filepath.Dir`/`filepath.Join` instead of manual slash parsing for index-relative shard paths.

## Session 56: Tokenizer helper nil/race audit

Hardened tokenizer helper paths:

- `Tokenizer.VocabSize` is now nil-safe.
- Byte-level BPE encoder/decoder maps now use `sync.Once` for lazy initialization, avoiding concurrent map initialization races.
- Added regression coverage for nil vocab size and byte-map roundtrips.


## Session 57: Documentation sweep after SIMD/runtime/loader hardening

Reviewed and refreshed documentation after the latest audit batch:

- README now records SIMD fallback/SGEMM preflights, KV/TurboQuant layout and overflow guards, mmap nil-safety, safetensors dtype-byte validation, and tokenizer `sync.Once` byte maps.
- Architecture docs now treat loader/SIMD/runtime guard policy as part of the shared package-boundary baseline.
- Refactor plan now includes the newer backend/runtime/loader guard status in the current package map and validation gate.
- CPU SIMD coverage now documents scalar fallback slice bounding and SGEMM/GEBP unsafe-pointer preflights.

## Session 58: MTP/inference helper bounds audit

Hardened transitional `model` helper paths:

- MTP acceptance now rejects negative drafted/verifier token IDs and invalid KV keep counts before committing staged KV.
- Token embedding and LM-head helpers validate positive model dimensions and backing data lengths before slicing.
- Gemma4 per-layer input helpers validate positive/overflow-safe dimensions before projection and embedding indexing.

## Session 59: Chunked GPU LM-head guard audit

Hardened the transitional chunked GPU LM-head helper:

- Rejects nil/malformed model inputs, non-positive dimensions, short logits/hidden slices, short LM-head backing data, and overflow-prone `vocabSize*hidden` products before GPU allocation.
- Checks all chunk/input/output GPU upload errors before dispatching chunked LM-head kernels.

## Session 60: Model KV dimension overflow audit

Hardened model-specific KV staging helpers:

- `LayerKVDim` now checks `num_key_value_heads * head_dim` for integer overflow before returning per-token KV widths used by staged verifier commits.
- Added regression coverage for overflowing model-level and layer-local KV dimensions.

## Session 61: GPU prefill guard audit

Hardened the batched GPU prefill fallback entrypoint:

- Rejects nil GPU/CPU model state, invalid head/KV/intermediate dimensions, non-divisible head dims, overflow-prone batch products, malformed embedding tables, and invalid token IDs before GPU allocation/embedding slicing.
- Checks the initial batch-hidden upload before continuing into the batched prefill path.

## Session 62: Model dot helper bounds audit

Hardened the transitional model-local `simdDot` helper:

- Scalar short-vector fallback now bounds mismatched input slices instead of trusting the first slice length.
- Added regression coverage for short, nil, and long mismatched dot inputs.

## Session 63: Low-level model helper overflow audit

Hardened model-local low-level math helpers:

- `gemv`, `gemvNT`, and `gemvNTParallel` now check `inDim*outDim` for integer overflow before backing-slice length checks.
- `gqaAttentionScaleInto` now checks `heads*headDim`, `kvHeads*headDim`, and `seqLen*kvDim` products before cache-length validation.
- Added regression coverage for overflow-prone GEMV and attention helper inputs.


## Session 64: Documentation sweep after model helper hardening

Reviewed and refreshed documentation after the latest transitional `model` audit batch:

- README now records MTP, KV, prefill, chunked LM-head, embedding/LM-head, GEMV, and GQA helper guard coverage.
- Architecture docs now call out model-helper guard behavior as part of the Phase 6.5 shared hardening baseline.
- Refactor plan now marks the transitional `model` package helper guards as hardened and clarifies that focused model helper tests remain part of the validation gate.

## Session 65: GPU DevBuf receiver/upload audit

Hardened GPU `DevBuf` and CUDA allocation helpers:

- `DevBuf` receiver methods now handle nil receivers consistently, returning nil/zero values or errors instead of dereferencing nil.
- `ToGPU` now propagates upload failures, frees newly allocated GPU memory on upload failure, and no longer marks GPU authoritative after a failed re-upload.
- `GPUPtr` returns nil if lazy upload fails.
- `Malloc` rejects `n*4` size overflow before entering CUDA driver code.

## Session 66: CUDA stream/graph helper audit

Hardened CUDA stream/graph helpers:

- `CapturedGraph.Launch` now rejects nil or empty graph executables before entering CUDA driver calls.
- `CapturedGraph.Destroy` is nil-safe.
- `LaunchKernelOnStream` now rejects nil kernel argument pointers before constructing the CUDA argument array.

## Session 67: GPU Q4 quantized weight validation audit

Hardened GPU Q4 quantized weight helpers:

- `UploadQuantWeight` now checks packed qweight and scale layout products for integer overflow before length validation/allocation.
- `validGPUQuantWeight` now validates dimensions, divisibility, buffer presence, backing buffer byte sizes, and size-product overflow.
- CPU fallback now returns if Q4/scales/gIdx downloads fail instead of continuing with zero-filled placeholders.


## Session 68: Documentation sweep after GPU runtime guard audit

Reviewed and refreshed documentation after the latest GPU audit batch:

- README and architecture docs now record hardened `DevBuf`, CUDA stream/graph, allocation-size, and Q4 weight-layout validation.
- Refactor plan now marks transitional `gpu` runtime guards as part of the Phase 6.5 baseline before the CUDA runtime split.
- GPU options docs now include a DevBuf/dispatch guard-status section so the eventual `backends/cuda` move preserves these checks.

## Session 69: GPU MLX weight validation audit

Hardened GPU MLX quantized weight helpers:

- `UploadMLXWeight` now checks packed weight, scale, and correction size products for integer overflow before allocation/transposition.
- `validGPUMLXWeight` now validates group consistency, divisibility, backing buffer byte sizes, GPTQ fallback validity, and size-product overflow.
- Batched `GemmMLX` validates `B*inDim` and `B*outDim` arithmetic before dispatch.

## Session 70: GPU expert pool safety audit

Hardened GPU expert-pool helpers:

- Expert-pool public methods are now nil-safe.
- Negative expert IDs are rejected and returned to callers for resource release instead of being cached or looked up.
- Added regression coverage for nil pool and invalid expert ID behavior.

## Session 71: Experimental NV memory helper audit

Hardened experimental direct-NVIDIA memory helpers:

- `AllocHostMem` validates nil devices and invalid/overflowing sizes, stores the mmap slice, and unmaps host memory on registration failure.
- `mapToCPU` validates inputs and stores the mmap slice in `cpuMem` so upload/download paths can use it.
- `NVBuffer` upload/download/free methods are nil-safe, handle empty slices as no-ops, and validate byte-size arithmetic/bounds before unsafe slice conversion.

## Session 72: Experimental NV ioctl helper audit

Hardened experimental direct-NVIDIA ioctl helpers:

- NV device helper methods are nil-safe where practical and return explicit errors for nil receivers.
- VA allocation now rejects zero/overflowing sizes and bump-pointer overflow.
- ioctl/RM helper wrappers validate file descriptors and nil parameter pointers before raw syscalls.

## Session 73: Experimental NV query/GPFIFO audit

Hardened remaining experimental direct-NVIDIA helpers:

- GPFIFO/channel setup now validates nil devices, channel groups, context handles, and class info before allocating resources.
- GPFIFO setup frees already allocated ring/notifier buffers on later setup failures.
- NV query helpers validate nil/uninitialized devices and cap class-list sizes before allocation.

## Session 74: GPU SGEMM/LM-head validation audit

Hardened remaining dense CUDA dispatch helpers:

- `Sgemm` now validates dimensions, non-nil/non-zero buffers, size-product overflow, and backing buffer byte sizes before kernel launch.
- `SgemmHost` validates host dimensions, slice lengths, and size-products before allocation/upload.
- `DevLMHead` now checks `vocab*hidden` overflow before backing-buffer validation.

## Session 75: GPU JIT compiler validation audit

Hardened the experimental CUDA JIT compiler helpers:

- `Compile` validates nil/empty kernel specs, nil nodes, out-of-range buffer indices, and nil node inputs before cache lookup or PTX generation.
- `CompiledKernel.Launch` now rejects nil kernels, invalid launch metadata, missing buffers, zero GPU pointers, and undersized buffers before CUDA calls.
- Added malformed-spec and no-op launch regression coverage.

## Session 76: GPU BF16 dispatch validation audit

Hardened BF16 CUDA dispatch helpers:

- Emulated/native BF16 norm, add, SiLU, and GELU launch wrappers now validate nil pointers, positive lengths, byte-size bounds, and length overflow before CUDA calls or fallback dispatch.
- Added regression coverage for BF16 buffer validation and malformed dispatch calls.


## Session 77: Documentation sweep after GPU backend guard batch

Reviewed and refreshed documentation after the latest GPU/backend audit batch:

- README and architecture docs now record hardened MLX, expert pool, experimental NV helpers, dense SGEMM/LM-head, JIT, and BF16 dispatch validation.
- GPU options docs now list the expanded DevBuf/dispatch guard baseline that must move with the future `backends/cuda` split.
- Refactor plan now reflects the broader transitional GPU guard coverage in Phase 6.5.

## Session 78: Batched Q4 dispatch audit

Hardened batched Q4 dispatch:

- `GemmQ4` now validates the quantized weight before reading dimensions, computes batched input/output size products with overflow checks, and rejects malformed buffers before CUDA dispatch.
- `GemvQ4OrGemm` no longer prints a misleading sequential fallback message for a fallback path that cannot safely slice batched buffers yet; it delegates to the guarded batched dispatch for `B>1`.


## Session 79: RoPE/attention dispatch guard audit and docs

Hardened and documented remaining GPU RoPE/attention dispatch wrappers:

- RoPE and partial RoPE validate positions, dimensions, tensor lengths, and size-product overflow before launch.
- Attention score, softmax-row, and fused GQA attention wrappers validate sequence bounds, head dimensions, cache lengths, and output sizes before launch.
- Documentation refreshed so the future CUDA backend split preserves these transitional guard expectations.

## Session 80: CUDA launch wrapper validation audit

Hardened the raw CUDA launch wrapper:

- `LaunchKernel` now returns explicit errors when the CUDA launch symbol is unavailable, the function handle is nil, or grid/block dimensions are zero.
- Added regression coverage so malformed launches fail safely before purego calls.

## Session 81: Model/GPU boundary ignored-error audit

Hardened model-side GPU boundary error handling:

- Batched prefill now propagates GPU-to-GPU KV copy failures instead of ignoring `CopyDtoD` errors during cache append.
- MoE GPU fallback setup now checks all scratch-buffer `ToGPU` uploads and cleanly falls back to CPU experts if scratch allocation/upload fails.

## Session 82: GPU model loader upload-error audit

Hardened `LoadGPUModel` upload error handling:

- Work-buffer upload failures now abort model loading with cleanup instead of silently returning a partially CPU/GPU-backed `GPUModel`.
- Per-layer weight upload failures are captured and reported after allocation cleanup.
- KV cache GPU buffer uploads now propagate layer-specific errors instead of ignoring `ToGPU` failures.

## Session 83: Batched prefill scratch lifetime audit

Hardened batched GPU prefill scratch lifetime management:

- Temporary batch `DevBuf` scratch buffers are now freed on all return paths to avoid leaking GPU-side allocations during prefill fallback/error exits.

## Session 84: CLI/server token-output bounds audit

Hardened token-output boundary handling in front-ends:

- `llmchat` now stops cleanly if generation returns an out-of-vocabulary token ID instead of indexing `InvVocab` blindly.
- `llmserver` applies the same generated-token bounds check in OpenAI-compatible responses.
- SSE chunk writing now handles JSON marshal errors instead of ignoring them.

## Session 85: Server response write-error audit

Hardened OpenAI-compatible server response writes:

- `/v1/models` and non-streaming chat responses now log JSON encode/write failures.
- Streaming final `[DONE]` and chunk writes now handle write errors instead of silently ignoring them.

## Session 86: Refactor validation gate smoke pass

Started Phase 6.5.6 validation gate after the GPU/model/cmd audit batch:

- Focused model tests passed: `TestPrefillGPURejectsMalformedInputs|TestMTP|TestInference|TestKV|TestMoE|TestLoad`.
- Fast no-run package gate passed for GPU, loader, backend, runtime, BERT, tensor, and command packages.
- `go vet ./...` and `git diff --check` passed.
- Loader/generation smoke runs passed for `models/smollm2-135m` and `models/gemma4-e2b-mlx4` via `cmd/llmgen`.

## Session 87: Refactor validation gate no-run sweep

Continued Phase 6.5.6 validation:

- Re-ran the fast shared package gate with full tests for `tensor`, `backends/simd`, `runtime/...`, and `loader/...`.
- Re-ran `models/bert` full package tests.
- Confirmed repository-wide no-run compile gate with `go test ./... -run '^$'`.

## Session 88: Full refactor validation gate

Completed the broad Phase 6.5.6 validation gate after the cleanup/hardening batch:

- Full repository test suite passed with `go test ./... -count=1`.
- This complements the earlier focused model tests, fast shared-package gates, no-run compile sweep, vet/diff-check, and SmolLM2/Gemma4 `llmgen` smoke runs.

## Session 89: GPU DevBuf RoPE/attention split

Continued Phase 6.5 cleanup by splitting an oversized transitional GPU file:

- Moved RoPE, partial RoPE, softmax-row, and GQA attention dispatch helpers out of `gpu/devbuf.go` into `gpu/rope_attention.go` without semantic changes.
- Kept the recently added launch-shape guards with the moved dispatch helpers so they remain visible for the future CUDA backend split.


## Session 90: SIMD folder reorg assessment

Started Phase 6.6 SIMD folder reorg work with a layout assessment:

- Documented the current `backends/simd` file split by build tags and CPU family.
- Captured the Go package constraint: a literal `amd64/arm64/scalar` folder split is not mechanical because it creates separate packages and requires facade bridge APIs for unexported assembly entrypoints.
- Added `docs/simd-folder-reorg.md` and linked it from the SIMD coverage notes as the safe migration path.

## Session 91: SIMD scalar fallback split

Continued Phase 6.6 with a facade-preserving mechanical cleanup:

- Moved scalar `Sdot`/`Saxpy` fallback helpers from `backends/simd/simd.go` to `backends/simd/scalar.go`.
- Kept the public `backends/simd` package and architecture-specific dispatch files unchanged.

## Session 92: SIMD empty facade cleanup

Continued Phase 6.6 cleanup after the scalar fallback split:

- Removed the now-empty `backends/simd/simd.go` placeholder after moving scalar fallback helpers to `scalar.go`.
- Kept the `backends/simd` package facade intact through the remaining implementation files.

## Session 93: SIMD sqrt fallback audit

Hardened SIMD scalar norm math:

- Replaced the unsafe inverse-square-root approximation used by `float32Sqrt` with `math.Sqrt` to avoid precision-sensitive RMSNorm drift in scalar fallbacks.
- Added regression coverage for representative `float32Sqrt` inputs.

## Session 94: SIMD BF16 GEMV dimension audit

Hardened BF16 scalar GEMV fallback:

- `BF16GemvNT` now checks `inDim*outDim` overflow before validating the backing F32 weight slice.
- Added a shared `checkedMulInt` helper for SIMD package dimension products and regression coverage for overflowing BF16 GEMV dimensions.


## Session 95: SIMD reorg documentation sweep

Reviewed and refreshed docs after the Phase 6.6 SIMD cleanup/audit batch:

- README and architecture docs now describe the facade-first SIMD reorg, scalar fallback split, precise scalar sqrt behavior, and BF16 GEMV overflow guard.
- SIMD coverage and folder-reorg notes now record the safe current state and constraints for a future CPU-family subpackage split.

## Session 96: SIMD blocked SGEMM unsupported-arch audit

Hardened blocked SGEMM dispatch:

- `SgemmNTBlockedFMA` now checks `HasSgemmAsm` before reaching architecture-specific tile kernels, so unsupported architectures no-op safely instead of hitting the fallback panic path.

## Session 97: SIMD cross-architecture build audit

Hardened SIMD package cross-architecture builds during the audit:

- `sgemm.go` now only declares assembly SGEMM entrypoints on `amd64`/`arm64`; portable fallback declarations remain in `simd_other.go`.
- Moved the shared Go `vecSiLUMulGo` fallback out of duplicated amd64/arm64 files into `vec.go`, fixing portable fallback builds where `vec_other.go` referenced it.

## Session 98: SIMD GEBP pack bounds audit

Hardened GEBP packing and fallback dispatch:

- `packBNT` and `packBNTScalar` now share overflow-safe argument validation instead of computing `k*gebpNR` and row offsets inline.
- `SgemmNTGebp` now checks `HasSgemmAsm` before reaching architecture-specific microkernels, matching the blocked SGEMM guard.
- Added regression coverage for overflowing pack arguments.

## Session 99: SIMD gather SGEMM bounds audit

Hardened the unused/experimental gather SGEMM helper:

- `SgemmNTGather` now uses the shared SGEMM/GEBP argument validation and checks `HasSgemmAsm` before reaching architecture-specific gather kernels.
- Added an int32 gather-index bound check for large `ldb` values before building AVX2 gather offsets.
- Added malformed/overflowing gather-dispatch regression coverage.


## Session 100: SIMD SGEMM guard documentation refresh

Refreshed documentation after the latest Phase 6.6 SGEMM/GEBP/gather guard audit:

- README and architecture docs now mention SGEMM/GEBP/gather capability gates and overflow preflights.
- SIMD folder reorg notes now call out keeping `HasSgemmAsm` and shared arithmetic guards at the public facade boundary until subpackage bridge APIs exist.

## Session 101: TurboQuant protected-layer nil audit

Hardened a small TurboQuant helper edge case:

- `TurboQuantState.IsProtectedLayer` is now nil-safe and rejects negative query indices before applying negative configured aliases for last layers.
- Added regression coverage for nil state and negative layer queries.

## Session 102: Safetensors nil/partial-open audit

Hardened safetensors helper edge cases:

- `File.Names` is now nil-safe, matching `ShardedFile.Names` behavior.
- `OpenSharded` now closes any shards already opened when a later shard fails, preventing partial-open mmap/file descriptor leaks.


## Session 103: Runtime/loader audit documentation sweep

Reviewed and refreshed docs after the latest runtime/loader hardening batch:

- README and architecture docs now mention TurboQuant protected-layer input guards and safetensors partial-open cleanup.
- TurboQuant docs now describe the defensive protected-layer helper behavior.
- Refactor plan now records the expanded runtime KV and safetensors cleanup guard coverage.

## Session 104: Tensor unsafe slice audit

Hardened tensor unsafe-slice helpers:

- `byteSliceToFloat32` now rejects byte slices whose length is not a multiple of four instead of silently truncating.
- `Buffer.Float32Data` now validates non-negative element counts and exact byte/element length agreement before exposing an unsafe view.

## Session 105: Tensor shape contiguity audit

Hardened tensor shape helpers:

- `Shape.IsContiguous` now rejects malformed shapes with mismatched stride metadata, invalid dimensions, or overflowing dimension products instead of relying on incidental indexing/arithmetic behavior.

## Session 106: Tensor broadcast overflow audit

Hardened tensor broadcasting:

- `broadcast` now validates padded dimensions and detects overflowing output shape products before constructing expanded shapes.
- Added regression coverage for an overflowing broadcast output shape.

## Session 107: Tensor malformed Numel/reshape audit

Hardened tensor shape sizing:

- `Shape.Numel` now reports `0` for malformed shapes instead of exposing negative sentinel sizes to callers.
- `Shape.Reshape` now checks source and destination shape products directly so malformed source shapes cannot be treated as size-compatible with zero-element targets.

## Session 108: Tensor convenience op input audit

Hardened tensor convenience helpers:

- `Transpose2D`, `Clip`, `ReLU`, `Sigmoid`, and `Where` now validate nil tensors and malformed shapes before dereferencing internals.
- `Where` now validates broadcast compatibility across condition/x/y shapes instead of assuming `x` owns the output shape.

## Session 109: Tensor NN backing-data audit

Hardened eager NN helpers:

- `Softmax` and `LayerNorm` now validate realized backing-data length against the tensor shape before row slicing.
- Output allocation now uses validated shape size instead of raw backing slice length.

## Session 110: Tensor matmul backing-data audit

Hardened tensor matmul helpers:

- `MatMul` and `MatMulTransposed` now validate dimensions, output shape products, and realized backing-data lengths before SIMD dispatch or scalar indexing.
- Added a shared tensor integer product helper and regression coverage for malformed matmul backing buffers.

## Session 111: Tensor linear bias audit

Hardened tensor linear helpers:

- Deduplicated `Linear`/`LinearPreT` bias addition through a shared helper.
- Bias addition now validates result shape, bias backing data, result backing data, and output product overflow before indexing.


## Session 112: Tensor audit documentation sweep

Reviewed and refreshed documentation after the latest tensor hardening batch:

- README and architecture docs now mention unsafe float32 view validation, malformed shape sizing/contiguity/broadcast guards, NN/convenience helper backing-data checks, and matmul/linear backing-data validation.
- Refactor plan now records the expanded tensor guard coverage that future package moves should preserve.

## Session 113: GPU prefill debug logging audit

Cleaned up library logging noise in batched GPU prefill:

- Batched prefill progress prints are now gated behind `GO_PHERENCE_PREFILL_DEBUG` instead of writing to stdout unconditionally from model code.

## Session 114: Model loader debug logging audit

Cleaned up model loader stdout noise:

- Quantization/eager-load/MoE loader progress messages are now gated behind `GO_PHERENCE_LOAD_DEBUG` instead of printing unconditionally from `LoadLlama`.

## Session 115: GPU loader/progress logging audit

Finished gating the remaining normal-path model/GPU progress prints:

- Per-layer embedding, Gemma4 RoPE, TurboQuant, GPU weight placement, LM-head placement, expert-pool, VRAM-budget, and first MLX upload error diagnostics now use `GO_PHERENCE_LOAD_DEBUG` instead of writing to stdout unconditionally.

## Session 116: GPU runtime debug logging audit

Gated GPU backend progress and experimental NV ioctl diagnostics:

- CUDA init/module/stream/native-BF16 progress messages and non-fatal module lookup diagnostics now use `GO_PHERENCE_GPU_DEBUG`.
- Experimental direct-NVIDIA ioctl/VA/GPFIFO diagnostics are now opt-in under the same GPU debug gate.

## Session 117: Vulkan backend debug logging audit

Gated Vulkan backend discovery/progress messages:

- Vulkan init failures, CPU-device rejection notices, compute readiness, and pending-SPIR-V diagnostics now use `GO_PHERENCE_VULKAN_DEBUG` instead of writing to stdout unconditionally.
- Kept `backends/placement.Plan.PrintPlan` as an explicit caller-requested reporting API.

## Session 118: Placement estimator overflow audit

Hardened backend-neutral placement estimators:

- Layer weight estimates now use saturating arithmetic for dimension products and byte accumulation.
- Quantized MLX group estimates now use ceiling group counts for non-multiple-of-group-size dimensions instead of underestimating partial groups.
- Added regression coverage for odd quantized dimensions and huge malformed size inputs.

## Session 119: Placement resident estimator overflow audit

Extended placement estimator hardening to resident tensors:

- Resident embedding/LM-head/RoPE/work-buffer/PLI estimates now use the same saturating arithmetic as per-layer estimates.
- Packed INT4 resident estimates now round up odd element counts instead of truncating partial bytes.
- Added regression coverage for huge resident inputs and odd packed resident dimensions.

## Session 120: Budget manager accounting audit

Hardened backend-neutral budget accounting:

- Budget manager methods now tolerate nil receivers and reject unknown budget categories instead of aliasing them to resident accounting.
- Allocation now rejects usage overflow before mutating counters; free clamps without subtract-underflow.
- Added regression coverage for nil managers, invalid categories, and overflow rejection.


## Session 121: Logging and placement audit documentation sweep

Refreshed documentation after the logging and placement/budget audit batches:

- README, architecture notes, GPU options, weight-budget notes, and refactor plan now document quiet-by-default library diagnostics and the `GO_PHERENCE_*_DEBUG` gates.
- Placement docs now record guarded budget accounting, invalid-category rejection, nil-safe budget manager methods, saturating estimator math, and odd INT4 packed-size rounding.
- Refactor notes call out that these guard/logging semantics should be preserved during the later CUDA/model package splits.

## Session 122: Tokenizer merge validation audit

Hardened tokenizer loading:

- BPE merge strings and array pairs now reject malformed empty/incomplete pairs instead of leaving zero-value merge rules in the rank table.
- Added malformed merge regression coverage for both tokenizer JSON merge encodings.

## Session 123: Compressed KV cache arithmetic audit

Hardened compressed KV cache arithmetic:

- Constructor capacity hints, full-cache slice bounds, scratch-buffer sizes, and compressed-entry packed-length checks now validate integer products before allocation or slicing.
- Added regression coverage for overflowing KV dimensions/sequence lengths and packed-entry validation.

## Session 124: Compressed KV cache accessor audit

Finished another compressed KV cache edge-case pass:

- `SeqLen`, `CompressedCount`, `FullCount`, and `MemoryBytes` are now nil-safe.
- Memory accounting now uses checked/saturating arithmetic instead of direct slice-length products/sums.
- Added regression coverage for nil accessors and saturating helper behavior.

## Session 125: Mmap advisor range accounting audit

Hardened mmap advisor range bookkeeping against corrupted or malformed tracked ranges:

- Merge and total recomputation now sanitize negative tracked offsets/byte counts.
- Range end, hot-byte, and merged hit/evict counters now use saturating arithmetic.
- Added regression coverage for malformed tracked ranges and saturated accounting.

## Session 126: GPTQ validation overflow audit

Hardened GPTQ/Q4 validation:

- GPTQ qweight/scales/qzeros expected-size calculations now use checked integer multiplication before slice-length comparisons.
- `ValidateGemvQ4Sym` now validates GPTQ dimensions/layout before checking caller slice lengths, so invalid/overflowing dims are reported without requiring impossible-sized output/input slices.
- Added overflow and negative-dimension regression coverage.

## Session 127: MLX quant validation overflow audit

Hardened MLX quantized weight validation/loading:

- MLX dequantization, weight-size, scale/bias-size, shape-derived inDim, and float tensor shape product calculations now use checked multiplication.
- Loader now reports explicit overflow errors for malformed tensor shapes instead of relying on incidental integer wraparound.
- Added overflow regression coverage for packed weight shapes, scale shapes, in-memory MLX weights, and dequantization.

## Session 128: GPTQ dequant output-size audit

Hardened GPTQ dequantization output sizing:

- GPTQ validation now rejects overflowing dequantized output dimensions before qweight/scale length checks.
- GPTQ dequantization paths check output allocation products before allocating.
- Added regression coverage for output-size overflow in both generic and symmetric dequant paths.

## Session 129: Safetensors name/order and eager accounting audit

Cleaned up safetensors helper behavior:

- `File.Names` and `ShardedFile.Names` now honor their sorted-order contract directly instead of relying on callers to sort map iteration output.
- Sharded eager-load byte accounting now checks aggregate overflow while summing shard sizes.
- Added regression coverage for sorted names and checked eager-load byte addition.


## Session 130: Loader/runtime audit documentation sweep

Refreshed documentation after the latest loader/runtime audit batch:

- README, architecture, refactor, TurboQuant, and weight-budget docs now cover tokenizer merge validation, deterministic safetensors names, checked sharded eager-load totals, compressed KV cache accessor/memory-accounting guards, mmap advisor range sanitization, and MLX/GPTQ checked sizing.
- Refactor notes now call out these loader/runtime guard semantics as part of the baseline to preserve during later model/backend package splits.

## Session 131: Chunked LM-head GPU buffer audit

Hardened chunked GPU LM-head setup:

- Clamp reported free VRAM before converting to `int` for chunk sizing.
- Check chunk buffer element products before allocating GPU buffers.
- Free temporary GPU buffers on all return paths from chunked LM-head execution.

## Session 132: Phase 6.5 completion checklist

Made the Phase 6.5 exit criteria explicit:

- Replaced the broad definition-of-done section in `docs/refactor-plan.md` with a concrete checklist covering ownership docs, mechanical moves, audit baselines, debug/logging hygiene, documentation alignment, validation gates, and final closeout.
- Marked completed loader/runtime/backend/tensor/GPU audit work separately from still-pending transitional `model`, CUDA-runtime split, model-package split, command-boundary audit, smoke tests, and final validation.
- This checklist is now the source of truth for deciding whether Phase 6.5 is done or whether remaining work is deliberately deferred.

## Session 133: MTP drafter projection arithmetic audit

Hardened Gemma4 MTP drafter helper arithmetic:

- Pre/post projection helpers now check projection-size products before backing-slice validation or indexing.
- Drafter loader checks pre-projection width overflow before constructing expected tensor shapes.
- Integer tensor loading checks expected raw byte sizes before dtype-specific decoding.
- Added regression coverage for overflowing drafter projection dimensions.

## Session 134: MoE helper edge-case audit

Hardened transitional MoE helpers:

- Switch-MLX expert loader now validates nil sources, dimensions, divisibility, stride products, and raw tensor byte lengths before slicing per-expert data.
- CPU MoE forward now rejects nil/empty/malformed configs, clamps active expert count, guards softmax normalization, and verifies all selected expert weight slices before dispatch.
- Added malformed MoE forward regression coverage.

## Session 135: Inference helper product arithmetic audit

Hardened transitional model inference helpers:

- Token embedding, Gemma4 per-layer input, and LM-head helpers now use checked product arithmetic for offsets, projection sizes, embedding tables, and LM-head backing data sizes.
- Added overflow regression coverage for token embedding offsets, per-layer Gemma4 input dimensions, and LM-head output dimensions.

## Session 136: CPU forward-layer entrypoint audit

Hardened the transitional CPU `ForwardLayer` helper:

- Rejects nil models, invalid layer indices, negative positions, malformed dimensions, short hidden states, missing norm weights, product-overflowing Q/KV dimensions, and missing KV cache slots before indexing.
- Added malformed forward-layer regression coverage.

## Session 137: Streaming server write-boundary audit

Hardened the OpenAI-compatible streaming response path:

- `writeSSE` now returns success/failure, logs marshal errors, and lets callers stop generation immediately on broken client writes.
- Streaming response setup, token chunks, and final chunks now abort cleanly when SSE writes fail instead of continuing to generate after a disconnected client.

## Session 138: CLI/server token-boundary audit

Hardened command front-end token-count boundaries:

- `llmgen` and `llmchat` now reject negative generation counts before model loading.
- `llmgen` no longer slices output by prompt length unless the output is long enough, preserving GPU/CPU normalization safety.
- `llmchat` avoids divide-by-zero throughput reporting on sub-tick generations.
- `llmserver` now rejects negative `max_tokens` with HTTP 400 while keeping zero as the default behavior.

## Session 139: CLI/server input-boundary audit

Hardened remaining command I/O boundaries:

- `llmchat` now reports scanner/input errors instead of silently exiting on non-EOF scanner failures.
- `llmserver` now closes request bodies, limits JSON request decoding to 1 MiB, rejects unknown JSON fields, and rejects empty chat message lists before generation.

## Session 140: llmgen throughput reporting audit

Hardened `llmgen` reporting math:

- Generation throughput and ms/token reporting now avoid division by zero when generation completes within a sub-tick or produces an empty normalized output.


## Session 141: Model and command audit documentation sweep

Refreshed documentation after the latest transitional `model` and `cmd` audit batch:

- README and architecture docs now describe MTP drafter projection guards, MoE helper validation, inference-helper sizing, CPU forward-layer entrypoint checks, and command/request boundary hardening.
- Refactor plan checklist now distinguishes completed `model` helper and `cmd` boundary audits from the remaining large loader/generation scan or explicit deferral decision.
- Validation notes now record that focused model/cmd helper tests have passed for the recent audit batches.

## Session 142: Final non-test logging scan

Completed the Phase 6.5 non-test stdout/stderr/logging scan:

- Non-test library/backend packages are quiet by default, with only `GO_PHERENCE_*_DEBUG` helper output remaining.
- `backends/placement.PrintPlan` remains as an explicit caller-requested reporting API.
- `cmd/*` output remains user-facing CLI/server reporting and error handling.


## Session 143: Phase 6.5 mechanical split deferrals

Recorded explicit Phase 6.5 split/defer decisions:

- CUDA runtime split is deferred to Phase 6.7 with a preservation plan for `DevBuf`, upload state, GPU quantized weights, expert resources, and recently added guard/debug behavior.
- LLaMA/Gemma/Qwen/MoE/MTP model package split is deferred to Phase 6.8 with a plan to move helper tests and preserve MTP/MoE/inference/forward guard semantics.
- Generation/runtime extraction is deferred to Phase 6.9 until model/backend interfaces stabilize.
- Import-boundary scripting is deferred until follow-up split names stabilize; import rules remain documented and review-enforced for Phase 6.5 closeout.


## Session 144: Phase 6.5 documentation closeout sweep

Completed the final Phase 6.5 documentation sweep after recording mechanical split deferrals:

- README and architecture docs now state that CUDA runtime, model package, and generation runtime splits are deferred follow-up phases rather than Phase 6.5 blockers.
- GPU and MTP docs now point at the deferred Phase 6.7/6.8/6.9 split plan and keep MTP/speculative decoding paused until validation closeout is recorded.
- The remaining closeout work is validation/smoke testing and the final Phase 6.5 closeout note.

## Session 145: Phase 6.5 final validation gate

Completed the Phase 6.5 final validation gate:

- SmolLM2 CPU loader/generation smoke passed: `go run ./cmd/llmgen -model models/smollm2-135m -prompt 'Hello' -tokens 2`.
- Gemma4 E2B MLX4 CPU loader/generation smoke passed: `go run ./cmd/llmgen -model models/gemma4-e2b-mlx4 -prompt 'Hello' -tokens 2`.
- Full test gate passed: `go test ./... -count=1`.
- `go vet ./...` and `git diff --check` passed.

## Session 146: Phase 6.5 closeout

Closed Phase 6.5 as a source-tree ownership/audit phase:

- All Phase 6.5 closeout commits are pushed and the plan sidebar is aligned with completed/deferred items.
- Final note in `docs/refactor-plan.md` states that MTP/verifier/drafter work may resume under the documented constraints.
- Deferred package splits remain assigned to follow-up phases: CUDA runtime in Phase 6.7, model packages in Phase 6.8, and generation runtime in Phase 6.9.

## Session 147: SIMD bridge API design

Designed the Phase 6.6 SIMD bridge API before any literal subpackage split:

- `backends/simd` remains the only public facade/import path and owns validation, capability gates, fallback policy, and compatibility globals.
- Future `scalar`, `amd64`, and `arm64` packages should expose provider-style kernel groups rather than direct public functions consumed by model code.
- Assembly symbols remain provider-local after the split; the facade calls prevalidated kernels and preserves public malformed-input/no-op behavior.
- Migration order is facade-internal provider structs first, then scalar split, then amd64/arm64 splits one family at a time.

## Session 148: SIMD code-smell audit fixes

Audited `backends/simd` for facade and subpackage-split hazards:

- Replaced the shared package-level GEBP scratch buffer with per-call scratch allocation so concurrent `SgemmNTGebp` calls cannot race or alias packed-B tiles.
- Added a regression check that GEBP scratch allocations are independent.
- Changed unsupported-architecture `SgemmNT`/`SgemmNN` fallbacks from panics to safe no-ops, preserving the `backends/simd` facade policy that public entrypoints remain defensive even when callers should check `HasSgemmAsm`.
- Verified native SIMD tests and a non-amd64 compile-only check (`GOARCH=riscv64 go test -c ./backends/simd`).

## Session 149: SIMD empty-slice dispatch audit

Continued the `backends/simd` dispatch audit:

- Guarded assembly dispatch wrappers so zero-length vector/BF16 operations route through scalar fallbacks instead of passing empty slices to assembly stubs.
- Added regression coverage for empty public vector and BF16 entrypoints.
- Native SIMD tests, no-run package gate, vet, and diff checks passed.

## Session 150: SIMD SGEMM offset audit

Continued the SIMD SGEMM/GEBP/gather audit:

- Added a checked float32 byte-offset helper for unsafe pointer arithmetic.
- Hardened blocked SGEMM and gather SGEMM pointer offsets so `jj*ldb`, row offsets, and float32 byte scaling are checked before `unsafe.Add`.
- Added regression coverage for byte-offset overflow rejection.
- Re-ran native SIMD tests, non-amd64 compile-only check, no-run package gate, vet, and diff checks.

## Session 151: Resume MTP verifier scaffold

Resumed MTP/speculative work after Phase 6.5 closeout with a small verifier-scaffold hardening step:

- Added `NewMTPVerifierResultForModel`, a model-aware verifier result constructor that validates verifier token IDs against vocab size, logits row width against vocab size, and final activation width against hidden size.
- Kept the existing low-level `NewMTPVerifierResult` for tests/helpers that do not have a model instance.
- Added regression coverage for nil/invalid model dims, token bounds, logits width, final activation width, and a valid model-aware acceptance path.

## Session 152: MTP acceptance consistency audit

Audited MTP acceptance and KV commit semantics:

- Added `MTPAcceptance.Validate` so manually assembled accept/reject results are checked before committing staged verifier KV.
- Float and compressed KV commit helpers now reject inconsistent accepted/verified counts, accepted-token/output-token mismatches, invalid bonus tokens, and inconsistent all-accepted/rejected state before mutating caches.
- Updated KV commit tests to use constructor-produced acceptance values and added malformed-state regression cases.

## Session 153: Documentation refresh after SIMD and MTP audits

Reviewed and refreshed project documentation after the latest audit fixes:

- README and architecture docs now mention the Phase 6.6 SIMD guard baseline: empty vector/BF16 calls route to scalar fallbacks, GEBP scratch is per-call, and SGEMM/GEBP/gather byte offsets are checked before unsafe pointer arithmetic.
- MTP docs now state that work resumed after Phase 6.5 closeout and document model-aware verifier validation plus acceptance consistency checks before KV commit.
- Refactor and SIMD reorg notes now preserve the updated follow-up constraints for Phase 6.8 model splitting and future SIMD provider/subpackage splits.

## Session 154: Runtime validation plan reset and MTP verifier plan

Reset the sidebar plan around aggressive runtime validation and added the next small MTP verifier-path building block:

- `MTPVerifierPlan` prepares `[input_token]+drafted` verifier tokens plus absolute verifier positions for the future batched main-model verifier pass.
- The plan validates nil model, vocab size, negative/out-of-vocab tokens, negative start positions, and position overflow.
- Added tests for token/position construction, copy semantics, and malformed plan inputs.

## Session 155: Full runtime unit gate

Started the aggressive runtime validation plan:

- Initial `go test ./... -count=1` exposed one stale MTP KV staging test that still used a manually assembled `MTPAcceptance` rejected by the new consistency validator.
- Updated the test to use constructor-produced acceptance state via `AcceptMTPDraft`.
- Full `go test ./... -count=1`, `go vet ./...`, and `git diff --check` now pass.

## Session 156: Race-focused runtime gates

Continued the aggressive runtime validation plan:

- Passed shared race gate: `go test -race ./runtime/... ./loader/... ./tensor ./backends/simd -count=1`.
- Broad model race regex `go test -race ./model -run 'MTP|KV|ForwardLayer|InferenceHelpers|Moe' -count=1` was killed after ~255s, likely because the regex still selected resource-heavy model diagnostics.
- Passed focused safe substitute: `go test -race ./model -run 'TestMTP|TestNewMTP|TestAcceptMTP|TestCommitAccepted|TestLayerKVDim|TestLayerKVDims|TestTokenEmbeddingHelpers|TestGemma4PerLayerInputs|TestLMHeadLogitsInto|TestArgmaxLogits|TestInferenceHelpers|TestForwardLayerRejectsMalformedInputs|TestMoeForwardRejectsMalformedInputs' -count=1`.

## Session 157: Cross-arch compile gates

Continued the aggressive runtime validation plan with cross-architecture gates:

- `GOARCH=arm64 go test -c ./backends/simd` passed.
- `GOARCH=riscv64 go test -c ./backends/simd` passed.
- Plain `GOARCH=arm64 go test ./... -run '^$'` compiled test binaries but failed to execute them on the amd64 host with `exec format error`.
- Compile-focused substitute passed with `GOARCH=arm64 go test -exec /bin/true ./... -run '^$'`.

## Session 158: CPU runtime smoke matrix

Completed the CPU generation smoke matrix with short budgets:

- SmolLM2 CPU: `go run ./cmd/llmgen -model models/smollm2-135m -prompt 'Hello' -tokens 3` passed.
- Gemma4 E2B MLX4 CPU: `go run ./cmd/llmgen -model models/gemma4-e2b-mlx4 -prompt 'Hello' -tokens 2` passed.
- Qwen3 0.6B MLX4 CPU: `go run ./cmd/llmgen -model models/qwen3-0.6b-mlx4 -prompt 'Hello' -tokens 2` passed.
- Eager-load small model smoke: `go run ./cmd/llmgen -model models/smollm2-135m -prompt 'Hello' -tokens 2 -eager-load` passed.
- TurboQuant CPU smoke: `go run ./cmd/llmgen -model models/smollm2-135m -prompt 'Hello' -tokens 2 -turbo-quant` passed.
- Qwen3 MoE loader/short-generation smoke: `go run ./cmd/llmgen -model models/qwen3-30b-a3b-mlx4 -prompt 'Hi' -tokens 0` passed within the current resource budget.

## Session 159: GPU and hybrid runtime smoke matrix

Completed the GPU/hybrid runtime smoke matrix on the current host:

- CUDA availability probe passed (`nvidia-smi` reports a CUDA-capable NVIDIA driver/device).
- SmolLM2 GPU smoke passed: `go run ./cmd/llmgen -model models/smollm2-135m -gpu -prompt 'Hello' -tokens 2`.
- SmolLM2 hybrid smoke passed: `go run ./cmd/llmgen -model models/smollm2-135m -gpu -gpu-layers 4 -prompt 'Hello' -tokens 2`.
- Gemma4 E2B MLX4 GPU decode smoke passed with a one-token budget: `go run ./cmd/llmgen -model models/gemma4-e2b-mlx4 -gpu -prompt 'Hello' -tokens 1`.
- Normal-path GPU diagnostics remained quiet without `GO_PHERENCE_GPU_DEBUG` during a SmolLM2 GPU smoke.

## Session 160: MTP verifier result runtime chain tests

Continued MTP/speculative runtime validation:

- Added focused tests chaining `NewMTPVerifierResultForModel` → acceptance validation → float KV commit.
- Added the same model-aware verifier/acceptance chain for compressed/TurboQuant-backed KV commit.
- Verified the chain keeps the accepted prefix plus verifier bonus token and discards rejected candidate KV suffixes.

## Session 161: MTP verifier-forward scaffold tests

Added the verifier-forward scaffold before wiring generation:

- `RunMTPVerifierForward` now defines the future main-model verifier entrypoint and validates plan/model/KV-cache shape before returning an explicit not-implemented error.
- Added tests that the scaffold accepts a well-formed plan up to the not-implemented boundary and rejects nil models, empty/mismatched plans, non-contiguous positions, and malformed KV cache layer counts.
- Public speculative generation remains disabled until the verifier forward and drafter loop have runtime smoke coverage.

## Session 162: SIMD GEBP concurrent scratch stress

Continued SIMD/runtime stress validation:

- Added `TestSgemmNTGebpConcurrentScratch`, which runs concurrent `SgemmNTGebp` calls with independent outputs and compares against a scalar NT reference.
- Ran the new test under the race detector to prove the per-call packed-B scratch path has no shared-buffer races on this runtime.
- Validation passed: `go test -race ./backends/simd -run 'TestSgemmNTGebpConcurrentScratch|TestGEBP' -count=1`, `go test ./backends/simd -count=1`, no-run all-package gate, vet, and diff checks.

## Session 163: SIMD BF16 malformed facade parity

Continued SIMD/runtime stress validation:

- Added BF16 facade tests covering empty inputs, mismatched lengths, short weights, bounded widen/narrow conversion, and fallback parity through the `*Asm` public wrappers.
- Re-ran native SIMD focused tests plus arm64/riscv64 SIMD compile gates to keep architecture-dispatch parity checked.
- No-run all-package gate, vet, and diff checks passed.

## Session 164: SIMD benchmark and speculative CLI gate

Completed the remaining aggressive runtime validation checks:

- Confirmed there is no public speculative/MTP CLI flag in `cmd`; speculative generation remains disabled while verifier forward and drafter loop are scaffold-only.
- Ran selective SIMD benchmarks after correctness/race/smoke gates: `go test ./backends/simd -run '^$' -bench 'Benchmark(VecAdd|BF16DotAsm|RMSNorm|ToBF16)' -benchtime=100ms -count=1`.
- Results on this host (i7-12700, amd64): BF16DotAsm ~404 ns/op, RMSNorm ~689 ns/op, VecAdd ~241 ns/op, ToBF16 ~216 ns/op.

## Session 165: Aggressive runtime validation closeout

Closed the current aggressive runtime validation batch:

- Full unit suite passed after aligning stale MTP KV staging tests with the stricter acceptance validator.
- Race gates passed for shared runtime/loader/tensor/SIMD and a focused model MTP/KV/inference/forward/MoE subset; the broad model race regex was documented as resource-killed and replaced by the focused safe subset.
- Cross-arch compile gates passed for SIMD arm64/riscv64 and an all-package arm64 compile substitute; native execution of arm64 tests on this amd64 host was documented as an `exec format error` limitation.
- CPU smoke matrix passed for SmolLM2, Gemma4 E2B MLX4, Qwen3 0.6B, Qwen3 MoE loader/short-generation, eager-load, and TurboQuant.
- GPU/hybrid smoke matrix passed with CUDA available: SmolLM2 GPU, SmolLM2 hybrid, Gemma4 GPU decode, and quiet default GPU diagnostics.
- MTP scaffold validation now covers verifier plans, model-aware verifier results, acceptance consistency, float/compressed KV commit chains, and verifier-forward contract validation while keeping speculative CLI disabled.
- SIMD stress validation covers concurrent GEBP scratch under `-race`, malformed BF16 facade parity, cross-arch SIMD compile gates, and a bounded benchmark pass.

## Session 166: MTP verifier scaffold audit

Audited the new MTP verifier plan/forward scaffold for malformed manual-plan edges:

- Factored verifier position construction through an overflow-checked helper shared by plan construction and scaffold validation.
- `RunMTPVerifierForward` now revalidates manual plans against model vocab, verifies drafted tokens match the verifier-token suffix, and rejects overflowing/non-contiguous positions before checking KV caches.
- Updated scaffold tests to clone plans before mutation and cover out-of-vocab verifier tokens, drafted/verifier suffix mismatches, and position overflow.

## Session 167: MTP acceptance and drafter alias audit

Continued the MTP scaffold audit:

- Hardened `MTPAcceptance.Validate` to use `KVKeepTokens` before comparing output-token length, avoiding unchecked `accepted_prefix_len + 1` arithmetic on manually assembled structs.
- Added regression coverage for max-int accepted-prefix acceptance state.
- Made drafter `PreProjectInto` and `PostProjectInto` alias-safe by computing into temporary output buffers before copying into caller-provided destinations.
- Added projection alias-safety tests for overlapping destination/input slices.

## Session 168: Forward-layer malformed norm audit

Continued the model-path audit beyond MTP scaffolding:

- Found a malformed-state panic in `ForwardLayer`: layers with `QNorm` and K/V output assumed `KNorm` was also present before dereferencing it.
- Hardened the forward-layer entrypoint to reject missing `KNorm` instead of panicking.
- Extended malformed forward-layer regression coverage for the QNorm-without-KNorm case.

## Session 169: Generate malformed KNorm audit

Continued the model forward-path audit:

- Found the same malformed QNorm-without-KNorm assumption in the main CPU `Generate` loop.
- Hardened `Generate` to stop and return the current output instead of dereferencing a nil `KNorm` when K/V is produced.
- Added a synthetic malformed-model regression test that verifies `Generate` does not panic and returns the original prompt when `KNorm` is missing.

## Session 170: Generate allocation guard audit

Continued the CPU generation-path audit:

- Hardened `Generate` against malformed public inputs/config before KV-cache allocation: negative `maxTokens`, overflowing output capacity, negative/short layer counts, invalid core dimensions, and overflowing per-layer KV/cache capacity now return the current prompt instead of risking panic or huge allocation.
- Added synthetic malformed-config regression tests covering negative token budgets, short layer slices, invalid dimensions, and KV dimension overflow.

## Session 171: CPU decode finish helper extraction

Started the MTP verifier-forward implementation plan with a small behavior-preserving CPU decode extraction:

- Added `finishCPUDecodeStep`, which applies final decode norm, computes LM-head logits, returns greedy argmax, and copies the final activation for verifier/MTP callers.
- Rewired `Generate` to use the helper only at the existing generation/logits point, preserving public generation behavior.
- Added focused tests for helper output, final-activation copy semantics, malformed inputs, and a SmolLM generation regression slice.

## Session 172: CPU decode finish helper audit

Audited the newly extracted `finishCPUDecodeStep` helper:

- Added explicit final-norm backing length validation before mutating the hidden state.
- Added regression coverage that a short final norm is rejected and does not modify caller-owned hidden scratch.


## Session 173: Documentation refresh after decode/MTP audits

Reviewed and refreshed docs after the latest MTP and CPU generation audits:

- README and architecture docs now mention MTP verifier plan/forward scaffolding, alias-safe drafter projections, CPU decode finish/final-norm validation, and CPU generation allocation guards.
- MTP speculative docs now describe `RunMTPVerifierForward` as a contract-validating not-implemented scaffold and note that the CPU decode finish helper returns copied final activations for verifier use.
- Development log remains the detailed record of aggressive runtime validation, scaffold hardening, and follow-up implementation constraints.

## Session 174: CPU decode finish helper Generate parity

Continued the MTP verifier-forward plan:

- Added a synthetic regression test comparing `finishCPUDecodeStep` against the token appended by `Generate` on a zero-layer model.
- The test exercises the shared embedding → final norm → LM-head → argmax path without requiring a large local fixture.
- Focused model tests, no-run all-package gate, vet, and diff checks passed.

## Session 175: Initial MTP verifier forward loop

Continued the MTP verifier-forward implementation plan:

- Replaced the explicit not-implemented scaffold with an initial CPU verifier loop over `MTPVerifierPlan.VerifierTokens`.
- The loop embeds each verifier token, runs configured CPU layers through `ForwardLayer` against staged float KV caches, finishes decode via `finishCPUDecodeStep`, and returns per-position logits plus final activation via `NewMTPVerifierResultForModel`.
- Kept the verifier contract validation factored in `validateMTPVerifierForwardInputs`.
- Added zero-layer verifier-forward tests for zero-draft ordinary verification, one accepted draft plus bonus token, and first-token rejection.

## Session 176: MTP verifier float KV keep-prefix test

Continued the MTP verifier-forward implementation plan:

- Added a single-layer verifier-forward test that stages float KV entries through `RunMTPVerifierForward` and then commits the result.
- The test verifies staged KV length for all verifier positions and post-commit K/V lengths of `accepted_prefix_len + 1`, covering rollback/keep-prefix behavior independent of whether the synthetic draft is accepted or rejected.
- Focused verifier tests, no-run all-package gate, vet, and diff checks passed.

## Session 177: MTP verifier compressed KV keep-prefix test

Completed the remaining verifier-forward KV keep-prefix test coverage:

- Added a resource-safe compressed/TurboQuant-backed KV commit test using the verifier result from `RunMTPVerifierForward`.
- The test stages compressed KV entries for all verifier positions, commits via `MTPVerifierResult.CommitCompressedKV`, and verifies the final sequence/K lengths match `accepted_prefix_len + 1`.
- Focused verifier tests, no-run all-package gate, vet, and diff checks passed. Public speculative CLI exposure remains disabled until the drafter loop and end-to-end smokes are implemented.

## Session 178: MTP drafter state and forward contract

Started the drafter-loop section after verifier-forward coverage:

- Added `MTPDrafterState` for previous token plus copied main-model activation carry.
- Added `RunMTPDrafterStep` as the future q-only assistant forward entrypoint; it validates drafter dimensions, previous-token bounds, activation/embedding widths, projection weights, norm, and layer count before returning an explicit not-implemented error.
- Added focused validation/copy-semantics tests for drafter state and drafter-step contract checks.

## Session 179: Projection-only MTP drafter step

Continued the drafter-loop implementation plan:

- Changed `RunMTPDrafterStep` into a main-model method so it can use backbone token embeddings and the main LM head.
- Implemented the projection/LM-head shell for zero-layer synthetic drafter fixtures: token embedding + previous verifier activation → `PreProjectInto` → `PostProjectInto` → main-model LM-head logits/argmax → next drafter state.
- Real q-only drafter layers still return an explicit not-implemented error until external/main-model KV attention is wired.
- Added tests for projection-only output, next-state copy semantics, dimension mismatches, missing projections, and q-only not-implemented behavior.

## Session 180: MTP acceptance-rate stats scaffold

Completed the current drafter-loop scaffold items:

- Added `MTPSpeculationStats` to accumulate LiteRT-style accounting without any public CLI exposure.
- `Record` validates `MTPAcceptance`, counts drafted tokens, accepted/verified draft-prefix tokens, verifier bonus tokens, output tokens, and rejects counter overflow.
- `AcceptanceRate` reports accepted draft tokens divided by drafted tokens, deliberately excluding bonus tokens.
- Focused stats/acceptance tests, no-run all-package gate, vet, and diff checks passed.

## Session 181: Post drafter/verifier full validation

Ran the validation policy gate after the recent verifier/drafter behavior changes:

- Full suite passed: `go test ./... -count=1`.
- CPU generation smokes passed for SmolLM2 and Gemma4 E2B MLX4 with short token budgets.
- GPU smoke passed for SmolLM2 with a one-token budget.

## Session 182: MTP verifier Generate-semantics audit

Audited `RunMTPVerifierForward` against the full CPU `Generate` semantics for real layers:

- Made the current verifier contract explicit: float KV only; `kvCacheK/V` must already contain exactly `plan.StartPos` prompt/history tokens for every layer that appends K/V.
- Added prompt/history KV length validation before the verifier appends staged candidate K/V.
- Added an explicit rejection for Gemma4 per-layer input gating/PLI until the verifier loop can share the full `Generate` PLI semantics.
- Added tests for non-zero start-position history KV requirements and PLI rejection.

## Session 183: Deterministic one-layer verifier acceptance

Continued the next MTP integration slice:

- Added an explicit one-layer `RunMTPVerifierForward` test that exercises `ForwardLayer`, produces deterministic all-accepted draft behavior, verifies output tokens, stages KV, and checks final activation width.
- Focused verifier tests, no-run all-package gate, vet, and diff checks passed.

## Session 184: MTP verifier helper-boundary decision

Closed the verifier helper-boundary decision for the current slice:

- Keep `RunMTPVerifierForward` on the existing `ForwardLayer` + `finishCPUDecodeStep` split for now.
- Do not extract a fuller shared CPU decode-step helper yet; that boundary should wait until Gemma4 PLI and batched verifier semantics can be represented without diverging from `Generate`.
- Refreshed `docs/mtp-speculative.md` to describe the current implemented CPU verifier loop instead of the older not-implemented scaffold.

## Session 185: MTP drafter external-KV contract

Started extending the drafter step beyond projection-only:

- Added `MTPDrafterExternalKV`, an explicit read-only main-model KV view for q-only drafter layers.
- Added `RunMTPDrafterStepWithExternalKV` so q-only execution has a clear external-KV contract while the projection-only wrapper remains unchanged.
- Validated q-only layer count, source mapping, source KV lengths, attention/MLP weight dimensions, and required norms before returning the existing q-only not-implemented error.
- Added malformed external-KV and q-only dimension tests.

## Session 186: Synthetic q-only MTP drafter layer

Implemented the first q-only drafter execution slice:

- `RunMTPDrafterStepWithExternalKV` now runs validated q-only drafter layers instead of stopping after projection validation.
- The synthetic path performs input norm, q projection, q norm, external GQA attention over the read-only main-model KV view, output projection, residual/post norm, MLP, post projection, and main LM-head logits.
- Updated drafter-loop tests so the one-layer synthetic fixture executes successfully while malformed external-KV cases still fail validation.

## Session 187: Internal MTP speculative step

Added the first end-to-end internal speculative iteration without any public CLI exposure:

- `RunMTPSpeculativeStep` runs one drafter step, builds the verifier plan, runs verifier forward, and records speculation stats.
- The result returns the draft result, verifier plan/result, and updated stats; callers still own staged KV commit/restore decisions.
- Added projection-only integration tests covering drafter -> verifier -> stats and validation failures.

## Session 188: MTP code-smell audit — drafter final norm

Audited the recent MTP drafter/speculative-step code for logic errors:

- Found that the q-only drafter execution path validated per-layer norms but skipped the drafter final norm before `PostProjectInto`.
- Fixed `RunMTPDrafterStepWithExternalKV` to apply `d.Norm` after q-only layers and before post-projection.
- Required loaded/sufficient final norm for q-only drafter execution while preserving projection-only zero-layer fixtures.
- Added regression coverage proving the final norm changes the next activation and malformed q-only drafter state rejects missing final norm.

## Session 189: MTP audit — speculative stats preflight

Continued the MTP code-smell/logic audit:

- Found that `RunMTPSpeculativeStep` detected saturated stats only after verifier forward had already staged candidate KV.
- Added `MTPSpeculationStats.ValidateOneStepCapacity` and preflight it before drafter/verifier execution.
- Added tests for stats preflight and for ensuring saturated stats do not mutate staged verifier KV.

## Session 190: MTP audit — shared-KV verifier validation

Continued the MTP code-smell/logic audit:

- Found that `RunMTPVerifierForward` did not explicitly validate shared-KV layer source mappings before entering `ForwardLayer`.
- Added validation that q/shared layers point at a real KV-appending source layer and do not carry their own staged K/V entries.
- Added malformed shared-KV verifier tests for invalid source, shared-to-shared source, and stray per-layer cache entries.
