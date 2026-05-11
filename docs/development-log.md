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
