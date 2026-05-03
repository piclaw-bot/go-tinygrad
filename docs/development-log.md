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

Built complete BERT model (`model/bert.go`):
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
| `safetensors/` | 3 | load, list, F16 conversion |
| `model/` | 2 | load weights, end-to-end embed |
| **Total** | **47** | |
