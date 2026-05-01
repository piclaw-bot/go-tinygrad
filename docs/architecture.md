# Architecture

## Design principles

1. **Lazy evaluation** — tensor ops build a DAG, computation happens at `Realize()`
2. **UOp graph** — single IR node type for the entire computation graph
3. **Elementwise fusion** — chains of unary/binary ops execute in one pass
4. **SIMD kernels** — AVX2+FMA (amd64) and NEON (arm64) from gte-go
5. **Zero-copy model loading** — safetensors F16→F32 conversion on read

## Package structure

```
tensor/          Core tensor types and operations
  tensor.go      Tensor API: constructors, lazy ops, Realize()
  ops.go         Op enum (30+ operations)
  dtype.go       Data types (Float32, Float16, Int32, ...)
  shape.go       Shape tracking with strides
  uop.go         UOp graph node (interned/hash-consed)
  broadcast.go   Shape broadcasting for binary ops
  realize.go     Eager graph interpreter
  fuse.go        Elementwise fusion engine
  matmul.go      MatMul + Linear with SIMD GEMM
  nn.go          Softmax, LayerNorm, GELU
  embedding.go   Token ID → vector lookup

safetensors/     HuggingFace model file loader
  safetensors.go Parse header, F16/BF16/F32 conversion

model/           Pre-built model architectures
  bert.go        BERT encoder (loads GTE-small, runs inference)

simd/            SIMD assembly kernels (ported from gte-go)
  Sdot, Saxpy    Vector dot product, scaled add
  SgemmNN/NT     Matrix multiply (NoTrans, Trans)
  GEBP           General Block Panel micro-kernels
  Gather         AVX2 VGATHERDPS for NT without packing

cmd/tinydemo/    Demo program
```

## UOp — the universal IR node

Every computation is a node in a directed acyclic graph:

```go
type UOp struct {
    Op    Ops       // ADD, MUL, REDUCE, RESHAPE, ...
    DType DType     // Float32, Int32, ...
    Src   []*UOp    // input nodes (immutable)
    Arg   any       // op-specific (shape, axis, const value)
    buf   *Buffer   // realized data (nil until Realize)
}
```

UOps are **interned** — identical subgraphs share a single node via hash-consing.
Buffer UOps are NOT interned (each allocation is unique by identity).

## Elementwise fusion

The fuser walks the UOp DAG and compiles chains of fusible ops into a single
kernel that evaluates all ops per-element in one pass:

```
Before: a.Add(b).Mul(c)
  → alloc tmp; for i { tmp[i] = a[i]+b[i] }
  → alloc out; for i { out[i] = tmp[i]*c[i] }

After fusion:
  → alloc out; for i { out[i] = (a[i]+b[i]) * c[i] }
```

Fusion rules:
- **Fusible**: all unary + binary ALU ops
- **Breaks fusion**: Reduce, MatMul, Permute, broadcast, already-realized buffers

## SIMD kernel dispatch

MatMul uses the gte-go SIMD suite:

| Operation | amd64 | arm64 |
|---|---|---|
| `MatMul` (A@B) | `SgemmNN` AVX2+FMA | `SgemmNN` NEON |
| `MatMulTransposed` (A@B^T) | `SgemmNT` VGATHERDPS | `SgemmNT` GEBP NEON |
| Attention Q·K^T | `Sdot` AVX2 | `Sdot` NEON |

## Model inference flow

```
LoadGTESmall(safetensors)
  → parse header, F16→F32 conversion
  → create Tensor weights for each layer

Embed(tokenIDs, attnMask)
  → word + position + type embeddings
  → LayerNorm
  → 12× {
      Q,K,V = Linear projections (MatMulTransposed + bias)
      scores = Q·K^T / sqrt(headDim) per head
      probs = softmax(scores)
      context = probs @ V
      output = Linear(context) + residual → LayerNorm
      ffn = Linear → GELU → Linear + residual → LayerNorm
    }
  → mean pooling → L2 normalize
```
