# go-tinygrad

A minimal tensor computation framework in pure Go with SIMD assembly,
inspired by [tinygrad](https://github.com/tinygrad/tinygrad).

**Not a 1:1 port.** Takes tinygrad's core ideas — lazy evaluation, graph-based
fusion, minimal op set — and reimplements them idiomatically in Go with the
same flat-latency, zero-allocation philosophy from [gte-go](https://github.com/rcarmo/gte-go).

## Goals

- **Pure Go + assembly** — single static binary, no CGo, no Python
- **Lazy DAG evaluation** — build computation graphs, fuse ops, execute on demand
- **SIMD kernels** — AVX2+FMA on amd64, NEON on arm64 (from gte-go)
- **Minimal allocations** — reusable buffers, zero GC pressure in hot paths
- **Inference-first** — optimize for running pretrained models, not training

## Architecture

```
tensor/          — Tensor type, lazy op construction
  tensor.go      — user-facing API (Add, Mul, MatMul, Reshape, etc.)
  ops.go         — op enum and UOp graph node
  dtype.go       — data types (float32, float16, int32, etc.)
  shape.go       — shape tracking, strides, broadcasting

engine/          — graph scheduling and execution
  schedule.go    — topological sort, kernel fusion, linearization
  realize.go     — buffer allocation, kernel dispatch
  fuse.go        — elementwise fusion rules

kernel/          — CPU kernel implementations
  buffer.go      — memory management (mmap, pool)
  unary.go       — exp2, log2, sqrt, neg, reciprocal, cast
  binary.go      — add, sub, mul, div, max, cmplt
  reduce.go      — sum, max reduction over axes
  matmul.go      — GEMM dispatch (reuses gte-go SIMD kernels)
  movement.go    — permute, reshape, expand, pad (view-only where possible)

simd/            — SIMD assembly kernels (shared with gte-go)
  sdot, saxpy, sgemm, gebp, gather, pack...

cmd/
  tinydemo/      — simple inference demo
```

## Core Concepts (from tinygrad)

### UOp — the universal IR node

Every computation is a node in a directed acyclic graph:

```go
type UOp struct {
    Op    Ops       // what operation (ADD, MUL, REDUCE, RESHAPE, ...)
    DType DType     // element type (Float32, Int32, ...)
    Src   []*UOp    // input nodes
    Arg   any       // op-specific data (shape, axis, const value)
}
```

UOps are **interned** (hash-consed) — identical subgraphs share a single node.
This makes graph comparison O(1) and enables efficient pattern matching for fusion.

### Lazy evaluation

```go
a := tensor.Rand([]int{3, 4})       // no computation yet
b := tensor.Rand([]int{3, 4})
c := a.Add(b).Mul(a)                // builds UOp DAG
result := c.Realize()                // fuses Add+Mul, dispatches single kernel
```

### Minimal op set for inference

| Category | Ops |
|---|---|
| **Movement** | Reshape, Permute, Expand, Pad, Shrink |
| **Unary** | Exp2, Log2, Sqrt, Neg, Reciprocal, Cast |
| **Binary** | Add, Sub, Mul, Div, Max, CmpLt, Where |
| **Reduce** | Sum, Max (over axes) |
| **Memory** | Buffer, Load, Store, Const |

MatMul is expressed as Expand + Mul + Reduce(Sum) — no special op needed.

## Status

🚧 Early development. Building the foundation.

## License

MIT
