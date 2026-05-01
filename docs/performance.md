# Performance

## GTE-small inference: go-tinygrad vs gte-go

| | go-tinygrad fast | go-tinygrad API | gte-go |
|---|---|---|---|
| **Latency** | **14.9 ms** | 15.5 ms | 10.5 ms |
| **Allocs/embed** | **8** | 1,668 | 1 |
| **Bytes/embed** | **93 KB** | 3.5 MB | 7 B |
| **Gap to gte-go** | **1.4×** | 1.5× | — |

## Optimization progression

| Step | ms | Allocs | Change |
|---|---|---|---|
| Initial (tensor API) | 30 | 1,672 | Per-op buffers, math.Tanh, UOp interning |
| + Pre-transpose weights | 27 | 1,746 | SgemmNN instead of SgemmNT |
| + Fast tanh GELU | 20 | 1,746 | Padé approximant, no float64 |
| + Skip UOp interning | 15 | 1,668 | No fmt.Sprintf in hot path |
| + Workspace (fast path) | 14.5 | 8 | Pre-allocated ping-pong buffers |
| + Fused QKV (3→1 GEMM) | **14.9** | **8** | Single NT matmul for Q,K,V |

## Where the remaining 1.4× gap comes from

| Source | go-tinygrad | gte-go |
|---|---|---|
| GEMM kernel | SgemmNN/NT (same SIMD) | gonum NT (different cache pattern) |
| Workspace alloc | 8 allocs at Embed() call | 0 (pre-allocated on model) |
| QKV split | memcpy deinterleave | Zero-copy slice |
| Attention | Scalar per-head | Scalar per-head (same) |

The compute kernels are identical (same SIMD assembly from gte-go).
The gap is purely in buffer management: gte-go pre-allocates everything
at model load and reuses across calls with zero per-embed allocation.
