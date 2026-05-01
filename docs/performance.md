# Performance

## GTE-small inference comparison

Both implementations produce identical embeddings (within F16 conversion tolerance)
for the same input tokens.

### Headline numbers

| Metric | go-tinygrad | gte-go | Ratio |
|---|---|---|---|
| **Latency (5 tokens)** | **30 ms** | **10 ms** | 3× |
| **Allocs/embed** | 1,672 | 1 | 1672× |
| **Bytes/embed** | 3.5 MB | 7 B | 500K× |
| **Model loading** | 0.24s (safetensors) | 0.15s (.gtemodel) | 1.6× |

### Where the gap comes from

| Source | go-tinygrad cost | gte-go approach | Potential fix |
|---|---|---|---|
| **Per-op buffer alloc** | ~10ms | Pre-allocated buffers, 1 alloc | Buffer pool |
| **Scalar attention** | ~8ms | SIMD for Q·K^T scores | Wire SIMD into MHA |
| **No fused residual+LN** | ~5ms | In-place fused add+layernorm | Extend fusion engine |
| **Tensor object overhead** | ~7ms | No tensor objects in hot path | Pre-allocate graph |

### What go-tinygrad does well

| Feature | go-tinygrad | gte-go |
|---|---|---|
| **Model format** | Any HuggingFace safetensors | Custom .gtemodel only |
| **Op coverage** | 30+ ops, any model architecture | GTE-small specific |
| **Elementwise fusion** | Automatic for chains | Manual per-function |
| **Code reuse** | General tensor API | Hand-rolled per model |
| **Development speed** | Built in ~2 hours | Built over multiple days |

### Micro-benchmarks

| Operation | go-tinygrad | Notes |
|---|---|---|
| Add+Mul (1M elements) | 441ns | Fused: single pass |
| 5-op chain (1M) | 921ns | 2.4× faster than unfused |
| MatMul 64×384 @ 384×384 | 0.49ms | SIMD SgemmNN from gte-go |
| GTE-small forward (5 tokens) | 30ms | 12 layers, 384 hidden |

### Path to closing the gap

1. **Buffer pool** — reuse layer buffers across the 12-layer loop (eliminates ~1600 allocs)
2. **SIMD attention** — use `Sdot` for Q·K^T scores per head (eliminates scalar inner loop)
3. **Fused residual+layernorm** — extend fusion to handle add+normalize in one pass
4. **Pre-transposed weights** — store W^T at load time, use SgemmNN instead of SgemmNT

With all four: estimated 12-15ms (within 1.5× of gte-go).

### Correctness verification

```
Input tokens: [CLS]=101, I=1045, love=2293, cats=8870, [SEP]=102

gte-go reference:      [-0.008219, -0.016258,  0.029982,  0.034200, -0.014450, ...]
go-tinygrad output:    [-0.008220, -0.016258,  0.029980,  0.034193, -0.014451, ...]
Max absolute error:    < 0.0001 (from F16→F32 conversion path differences)
```

Both produce 384-dim L2-normalized vectors with norm ≈ 1.0000.
20 numpy reference tests verify all individual ops match Python ground truth.
