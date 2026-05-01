# go-tinygrad

A minimal tensor computation framework in pure Go with SIMD assembly,
inspired by [tinygrad](https://github.com/tinygrad/tinygrad).

**Not a 1:1 port.** Takes tinygrad's core ideas — lazy evaluation, graph-based
fusion, minimal op set — and reimplements them idiomatically in Go with the
same flat-latency, zero-allocation philosophy from [gte-go](https://github.com/rcarmo/gte-go).

## Status: runs GTE-small end-to-end

```
gte-go reference:      [-0.008219, -0.016258,  0.029982,  0.034200, -0.014450]
go-tinygrad output:    [-0.008220, -0.016258,  0.029980,  0.034193, -0.014451]
```

Loads a HuggingFace safetensors model, runs 12 transformer layers, produces
L2-normalized embeddings matching the hand-optimized reference within F16 tolerance.

| | go-tinygrad | gte-go |
|---|---|---|
| **Latency** | 30 ms | 10 ms |
| **Correctness** | ✅ | ✅ |
| **Model format** | Safetensors (any HF model) | Custom .gtemodel |
| **Code** | 4,240 lines (general framework) | ~8,000 lines (hand-tuned) |

## Quick Start

```go
import (
    "github.com/rcarmo/go-tinygrad/model"
    "github.com/rcarmo/go-tinygrad/tensor"
)

m, _ := model.LoadGTESmall("model.safetensors")
emb := m.Embed([]int{101, 1045, 2293, 8870, 102}, []bool{true, true, true, true, true})
// emb is a 384-dim L2-normalized float32 embedding
```

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — design and core abstractions
- **[docs/development-log.md](docs/development-log.md)** — step-by-step build process
- **[docs/performance.md](docs/performance.md)** — benchmarks and optimization analysis

## License

MIT
