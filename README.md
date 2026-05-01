# go-tinygrad

A minimal tensor computation framework in pure Go with SIMD assembly,
inspired by [tinygrad](https://github.com/tinygrad/tinygrad).

## Runs LLMs in pure Go

```
$ llmgen -model SmolLM2-1.7B -tokens 50 -prompt "The capital of France is"

The capital of France is Paris.
The capital of the United States is Washington, D.C.
The capital of the United Kingdom is London.
The capital of Canada is Ottawa.
The capital of Australia is Canberra.
```

| Model | Params | tok/s | ms/tok |
|---|---|---|---|
| SmolLM2-135M | 135M | **29.3** | 34 |
| SmolLM2-360M | 360M | **10.8** | 93 |
| SmolLM2-1.7B | 1.7B | **1.6** | 621 |
| GTE-small (encoder) | 23M | — | 10.8ms/embed |

Single static binary. No Python, no C, no GGUF, no external dependencies.
AVX2+FMA on amd64, NEON on arm64. Zero per-inference allocations.

## Features

- **Lazy tensor DAG** with elementwise fusion
- **Pattern matcher + graph rewrite** (tinygrad-style)
- **SIMD GEMM kernels** — AVX2 VGATHERDPS, NEON GEBP (from gte-go)
- **Safetensors loader** — F16/BF16/F32 from HuggingFace
- **LLaMA decoder** — RoPE, GQA, KV cache, SiLU MLP
- **BERT encoder** — GTE-small at gte-go parity (10.8ms)
- **BPE tokenizer** — GPT-2 byte-level
- **NN modules** — Linear, LayerNorm, Embedding

## Quick Start

```bash
# Download a model
mkdir -p models/smollm2-135m
curl -L https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors -o models/smollm2-135m/model.safetensors
curl -L https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/config.json -o models/smollm2-135m/config.json
curl -L https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/tokenizer.json -o models/smollm2-135m/tokenizer.json

# Run
go run ./cmd/llmgen -model models/smollm2-135m -tokens 50 -prompt "Once upon a time"
```

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — UOp graph, fusion, SIMD dispatch
- **[docs/development-log.md](docs/development-log.md)** — build process
- **[docs/performance.md](docs/performance.md)** — benchmarks vs gte-go

## License

MIT
