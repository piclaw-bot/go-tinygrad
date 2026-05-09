# go-pherence

![go-pherence](docs/icon-256.png)

**Run MLX models on any hardware.** A pure Go inference engine for Apple MLX, GPTQ and BF16 model weights — with CUDA, Vulkan, and SIMD assembly backends. Single static binary, no Python, no CGo, no external dependencies.

## Why

Apple's [MLX](https://github.com/ml-explore/mlx) ecosystem has the best quantized model collection on HuggingFace (mlx-community). But MLX only runs on Apple Silicon. **go-pherence runs those same MLX weights on NVIDIA GPUs, Intel/AMD CPUs, ARM SBCs, and (soon) any Vulkan device.**

## Performance

| Model | Arch | Format | GPU tok/s | CPU tok/s |
|---|---|---|---|---|
| **Qwen2.5-7B** | qwen2 | MLX 4-bit | **217** | 1.1 |
| **SmolLM2-135M** | llama | BF16 | **86** | 35.5 |
| **Qwen2.5-7B** | qwen2 | GPTQ 4-bit | **51** | 0.9 |
| **Qwen2.5-0.5B** | qwen2 | MLX 4-bit | **31** | 7.2 |
| **Qwen3-0.6B** | qwen3 | MLX 4-bit | **25** | 7.2 |
| **Gemma3-1B** | gemma3 | MLX 4-bit | **18** | 4.9 |
| **Gemma4-E2B** | gemma4 | MLX 4-bit | **14** | — |

*RTX 3060 12GB + i7-12700 6-core. Pure Go, zero CGo.*

## Supported Models

| Architecture | Models | Formats | Status |
|---|---|---|---|
| **llama** | SmolLM2, LLaMA 3.x | BF16, F16, F32 | ✅ |
| **qwen2** | Qwen2.5 0.5B–7B | MLX 4-bit, GPTQ 4-bit | ✅ |
| **qwen3** | Qwen3 0.6B+ | MLX 4-bit, BF16 | ✅ |
| **qwen3_moe** | Qwen3-30B-A3B MoE | MLX 4-bit | ✅ |
| **gemma3** | Gemma 3 1B+ | MLX 4-bit, BF16 | ✅ |
| **gemma4** | Gemma 4 E2B+ | MLX 4-bit | ✅ |

Any model from [mlx-community](https://huggingface.co/mlx-community) using these architectures works out of the box.

## Quick Start

```bash
# Download any MLX model from HuggingFace
mkdir -p models/qwen3-0.6b
for f in config.json model.safetensors tokenizer.json; do
  curl -L "https://huggingface.co/mlx-community/Qwen3-0.6B-4bit/resolve/main/$f" \
    -o "models/qwen3-0.6b/$f"
done

# Run on CPU (AVX2/NEON SIMD)
go run ./cmd/llmgen -model models/qwen3-0.6b -tokens 50 -prompt "The meaning of life is"

# Run on GPU (auto-detects NVIDIA at runtime, zero CGo)
go run ./cmd/llmgen -gpu -model models/qwen3-0.6b -tokens 50 -prompt "The meaning of life is"
```

## Backend Stack

### GPU: CUDA PTX (NVIDIA)

21 hand-written PTX kernels compiled by the driver at runtime via `purego` dlopen:

- **Quantized GEMV**: INT4 dequant+multiply with shared memory tiling (GPTQ + MLX)
- **Batched GEMM**: multi-token prefill, reads weights once for all tokens
- **LM Head**: dedicated large-vocab GEMV with 2D grid
- **RMSNorm, RoPE, GQA Attention, SiLU, VecAdd/Mul/Scale**
- **BF16 kernels**: native `cvt.f32.bf16` on Ampere+ (sm_86), emulated on sm_80
- **MLX bias correction**: post-GEMV fixup for MLX affine quantization

### GPU: Vulkan Compute (any GPU)

Portable compute backend for Intel iGPU, AMD, ARM Mali, Adreno:

- 35 Vulkan API functions via `purego` (no SDK required)
- GLSL compute shaders for all inference ops (F32 + BF16)
- Command buffer dispatch with descriptor sets and push constants
- Device auto-selection: discrete → integrated → software fallback

### CPU: SIMD Assembly

AVX2+FMA (amd64) and NEON (arm64) assembly for all hot paths:

| Operation | AVX2 | NEON | Notes |
|---|---|---|---|
| **Sdot** | 16-wide FMA | 8-wide VFMLA | dot product |
| **RMSNorm** | fused sum-sq + scale | 4-wide | 677ns / 3584 elements |
| **VecAdd** | 16-wide VADDPS | 8-wide FADD | 217ns / 3584 elements |
| **ToBF16** | 16-wide VANDPS | 8-wide AND | 179ns / 3584 elements |
| **BF16 Dot** | widen+FMA+narrow | USHLL+VFMLA | 445ns / 3584 elements |
| **BF16 RMSNorm** | fused BF16 | USHLL+XTN | 1.4µs / 3584 elements |
| **BF16 Widen** | VPMOVZXWD+VPSLLD | USHLL+SHL | 292ns / 3584 elements |
| **SiLU×Mul** | Go (exp not SIMD) | Go fallback | |

Scalar fallback for `!amd64 && !arm64`.

### Native BF16

End-to-end BF16 pipeline for models trained in BF16 (Gemma3/4):

- **Safetensors**: `GetBF16()` returns `[]uint16` without F32 conversion
- **SIMD**: AVX2 and NEON assembly for BF16 dot/norm/add/widen/narrow
- **CUDA**: native `ld.global.b16` / `cvt.f32.bf16` on Ampere+
- **Vulkan**: BF16 emulated via uint16 bitshift (universal)
- **Model**: `BF16Hidden` type with zero-copy operations

## Weight Format Support

| Format | Detection | Dequant | GPU | Notes |
|---|---|---|---|---|
| **MLX affine 4-bit** | `config.json` quantization block | `val × scale + bias` | Transpose → GPTQ kernel | Primary format |
| **GPTQ INT4** | `quantize_config.json` | `(val - 8) × scale` | Native tiled GEMV | Symmetric |
| **BF16** | safetensors dtype | Direct load | F32 on GPU | Half bandwidth |
| **F16** | safetensors dtype | F16→F32 at load | F32 on GPU | |
| **F32** | safetensors dtype | Direct load | Native | |

## Commands

### llmgen — one-shot text generation

```bash
go run ./cmd/llmgen -model models/qwen3-0.6b-mlx4 -gpu -tokens 50 -prompt "The meaning of life is"
```

### llmchat — interactive chat

```bash
go run ./cmd/llmchat -model models/gemma4-e2b-mlx4 -gpu -n 256
> Hello
Hello! How can I help you today?
[8 tok, 13.5 tok/s, 592ms]
```

### llmserver — OpenAI-compatible API server

```bash
go run ./cmd/llmserver -model models/gemma4-e2b-mlx4 -gpu -listen :8080
# Test with curl:
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-e2b-mlx4","messages":[{"role":"user","content":"Hello"}]}'
```

All commands support `--gpu-layers N` for hybrid CPU/GPU inference (0=all on GPU).

## Architecture Details

- **Lazy tensor DAG** with elementwise fusion
- **Pattern matcher + graph rewrite** (tinygrad-style, 16 rules)
- **Safetensors loader** — sharded, F16/BF16/F32, GPTQ/MLX quantized
- **Tokenizer** — BPE with auto-detect SentencePiece `▁` vs GPT-2 `Ġ` prefix
- **LLaMA decoder** — RoPE (global + local), GQA, KV cache, SiLU/GELU MLP
- **QK-Norm** — per-head RMSNorm (Qwen3, Gemma3/4)
- **4-norm residual** — pre/post FFN norms (Gemma3/4)
- **Sliding window attention** — alternating local/global (Gemma3/4)
- **Embedding scaling** — `× √hidden_size` (Gemma3/4)
- **Batched prefill** — GEMM for multi-token prompt processing
- **Chunked LM head** — splits across available VRAM
- **GPU DevBuf** — device-agnostic buffers, lazy CPU↔GPU transfer

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — UOp graph, fusion, SIMD dispatch
- **[docs/gemma4-precision.md](docs/gemma4-precision.md)** — Gemma4 GPU correctness & precision
- **[docs/weight-budget.md](docs/weight-budget.md)** — tiered weight budget manager design
- **[docs/performance.md](docs/performance.md)** — benchmarks, roadmap
- **[docs/gpu-options.md](docs/gpu-options.md)** — GPU compute paths
- **[docs/development-log.md](docs/development-log.md)** — build process

## License

MIT
