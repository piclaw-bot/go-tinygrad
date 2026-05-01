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

## LLM generation

| Model | Params | Hidden | Layers | tok/s | ms/tok |
|---|---|---|---|---|---|
| SmolLM2-135M | 135M | 576 | 30 | **29.3** | 34 |
| SmolLM2-360M | 360M | 960 | 32 | **10.8** | 93 |
| SmolLM2-1.7B | 1.7B | 2048 | 24 | **1.6** | 621 |
| Qwen2.5-7B | 7.6B | 3584 | 28 | **0.9** | 1075 |

All CPU, pure Go + AVX2/FMA SIMD. F32 weights from BF16 safetensors.

## Roadmap

### Quantization: Intel AutoRound

[AutoRound](https://github.com/intel/auto-round) is an advanced weight
quantization algorithm that exports to multiple formats (AutoRound, GPTQ,
AWQ, GGUF). Key benefits for go-tinygrad:

- **INT4/INT8 quantized weights** — 4× less memory, enabling 7B models
  in ~4GB instead of ~28GB (F32) or ~14GB (BF16)
- **Safetensors export** — AutoRound/GPTQ models are stored as
  safetensors with quantization metadata, so our loader can be extended
  to handle them directly
- **Minimal accuracy loss** — SignRound optimization preserves model
  quality better than naive RTN quantization
- **Wide model coverage** — pre-quantized models available on HuggingFace
  for Qwen, LLaMA, Phi, Mistral, etc.

**Implementation plan:**
1. Add INT4 dequantization kernels (SIMD: unpack 4-bit → F32 with scale/zero)
2. Extend safetensors loader for GPTQ/AutoRound quantization metadata
3. Add group-wise dequantization (typically groups of 128 weights)
4. Load pre-quantized models from HuggingFace (e.g., `Intel/Qwen2.5-7B-Instruct-int4-inc`)
5. Optional: CUDA INT4 GEMM via cuBLAS for GPU-accelerated quantized inference

**Expected impact:** Qwen2.5-7B in ~4GB RAM at 3–5× faster inference
(memory bandwidth is the bottleneck for token generation).

### GPU: cuBLAS backend

- `gpu/` package with cuBLAS SGEMM and graceful CPU fallback
- RTX 3060 (12GB VRAM): estimated ~20× speedup for GEMM-bound inference
- Build with `CGO_ENABLED=1` for GPU; `CGO_ENABLED=0` for static CPU binary

### Future

- Vulkan compute backend (portable GPU across vendors)
- Conv2d for vision models
- Autograd for training
- More architectures (Mistral, Phi)
