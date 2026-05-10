# TurboQuant KV Cache Compression

TurboQuant is an optional CPU-backend KV cache compression path for long-context
inference.

## Usage

```bash
# CPU backend with compressed KV cache
./bin/llmgen -model models/gemma4-e2b-mlx4 -prompt "..." -tokens 256 --turbo-quant

# Existing environment variable remains supported for diagnostics
TURBO_QUANT=1 ./bin/llmgen -model models/gemma4-e2b-mlx4 -prompt "..." -tokens 256
```

`--turbo-quant` currently applies to the CPU backend only. If `--gpu` is also
provided, the CLI prints a warning; GPU KV-cache compression is not wired yet.

## Current implementation

- Per-layer `CompressedKVCache` wrapper for CPU `LlamaModel.Generate`.
- Recent tokens stay full precision via a 128-token residual window.
- Older tokens are compressed on append.
- First/last protected layers stay full precision.
- K uses 4-bit quantization; V uses 2-bit quantization.
- K/V are rotated by deterministic random orthogonal matrices before quantizing.
- Gemma4 variable head dimensions are handled by caching one TurboQuant state per
  `headDim`.
- Attention reads call `GetK()`/`GetV()`, which decompress into reusable scratch
  buffers to avoid per-token full-cache allocation churn.

## Validation snapshot

Unit tests cover:

- Roundtrip quality: 4-bit K max error about `0.455`, 2-bit V max error about
  `1.715` for the synthetic test vector.
- Compression layout: 200-token cache with a 128-token residual window stores 72
  compressed and 128 full-precision entries.
- Protected layers never compress.
- Short prompts under the residual window produce the same first tokens as the
  uncompressed CPU path.

## Limitations

- CPU path only.
- Quantization is currently range-adaptive uniform after rotation. The code keeps
  analytic codebook scaffolding, but non-uniform/Beta-optimal levels are not used
  yet because the current uniform path has lower measured error in tests.
- Decompression still happens before the existing attention kernel; a future
  optimized path should read compressed blocks directly or cache per-window
  dequantized pages.
- GPU-side compression/decompression and compressed KV attention are future work.
