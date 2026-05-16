# Orthrus port notes

Orthrus is a Qwen3-derived dual-view generation scheme from `chiennv2000/orthrus` and the `chiennv/Orthrus-Qwen3-*` checkpoints. It adds a diffusion attention view to dense Qwen3 and verifies proposed blocks with the normal autoregressive path, preserving the base model distribution.

## Fit with go-pherence

Orthrus fits as a new generation mode rather than a new quantization family:

- the base path is standard dense Qwen3;
- checkpoints add per-layer diffusion attention tensors;
- MLP, embeddings, final norm, LM head, and autoregressive KV cache are shared;
- the verifier is the existing autoregressive forward path over a proposed block.

The initial safe behavior is to load Orthrus checkpoints as baseline Qwen3 and ignore diffusion tensors. The speedup requires explicit Orthrus generation support.

## Checkpoint markers

Known public checkpoints:

- `chiennv/Orthrus-Qwen3-1.7B`
- `chiennv/Orthrus-Qwen3-4B`
- `chiennv/Orthrus-Qwen3-8B`

Config markers:

```json
{
  "architectures": ["OrthrusLM"],
  "model_type": "qwen3",
  "block_size": 32,
  "mask_token_id": 151669
}
```

Additional tensors per layer:

```text
model.layers.N.self_attn.q_proj_diff.weight
model.layers.N.self_attn.k_proj_diff.weight
model.layers.N.self_attn.v_proj_diff.weight
model.layers.N.self_attn.o_proj_diff.weight
model.layers.N.self_attn.q_norm_diff.weight
model.layers.N.self_attn.k_norm_diff.weight
```

## Port sequence

1. Detect Orthrus metadata in `LlamaConfig`.
2. Baseline-load checkpoints through the existing Qwen3 path, ignoring `*_diff` tensors.
3. Add CPU structs for diffusion attention weights.
4. Add a greedy-only Orthrus generation mode:
   - run prompt AR pass and keep shared KV cache;
   - generate the first next token normally;
   - build a masked block of length `block_size`;
   - run the diffusion attention view over that block against the AR KV cache;
   - run an AR verifier pass over the proposed block;
   - accept the longest matching prefix and crop KV cache.
5. Add exact sampling later using the paper/PyTorch residual distribution logic.
6. Move hot diffusion/verifier block paths to GPU only after CPU parity tests pass.

## Non-goals

- This does not advance NVFP4 support directly.
- This does not apply to Qwen3 MoE checkpoints in the current Orthrus release.
- Do not enable diffusion generation by default until token-level parity against the Python implementation is established.
