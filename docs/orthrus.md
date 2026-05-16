# Orthrus port notes

Orthrus is a Qwen3-derived dual-view generation scheme from `chiennv2000/orthrus` and the `chiennv/Orthrus-Qwen3-*` checkpoints. It adds a diffusion attention view to dense Qwen3 and verifies proposed blocks with the normal autoregressive path, preserving the base model distribution.

## Fit with go-pherence

Orthrus is useful primarily as an algorithmic reference. The published implementation obtains its lossless speedup from custom-trained diffusion-view attention tensors, but go-pherence should not require custom Orthrus checkpoint weights.

Useful pieces:

- the verifier is an ordinary autoregressive pass over a proposed block;
- accepted tokens share the normal high-fidelity KV cache;
- cache crop/accept/reject mechanics are directly relevant to generic speculative decoding.

Not directly reusable without custom weights:

- the diffusion proposer itself;
- the `*_diff` projection/norm tensors;
- the strict Orthrus acceptance-rate claims.

The initial safe behavior is to recognize Orthrus checkpoints for metadata/debugging only. For normal go-pherence models, reuse the verification/cache-control structure rather than the custom diffusion view.

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

1. Detect Orthrus metadata in `LlamaConfig` for compatibility/debugging.
2. Keep Orthrus diffusion generation disabled unless custom `*_diff` tensors are explicitly loaded.
3. Port the generic verifier mechanics to standard models:
   - run prompt AR pass and keep shared KV cache;
   - accept a proposed block from a non-custom proposer;
   - run an AR verifier pass over the proposed block;
   - accept the longest matching prefix and crop KV cache.
4. Start with deterministic greedy verification.
5. Add exact sampling later using speculative-decoding residual distribution logic.
6. Move hot verifier-block paths to GPU only after CPU parity tests pass.

Current opt-in hook:

- `llmgen -speculative`, `llmchat -speculative`, `llmserver -speculative` / `GO_PHERENCE_SPECULATIVE=1` route CPU generation through `GenerateSpeculative`.
- Tunables: `-speculative-block`, `-speculative-ngram`, `-speculative-min-proposal`, `-speculative-proposer`, `-speculative-debug` or `GO_PHERENCE_SPECULATIVE_BLOCK`, `GO_PHERENCE_SPECULATIVE_NGRAM`, `GO_PHERENCE_SPECULATIVE_MIN_PROPOSAL`, `GO_PHERENCE_SPECULATIVE_PROPOSER`, `GO_PHERENCE_SPECULATIVE_DEBUG=1`.
- The first proposer is `PromptLookupProposer`, an n-gram/prompt-lookup proposer for repeated text behind the `SpeculativeProposer` interface. `-speculative-proposer none` disables proposals for fallback-overhead measurements.
- The current verifier scaffold runs greedy verification with the real model and accepts the longest matching prompt-lookup prefix. Debug stats report `backend=replay proposer=prompt` until the KV-reusing backend lands.
- This first verifier is correctness-oriented and reuses the existing CPU generator from a prepared prompt, so it can be slower; the speedup requires replacing it with a stateful/batched verifier block that reuses KV cache.
- `CPUDecodeState` now defines the KV/output checkpoint, restore, `GenerateGreedy`/`DecodeOneGreedy`, accepted-prefix commit, and `VerifyGreedyBlock` contract for that stateful verifier.
- `GenerateSpeculativeWithStats` returns structured acceptance/fallback metrics for benchmarks without scraping debug stderr, including emitted-token and tokens-per-step derived metrics.
- `cmd/specbench` runs normal vs speculative generation and emits CSV rows with timing, speedup-vs-normal, parity, and structured speculative stats.

Candidate non-custom proposers:

- n-gram/cache proposer for repeated text;
- prompt lookup decoding;
- smaller existing model as drafter;
- self-speculative early-exit/layer-skip proposer if confidence is good enough;
- CPU heuristic proposer for structured/code completions.

## Non-goals

- This does not advance NVFP4 support directly.
- This does not apply to Qwen3 MoE checkpoints in the current Orthrus release.
- Do not require or depend on Orthrus custom checkpoint weights.
- Do not claim Orthrus-style lossless diffusion speedups for stock Qwen/Gemma weights; only the verifier is lossless, while speedup depends on proposer quality.
