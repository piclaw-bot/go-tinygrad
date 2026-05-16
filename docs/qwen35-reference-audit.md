# Qwen3.5/Qwen3.6 reference audit

Reference: `/tmp/llama-mtp-clean/src/models/qwen35.cpp` and `/tmp/llama-mtp-clean/conversion/qwen.py`.

## Confirmed matches

- Full-attention path uses Q+gate projection, Q/K RMSNorm, RoPE, GQA attention, sigmoid gate, output projection, residual, post-attention RMSNorm, SwiGLU MLP.
- MTP block is treated as dense/full-attention only and delegates through the shared Qwen3.5 full-attention path.
- Main layer routing uses recurrent/linear layers except full-attention interval and excludes MTP tail.
- Linear-attention state is separated from full-attention KV cache and can be cloned for rollback.

## Divergences found and fixed

- `in_proj_qkvz` conversion in the reference splits HF `[q,k,v,z]` into two tensors: `wqkv=[q,k,v]` and `wqkv_gate=z`. Our previous shape incorrectly included `z` inside QKV. Fixed `Qwen35LinearAttentionShapes.QKV` back to `conv_dim` and use `GateW` for z.
- Conv input in reference is the full QKV stream (`q,k,v`), not just K/V. Fixed linear forward to append Q,K,V to conv state.
- Conv output in reference is passed through SiLU before splitting. Added `siluInPlace` before Q/K/V split.
- Reference splits convolved stream as Q, K, V and L2-normalizes Q and K. Added Q/K split and `l2NormalizeInPlace`.
- Reference applies sigmoid to beta. Added beta sigmoid before recurrent update.
- Reference uses converted `A = -exp(A_log)` and `gate = softplus(alpha + dt_bias) * A`. Fixed decay prep to `exp(dt * A)` assuming converted negative A.
- Reference applies gated RMSNorm: `RMSNorm(recurrent_out, ssm_norm) * SiLU(z)`. Replaced simple sigmoid-z gating with RMSNorm + SiLU(z).

## Still approximate / needs parity

- `applyQwen35LinearDeltaUpdate` is a scalar correctness-first approximation of `build_recurrent_attn`, not proven equivalent to ggml fused/non-fused GDN.
- Conv kernel orientation may still need real tensor parity checks against a converted checkpoint.
- State tensor ordering is plausible but must be checked against `llama-memory-recurrent` layout. The scalar recurrence now explicitly uses tiled key-head mapping (`value_head % key_heads`) to mirror the reference repeat path when `num_v_heads > num_k_heads`.
- MRoPE section handling is not fully equivalent; current Go RoPE path is simple frequency-table driven.
