# DeepSeek-V4 Mini Inference

This folder contains the inference layer for the project. It is designed around the cache shapes implied by DeepSeek-style hybrid attention instead of pretending every layer is a plain GPT-style `past_key_values` tuple.

## Scope

Implemented in this pass:

- Inference configuration validation.
- Heterogeneous per-layer cache dataclasses:
  - `MHACache`
  - `HCALayerCache`
  - `CSALayerCache`
  - `DeepSeekV4InferenceCache`
- Sampling utilities:
  - greedy
  - temperature
  - top-k
  - top-p
  - repetition penalty
- Prompt prefill.
- Single-token decode step.
- Autoregressive generation.
- Optional MTP draft diagnostics.
- Cache memory and generation-speed summaries.
- High-level `inference_autoregresive(...)` wrapper.
- Notebook/debug audit wrapper: `audit_inference_pipeline(...)`.

Inference has explicit cache modes:

- `cache_mode="audit"` keeps generation exact with full-forward logits while maintaining HCA/CSA/MHA shadow caches for inspection.
- `cache_mode="mha_decode"` uses active MHA KV-cache decode. This is for pure `attention_type="mha"` models.
- `cache_mode="deepseek_decode"` uses active HCA/CSA compressed-cache decode for HCA, CSA, and hybrid HCA/CSA models. It supports mHC, MoE, and MTP diagnostics.

In `deepseek_decode`, logits come from cached states. The path updates real layer-projection cache states, compresses ready HCA/CSA blocks, keeps local windows and pending tails, and avoids full-sequence recomputation.

## Files

```text
inference/
├── __init__.py
├── inference_config.py      # generation/cache configuration
├── cache_base.py            # common cache protocol
├── mha_cache.py             # token-level K/V cache
├── hca_cache.py             # compressed + local + pending HCA cache
├── csa_cache.py             # compressed main/index + local + pending/previous CSA cache
├── hybrid_cache.py          # whole-model heterogeneous cache
├── cache_utils.py           # dtype/device/tokenizer/cache helpers
├── prefill.py               # prompt processing and cache initialization
├── decode.py                # single-token decode step and cache update
├── generate.py              # autoregressive generation and wrapper API
├── sampling.py              # sampling/filtering utilities
├── mtp_decode.py            # optional MTP draft diagnostics
└── metrics.py               # cache and generation metrics
```

## Basic Usage

```python
from inference import inference_autoregresive

out = inference_autoregresive(
    model,
    prompt="key key_1 is value_4 question what is key_1 ? answer :",
    tokenizer=tokenizer,
    max_new_tokens=32,
    do_sample=False,
    eos_token_id=tokenizer.eos_id,
    pad_token_id=tokenizer.pad_id,
    return_cache_stats=True,
)

print(out["text"])
print(out["cache_stats"])
```

For a broader notebook audit:

```python
from inference import audit_inference_pipeline

audit = audit_inference_pipeline(
    model,
    prompt="key key_1 is value_4 question what is key_1 ? answer :",
    tokenizer=tokenizer,
    max_new_tokens=8,
    do_sample=False,
    compare_logits=True,
)

print(audit["generation"]["sequences"])
print(audit["cache_stats"])
print(audit["full_vs_cached"])
```

The correctly spelled alias is also exported:

```python
from inference import inference_autoregressive
```

## Lower-Level API

```python
from inference import InferenceConfig, generate

cfg = InferenceConfig(
    cache_mode="deepseek_decode",
    max_new_tokens=64,
    do_sample=False,
    return_cache_stats=True,
)

out = generate(model, input_ids=input_ids, inference_config=cfg)
```

## Current Limitation

The active DeepSeek path is research-code scale, not production serving infrastructure. It intentionally does not implement:

- custom CUDA kernels,
- paged attention,
- disk-backed KV cache,
- quantized FP4/FP8 KV storage,
- distributed serving.
