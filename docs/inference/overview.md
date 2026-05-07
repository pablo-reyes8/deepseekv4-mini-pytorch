# Inference Overview

The inference stack is designed around one public wrapper:

```python
from inference import inference_autoregresive

out = inference_autoregresive(
    model=model,
    prompt="key key_1 is value_7 question : what is key_1 ? answer :",
    tokenizer=tokenizer,
    max_new_tokens=32,
    cache_mode="deepseek_decode",
    deepseek_prefill_mode="parallel",
    do_sample=False,
    return_cache_stats=True,
)
```

`prompt` can be text, token ids, or a tensor. Text prompts require a tokenizer with `encode` and `decode` methods. Token-id prompts do not require a tokenizer.

## Main Pipeline

```text
inference_autoregresive(...)
    -> encode prompt
    -> generate(...)
        -> prefill(...)
        -> decode_step(...) repeated for new tokens
    -> optional decode back to text
```

The wrapper returns generated token ids, optional decoded text, cache statistics, timing metrics, and optional MTP draft diagnostics.

## Recommended Mode

Use this for DeepSeek-style HCA/CSA/hybrid models:

```python
cache_mode="deepseek_decode"
deepseek_prefill_mode="parallel"
```

This mode runs the prompt through the model once, captures each layer's normalized attention input, builds real HCA/CSA layer caches, and then decodes future tokens one at a time from those caches.

## Debug Mode

Use this when comparing cache behavior token by token:

```python
cache_mode="deepseek_decode"
deepseek_prefill_mode="sequential_debug"
```

This mode fills the cache by calling `forward_decode` once per prompt token. It is slower, but useful for validating cache transitions.

## CLI

The same generation path is exposed as:

```bash
python -m scripts.inference_cli generate \
  --checkpoint outputs/deepseekv4_mini_muon_last_manual.pt \
  --config-json outputs/deepseekv4_mini_muon_last_manual.json \
  --prompt "key key_1 is value_7 question : what is key_1 ? answer :" \
  --synthetic-tokenizer \
  --cache-mode deepseek_decode \
  --deepseek-prefill-mode parallel \
  --max-new-tokens 16 \
  --no-do-sample \
  --return-cache-stats
```

The installed console entry point is:

```bash
deepseekv4-infer generate ...
```
