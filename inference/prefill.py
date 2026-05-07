from __future__ import annotations

from typing import Any, Optional

import torch

from inference.cache_utils import (
    normalize_position_ids,
    resolve_cache_dtype,
    resolve_device,
)
from inference.decode import configure_cache_metadata
from inference.decode import update_cache_with_token
from inference.hybrid_cache import DeepSeekV4InferenceCache, build_inference_cache
from inference.inference_config import InferenceConfig


@torch.no_grad()
def prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    inference_config: Optional[InferenceConfig] = None,
    return_aux: bool = False,
) -> dict[str, Any]:
    cfg = inference_config or InferenceConfig()
    cfg.validate()

    device = resolve_device(model, cfg.device)
    cache_dtype = resolve_cache_dtype(cfg.cache_dtype)
    input_ids = input_ids.to(device=device, dtype=torch.long)
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must have shape [B,T], got {tuple(input_ids.shape)}")
    if input_ids.shape[1] == 0:
        raise ValueError("prefill requires at least one prompt token.")

    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    batch_size, seq_len = input_ids.shape
    position_ids = normalize_position_ids(
        position_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    cache = build_inference_cache(
        model,
        batch_size=batch_size,
        device=device,
        dtype=cache_dtype,
        local_window_size=cfg.local_window_size,
    )
    configure_cache_metadata(cache, cfg)

    if cfg.cache_mode == "deepseek_decode" and cfg.deepseek_prefill_mode == "parallel":
        if not hasattr(model, "prefill_decode_cache"):
            raise NotImplementedError(
                "cache_mode='deepseek_decode' with deepseek_prefill_mode='parallel' "
                "requires model.prefill_decode_cache(...)."
            )
        out = model.prefill_decode_cache(
            input_ids=input_ids,
            cache=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inference_config=cfg,
            return_aux=return_aux,
        )
        if return_aux:
            aux = out.get("aux", {})
            aux["cache_summary"] = out["cache"].cache_summary()
            out["aux"] = aux
        return out

    if cfg.cache_mode in {"mha_decode", "deepseek_decode"}:
        if getattr(getattr(model, "config", None), "attention_type", None) != "mha":
            if cfg.cache_mode == "mha_decode":
                raise NotImplementedError("cache_mode='mha_decode' only supports attention_type='mha'.")
        logits_steps = []
        hidden_states = None
        aux = {}
        for idx in range(seq_len):
            mask_t = attention_mask[:, idx : idx + 1] if attention_mask is not None else None
            out = model.forward_decode(
                input_ids_t=input_ids[:, idx : idx + 1],
                cache=cache,
                position_ids_t=position_ids[:, idx : idx + 1],
                attention_mask_t=mask_t,
                return_aux=return_aux,
            )
            cache = out["cache"]
            logits_steps.append(out["logits"])
            hidden_states = out.get("hidden_states")
            aux = out.get("aux", {})

        full_logits = torch.cat(logits_steps, dim=1)
        if return_aux:
            aux["cache_summary"] = cache.cache_summary()
        return {
            "logits": full_logits[:, -1:, :],
            "full_logits": full_logits,
            "hidden_states": hidden_states,
            "cache": cache,
            "aux": aux if return_aux else {},
        }

    for idx in range(seq_len):
        mask_t = attention_mask[:, idx : idx + 1] if attention_mask is not None else None
        update_cache_with_token(
            model,
            input_ids_t=input_ids[:, idx : idx + 1],
            cache=cache,
            position_ids_t=position_ids[:, idx : idx + 1],
            attention_mask_t=mask_t,
            inference_config=cfg,
        )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_aux=return_aux,
    )
    aux = outputs.get("aux", {}) if return_aux else {}
    if return_aux:
        aux["cache_summary"] = cache.cache_summary()

    return {
        "logits": outputs["logits"][:, -1:, :],
        "full_logits": outputs["logits"],
        "hidden_states": outputs.get("hidden_states", None),
        "cache": cache,
        "aux": aux,
    }


def empty_cache_like_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    inference_config: Optional[InferenceConfig] = None,
) -> DeepSeekV4InferenceCache:
    cfg = inference_config or InferenceConfig()
    device = resolve_device(model, cfg.device)
    return build_inference_cache(
        model,
        batch_size=int(input_ids.shape[0]),
        device=device,
        dtype=resolve_cache_dtype(cfg.cache_dtype),
        local_window_size=cfg.local_window_size,
    )
