from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from inference.cache_utils import resolve_cache_dtype
from inference.csa_cache import CSALayerCache
from inference.hca_cache import HCALayerCache
from inference.mha_cache import MHACache


@dataclass
class DeepSeekV4InferenceCache:
    layer_caches: list[Any]
    batch_size: int
    device: torch.device
    dtype: torch.dtype
    tokens_seen: int = 0
    sequence_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        for cache in self.layer_caches:
            cache.reset()
        self.tokens_seen = 0
        self.sequence_ids = None
        self.attention_mask = None
        self.metadata.clear()

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "DeepSeekV4InferenceCache":
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        for cache in self.layer_caches:
            cache.to(device=device, dtype=dtype)
        if self.sequence_ids is not None and device is not None:
            self.sequence_ids = self.sequence_ids.to(device=device)
        if self.attention_mask is not None and device is not None:
            self.attention_mask = self.attention_mask.to(device=device)
        return self

    def append_input_ids(
        self,
        input_ids_t: torch.Tensor,
        attention_mask_t: Optional[torch.Tensor] = None,
    ) -> None:
        input_ids_t = input_ids_t.to(device=self.device, dtype=torch.long)
        self.sequence_ids = (
            input_ids_t if self.sequence_ids is None else torch.cat([self.sequence_ids, input_ids_t], dim=1)
        )
        if attention_mask_t is not None:
            attention_mask_t = attention_mask_t.to(device=self.device)
            self.attention_mask = (
                attention_mask_t
                if self.attention_mask is None
                else torch.cat([self.attention_mask, attention_mask_t], dim=1)
            )
        self.tokens_seen += int(input_ids_t.shape[1])

    def crop_sequence(self, max_length: Optional[int]) -> None:
        if max_length is None:
            return
        if self.sequence_ids is not None and self.sequence_ids.shape[1] > max_length:
            self.sequence_ids = self.sequence_ids[:, -max_length:]
        if self.attention_mask is not None and self.attention_mask.shape[1] > max_length:
            self.attention_mask = self.attention_mask[:, -max_length:]
        for cache in self.layer_caches:
            if hasattr(cache, "crop"):
                cache.crop(max_length)

    def memory_bytes(self) -> int:
        total = sum(cache.memory_bytes() for cache in self.layer_caches)
        if self.sequence_ids is not None:
            total += int(self.sequence_ids.numel() * self.sequence_ids.element_size())
        if self.attention_mask is not None:
            total += int(self.attention_mask.numel() * self.attention_mask.element_size())
        return total

    def cache_summary(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        compressed_hca = 0
        compressed_csa = 0
        pending_hca = 0
        pending_csa = 0
        local_windows: list[int] = []

        for cache in self.layer_caches:
            name = type(cache).__name__
            by_type[name] = by_type.get(name, 0) + 1
            if isinstance(cache, HCALayerCache):
                compressed_hca += 0 if cache.compressed_kv is None else int(cache.compressed_kv.shape[1])
                pending_hca += 0 if cache.pending_c is None else int(cache.pending_c.shape[1])
                if cache.local_c is not None:
                    local_windows.append(int(cache.local_c.shape[1]))
            elif isinstance(cache, CSALayerCache):
                compressed_csa += 0 if cache.compressed_main is None else int(cache.compressed_main.shape[1])
                pending_csa += 0 if cache.pending_a_c is None else int(cache.pending_a_c.shape[1])
                if cache.local_c is not None:
                    local_windows.append(int(cache.local_c.shape[1]))

        return {
            "batch_size": self.batch_size,
            "tokens_seen": self.tokens_seen,
            "sequence_length": 0 if self.sequence_ids is None else int(self.sequence_ids.shape[1]),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "cache_mode": self.metadata.get("cache_mode", "unknown"),
            "active_decode": bool(self.metadata.get("active_decode", False)),
            "logits_from_cache": bool(self.metadata.get("logits_from_cache", False)),
            "cache_population": self.metadata.get("cache_population", "unknown"),
            "deepseek_active_decode": bool(
                self.metadata.get("cache_mode") == "deepseek_decode"
                and self.metadata.get("logits_from_cache", False)
            ),
            "num_layers": len(self.layer_caches),
            "layers_by_cache_type": by_type,
            "cache_memory_bytes": self.memory_bytes(),
            "cache_memory_mb": self.memory_bytes() / (1024 * 1024),
            "num_compressed_entries_hca": compressed_hca,
            "num_compressed_entries_csa": compressed_csa,
            "num_pending_tokens_hca": pending_hca,
            "num_pending_tokens_csa": pending_csa,
            "num_hca_compressed_entries": compressed_hca,
            "num_csa_compressed_main_entries": compressed_csa,
            "num_csa_compressed_index_entries": compressed_csa,
            "num_hca_pending_tokens": pending_hca,
            "num_csa_pending_tokens": pending_csa,
            "local_window_size": max(local_windows) if local_windows else 0,
        }


def _layer_cache_type(block: torch.nn.Module) -> str:
    return str(getattr(block, "attention_type", "mha"))


def build_inference_cache(
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    dtype: str | torch.dtype,
    local_window_size: Optional[int] = None,
) -> DeepSeekV4InferenceCache:
    cfg = getattr(model, "config", None)
    cache_dtype = resolve_cache_dtype(dtype)
    layer_caches: list[Any] = []

    for block in getattr(model, "blocks", []):
        attention_type = _layer_cache_type(block)
        if attention_type == "mha":
            layer_caches.append(MHACache())
        elif attention_type == "hca":
            layer_caches.append(
                HCALayerCache(
                    compression_factor=int(getattr(cfg, "hca_compression_factor", 1) or 1),
                    local_window_size=local_window_size or getattr(cfg, "window_size", None),
                )
            )
        elif attention_type == "csa":
            layer_caches.append(
                CSALayerCache(
                    compression_factor=int(getattr(cfg, "compression_factor", 1) or 1),
                    local_window_size=local_window_size or getattr(cfg, "window_size", None),
                )
            )
        else:
            raise ValueError(f"Unsupported attention type for inference cache: {attention_type!r}")

    return DeepSeekV4InferenceCache(
        layer_caches=layer_caches,
        batch_size=batch_size,
        device=device,
        dtype=cache_dtype,
        metadata={
            "cache_mode": "unconfigured",
            "active_decode": False,
            "logits_from_cache": False,
            "cache_population": "unconfigured",
        },
    )
