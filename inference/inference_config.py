from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceConfig:
    max_new_tokens: int = 128

    use_cache: bool = True
    cache_mode: str = "audit"
    cache_dtype: str = "fp32"
    device: str = "auto"

    # Generation
    do_sample: bool = True
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    # Cache behavior
    deepseek_prefill_mode: str = "parallel"
    deepseek_cache_population: str = "layer_projection_real"
    validate_deepseek_cache_equivalence: bool = False
    local_window_size: Optional[int] = None
    max_cache_length: Optional[int] = None
    compress_on_block_ready: bool = True

    # MTP-assisted diagnostics/draft decoding
    use_mtp_draft: bool = False
    mtp_accept_mode: str = "greedy"
    max_mtp_draft_tokens: Optional[int] = None

    # Debug
    return_cache_stats: bool = False
    validate_cache_shapes: bool = True

    # Current v1 keeps generation correct by full-context recomputation while
    # maintaining explicit DeepSeek-style caches for inspection.
    fallback_to_full_forward: bool = True

    def validate(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {self.max_new_tokens}")

        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k must be > 0 when provided, got {self.top_k}")

        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must satisfy 0 < top_p <= 1, got {self.top_p}")

        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ValueError(
                "repetition_penalty must be > 0 when provided, "
                f"got {self.repetition_penalty}"
            )

        if self.cache_dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"cache_dtype must be one of {{'fp32','bf16','fp16'}}, got {self.cache_dtype}")

        if self.cache_mode not in {"audit", "mha_decode", "deepseek_decode"}:
            raise ValueError(
                "cache_mode must be one of {'audit','mha_decode','deepseek_decode'}, "
                f"got {self.cache_mode!r}"
            )

        if self.deepseek_prefill_mode not in {"parallel", "sequential_debug"}:
            raise ValueError(
                "deepseek_prefill_mode must be one of {'parallel','sequential_debug'}, "
                f"got {self.deepseek_prefill_mode!r}"
            )

        if self.deepseek_cache_population not in {"layer_projection_real", "embedding_proxy_debug"}:
            raise ValueError(
                "deepseek_cache_population must be one of "
                "{'layer_projection_real','embedding_proxy_debug'}, "
                f"got {self.deepseek_cache_population!r}"
            )

        if self.local_window_size is not None and self.local_window_size <= 0:
            raise ValueError(
                f"local_window_size must be > 0 when provided, got {self.local_window_size}"
            )

        if self.max_cache_length is not None and self.max_cache_length <= 0:
            raise ValueError(
                f"max_cache_length must be > 0 when provided, got {self.max_cache_length}"
            )

        if self.max_mtp_draft_tokens is not None and self.max_mtp_draft_tokens <= 0:
            raise ValueError(
                "max_mtp_draft_tokens must be > 0 when provided, "
                f"got {self.max_mtp_draft_tokens}"
            )

        if self.mtp_accept_mode not in {"greedy", "match_main"}:
            raise ValueError(
                "mtp_accept_mode must be one of {'greedy','match_main'}, "
                f"got {self.mtp_accept_mode!r}"
            )
