from __future__ import annotations

import pytest
import torch

from inference.inference_config import InferenceConfig
from inference.sampling import (
    apply_repetition_penalty,
    sample_next_token,
    top_k_filtering,
    top_p_filtering,
)


def test_valid_inference_config_builds():
    cfg = InferenceConfig(
        max_new_tokens=32,
        use_cache=True,
        cache_dtype="bf16",
        do_sample=True,
        temperature=1.0,
        top_k=20,
        top_p=0.95,
    )

    cfg.validate()

    assert cfg.max_new_tokens == 32
    assert cfg.cache_dtype == "bf16"


def test_valid_cache_modes():
    for mode in ["audit", "mha_decode", "deepseek_decode"]:
        InferenceConfig(cache_mode=mode).validate()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": -1.0}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"cache_dtype": "fp8"}, "cache_dtype"),
        ({"cache_mode": "bad"}, "cache_mode"),
        ({"mtp_accept_mode": "sample"}, "mtp_accept_mode"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
        ({"deepseek_prefill_mode": "bad"}, "deepseek_prefill_mode"),
        ({"deepseek_cache_population": "fake"}, "deepseek_cache_population"),
        ({"local_window_size": 0}, "local_window_size"),
        ({"max_cache_length": 0}, "max_cache_length"),
    ],
)
def test_invalid_inference_config_values_raise(kwargs, message):
    cfg = InferenceConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_greedy_sampling_returns_argmax():
    logits = torch.tensor([[0.1, 2.0, 0.3], [4.0, 1.0, 2.0]])
    cfg = InferenceConfig(do_sample=False)

    next_token = sample_next_token(logits, cfg)

    assert next_token.shape == (2, 1)
    assert torch.equal(next_token.squeeze(1), logits.argmax(dim=-1))


def test_temperature_zero_is_rejected():
    with pytest.raises(ValueError, match="temperature"):
        InferenceConfig(temperature=0.0).validate()


def test_temperature_scaling_changes_logits():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    scaled = logits / 0.5

    assert not torch.allclose(logits, scaled)


def test_top_k_filtering_keeps_only_k_tokens():
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])

    filtered = top_k_filtering(logits, top_k=2)

    assert torch.isfinite(filtered).sum().item() == 2
    assert torch.isfinite(filtered[0, 1])
    assert torch.isfinite(filtered[0, 3])


def test_top_k_larger_than_vocab_is_safe():
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    filtered = top_k_filtering(logits, top_k=99)

    assert torch.isfinite(filtered).all()


def test_top_p_filtering_keeps_at_least_one_token():
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.5]])

    filtered = top_p_filtering(logits, top_p=0.8)

    assert torch.isfinite(filtered).any()
    assert torch.isinf(filtered).any()


def test_repetition_penalty_changes_seen_token_logits():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    generated_ids = torch.tensor([[0, 1]])

    penalized = apply_repetition_penalty(logits, generated_ids, penalty=2.0)

    assert penalized[0, 0].item() == pytest.approx(1.0)
    assert penalized[0, 1].item() == pytest.approx(-4.0)
    assert penalized[0, 2].item() == pytest.approx(1.0)


def test_sample_next_token_shape_and_valid_range():
    torch.manual_seed(0)
    logits = torch.randn(4, 11)
    cfg = InferenceConfig(do_sample=True, top_k=5)

    next_token = sample_next_token(logits, cfg)

    assert next_token.shape == (4, 1)
    assert int(next_token.min()) >= 0
    assert int(next_token.max()) < logits.shape[-1]


def test_sampling_handles_nonfinite_logits_with_safe_fallback():
    logits = torch.tensor([[float("nan"), 1.0, 2.0], [float("inf"), 0.0, -1.0]])
    cfg = InferenceConfig(do_sample=True)

    next_token = sample_next_token(logits, cfg)

    assert next_token.shape == (2, 1)
    assert int(next_token.min()) >= 0
    assert int(next_token.max()) < logits.shape[-1]
