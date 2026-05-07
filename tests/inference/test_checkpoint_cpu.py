from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from inference import InferenceConfig, audit_inference_pipeline, generate, inference_autoregresive
from inference.hybrid_cache import build_inference_cache
from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig


CHECKPOINT_JSON = Path("outputs/deepseekv4_mini_muon_last_manual.json")
CHECKPOINT_PT = Path("outputs/deepseekv4_mini_muon_last_manual.pt")


def load_checkpoint_model_cpu() -> DeepSeekV4LM:
    meta = json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
    cfg_dict = dict(meta["config"])
    if isinstance(cfg_dict.get("attention_pattern"), list):
        cfg_dict["attention_pattern"] = tuple(cfg_dict["attention_pattern"])
    if isinstance(cfg_dict.get("mtp_depth_loss_weights"), list):
        cfg_dict["mtp_depth_loss_weights"] = tuple(cfg_dict["mtp_depth_loss_weights"])

    checkpoint = torch.load(CHECKPOINT_PT, map_location="cpu")
    model = DeepSeekV4LM(DeepSeekV4LMConfig(**cfg_dict))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


pytestmark = pytest.mark.skipif(
    not CHECKPOINT_JSON.exists() or not CHECKPOINT_PT.exists(),
    reason="manual trained checkpoint is not available",
)


def test_manual_checkpoint_loads_on_cpu_and_builds_hybrid_cache():
    model = load_checkpoint_model_cpu()
    cache = build_inference_cache(model, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    cache_types = [type(layer_cache).__name__ for layer_cache in cache.layer_caches]

    assert next(model.parameters()).device.type == "cpu"
    assert cache_types == ["CSALayerCache", "HCALayerCache", "CSALayerCache", "HCALayerCache"]


def test_manual_checkpoint_generate_one_token_on_cpu():
    model = load_checkpoint_model_cpu()
    input_ids = torch.tensor([[1, 4, 5, 6, 7, 8]], dtype=torch.long)

    out = generate(
        model,
        input_ids,
        InferenceConfig(max_new_tokens=1, do_sample=False, return_cache_stats=True),
    )

    assert out["sequences"].shape == (1, 7)
    assert int(out["sequences"].min()) >= 0
    assert int(out["sequences"].max()) < model.config.vocab_size
    assert out["cache_stats"]["num_layers"] == model.config.n_layers


def test_manual_checkpoint_high_level_wrappers_on_cpu():
    model = load_checkpoint_model_cpu()

    generated = inference_autoregresive(
        model,
        prompt=[1, 4, 5, 6],
        max_new_tokens=1,
        do_sample=False,
        return_cache_stats=True,
    )
    audit = audit_inference_pipeline(
        model,
        prompt=[1, 4, 5, 6],
        max_new_tokens=1,
        do_sample=False,
        compare_logits=True,
    )

    assert generated["sequences"].shape == (1, 5)
    assert audit["cache_stats"]["num_layers"] == model.config.n_layers
    assert audit["full_vs_cached"]["allclose"]


def test_manual_checkpoint_deepseek_decode_runs_on_cpu_without_full_forward(monkeypatch):
    model = load_checkpoint_model_cpu()

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("deepseek_decode must not call full forward")

    monkeypatch.setattr(model, "forward", forbidden_forward)
    out = generate(
        model,
        torch.tensor([[1, 4, 5, 6]], dtype=torch.long),
        InferenceConfig(
            cache_mode="deepseek_decode",
            max_new_tokens=1,
            do_sample=False,
            return_cache_stats=True,
            use_mtp_draft=True,
        ),
    )

    assert out["sequences"].shape == (1, 5)
    assert out["cache_stats"]["deepseek_active_decode"] is True
    assert out["cache_stats"]["layers_by_cache_type"] == {"CSALayerCache": 2, "HCALayerCache": 2}
    assert out["mtp_drafts"][0]["draft_confidence"] is not None
