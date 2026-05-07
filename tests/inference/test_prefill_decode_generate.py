from __future__ import annotations

import pytest
import torch
import importlib

from inference import InferenceConfig, generate, inference_autoregresive, prefill
from inference.audit import audit_inference_pipeline
from inference.decode import decode_step, prepare_input_ids, update_cache_with_token
from inference.hybrid_cache import build_inference_cache
from inference.metrics import cache_summary, compare_full_vs_cached_logits
from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig


def make_config(**overrides) -> DeepSeekV4LMConfig:
    cfg = dict(
        vocab_size=64,
        d_model=16,
        n_layers=2,
        max_seq_len=16,
        pad_token_id=0,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        attention_type="mha",
        n_heads=2,
        head_dim=8,
        rotary_dim=8,
        compression_factor=2,
        hca_compression_factor=2,
        window_size=2,
        top_k_blocks=1,
        indexer_dim=8,
        n_indexer_heads=2,
        query_compression_dim=8,
        use_grouped_output_projection=False,
        use_indexer_score_bias=False,
        use_separate_local_kv=True,
        ffn_type="dense",
        mlp_hidden_dim=32,
        mlp_dropout=0.0,
        num_experts=2,
        top_k_experts=1,
        expert_hidden_dim=32,
        shared_experts=1,
        shared_hidden_dim=32,
        balance_loss_weight=0.0,
        sequence_balance_loss_weight=0.0,
        router_jitter_noise=0.0,
        use_mhc=False,
        n_hc=2,
        mhc_sinkhorn_iters=2,
        use_mtp=False,
        mtp_depth=2,
        mtp_hidden_dim=16,
        mtp_dropout=0.0,
    )
    cfg.update(overrides)
    return DeepSeekV4LMConfig(**cfg)


def make_model(**overrides) -> DeepSeekV4LM:
    torch.manual_seed(123)
    model = DeepSeekV4LM(make_config(**overrides))
    model.eval()
    return model


def ids(batch: int = 2, seq: int = 5, vocab: int = 64) -> torch.Tensor:
    return torch.randint(1, vocab, (batch, seq), dtype=torch.long)


@pytest.mark.parametrize(
    ("attention_type", "cache_name"),
    [("mha", "MHACache"), ("hca", "HCALayerCache"), ("csa", "CSALayerCache")],
)
def test_hybrid_cache_builds_correct_layer_types(attention_type, cache_name):
    model = make_model(attention_type=attention_type)
    cache = build_inference_cache(model, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

    assert len(cache.layer_caches) == model.config.n_layers
    assert all(type(layer_cache).__name__ == cache_name for layer_cache in cache.layer_caches)
    assert cache.cache_summary()["num_layers"] == model.config.n_layers


def test_hybrid_cache_tokens_to_memory_summary_and_to():
    model = make_model(attention_type="mha")
    cache = build_inference_cache(model, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    x = ids(seq=1)

    update_cache_with_token(model, x, cache, inference_config=InferenceConfig())
    cache.to(device=torch.device("cpu"), dtype=torch.float16)
    summary = cache.cache_summary()

    assert cache.tokens_seen == 1
    assert cache.memory_bytes() > 0
    assert summary["tokens_seen"] == 1
    assert summary["cache_memory_mb"] >= 0
    assert "layers_by_cache_type" in summary
    assert summary["cache_mode"] == "audit"
    assert summary["active_decode"] is False
    assert summary["logits_from_cache"] is False
    assert summary["cache_population"] == "embedding_proxy"


def test_prefill_mha_hca_csa_builds_expected_cache_state():
    prompt = ids(seq=5)

    mha = prefill(make_model(attention_type="mha"), prompt, inference_config=InferenceConfig())
    assert mha["logits"].shape == (2, 1, 64)
    assert mha["cache"].layer_caches[0].k.shape[2] == 5

    hca = prefill(make_model(attention_type="hca"), prompt, inference_config=InferenceConfig())
    hca_cache = hca["cache"].layer_caches[0]
    assert hca_cache.compressed_kv.shape[1] == 2
    assert hca_cache.pending_c.shape[1] == 1
    assert hca_cache.local_c.shape[1] == 2

    csa = prefill(make_model(attention_type="csa"), prompt, inference_config=InferenceConfig())
    csa_cache = csa["cache"].layer_caches[0]
    assert csa_cache.compressed_main.shape[1] == 2
    assert csa_cache.compressed_index.shape[1] == 2
    assert csa_cache.pending_a_c.shape[1] == 1
    assert csa_cache.local_c.shape[1] == 2


@pytest.mark.parametrize("attention_type", ["mha", "hca", "csa"])
def test_decode_step_shape_updates_cache_and_no_nan(attention_type):
    model = make_model(attention_type=attention_type)
    cache = build_inference_cache(model, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    cfg = InferenceConfig(do_sample=False)

    out = decode_step(model, ids(seq=1), cache, inference_config=cfg, return_aux=True)

    assert out["logits"].shape == (2, 1, 64)
    assert torch.isfinite(out["logits"]).all()
    assert out["cache"].tokens_seen == 1
    assert "cache_summary" in out["aux"]


def test_mha_decode_step_active_cache_shape_growth_and_metadata():
    model = make_model(attention_type="mha")
    cache = build_inference_cache(model, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    cfg = InferenceConfig(cache_mode="mha_decode", do_sample=False)

    for step in range(3):
        out = decode_step(model, ids(seq=1), cache, inference_config=cfg, return_aux=True)
        cache = out["cache"]
        assert out["logits"].shape == (2, 1, 64)
        assert cache.tokens_seen == step + 1
        assert all(layer_cache.k.shape[2] == step + 1 for layer_cache in cache.layer_caches)
        assert all(layer_cache.v.shape[2] == step + 1 for layer_cache in cache.layer_caches)

    summary = cache.cache_summary()
    assert summary["cache_mode"] == "mha_decode"
    assert summary["active_decode"] is True
    assert summary["logits_from_cache"] is True
    assert summary["cache_population"] == "active_mha_kv"


def test_mha_decode_mode_rejects_non_mha_and_deepseek_decode_raises():
    hca_model = make_model(attention_type="hca")
    hca_cache = build_inference_cache(hca_model, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="mha_decode"):
        decode_step(hca_model, ids(seq=1), hca_cache, inference_config=InferenceConfig(cache_mode="mha_decode"))


@pytest.mark.parametrize("attention_type", ["hca", "csa"])
def test_deepseek_decode_step_shape_metadata_and_real_cache(attention_type):
    model = make_model(attention_type=attention_type)
    cache = build_inference_cache(model, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    cfg = InferenceConfig(cache_mode="deepseek_decode", do_sample=False)

    out = decode_step(model, ids(seq=1), cache, inference_config=cfg, return_aux=True)
    summary = out["cache"].cache_summary()

    assert out["logits"].shape == (2, 1, 64)
    assert torch.isfinite(out["logits"]).all()
    assert summary["cache_mode"] == "deepseek_decode"
    assert summary["active_decode"] is True
    assert summary["logits_from_cache"] is True
    assert summary["cache_population"] == "layer_projection_real"
    assert summary["deepseek_active_decode"] is True


def test_deepseek_decode_hybrid_csa_hca_moe_mhc_mtp_runs_without_full_forward(monkeypatch):
    model = make_model(
        attention_type="hybrid",
        attention_pattern=("csa", "hca"),
        ffn_type="moe",
        use_mhc=True,
        use_mtp=True,
        hca_compression_factor=2,
        compression_factor=2,
    )

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("deepseek_decode must not call full forward")

    monkeypatch.setattr(model, "forward", forbidden_forward)
    out = generate(
        model,
        ids(batch=1, seq=4),
        InferenceConfig(
            cache_mode="deepseek_decode",
            deepseek_prefill_mode="sequential_debug",
            max_new_tokens=2,
            do_sample=False,
            return_cache_stats=True,
            use_mtp_draft=True,
        ),
    )

    assert out["sequences"].shape == (1, 6)
    assert out["cache_stats"]["deepseek_active_decode"] is True
    assert out["cache_stats"]["num_csa_compressed_main_entries"] > 0
    assert out["cache_stats"]["num_hca_compressed_entries"] > 0
    assert out["mtp_drafts"][0]["is_speculative_decode"] is False


def test_deepseek_parallel_prefill_does_not_loop_forward_decode(monkeypatch):
    model = make_model(attention_type="hca", hca_compression_factor=2)
    calls = {"forward_decode": 0}
    original = model.forward_decode

    def counted_forward_decode(*args, **kwargs):
        calls["forward_decode"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward_decode", counted_forward_decode)
    out = prefill(
        model,
        ids(batch=1, seq=5),
        inference_config=InferenceConfig(
            cache_mode="deepseek_decode",
            deepseek_prefill_mode="parallel",
            return_cache_stats=True,
        ),
        return_aux=True,
    )

    assert calls["forward_decode"] == 0
    assert out["cache"].tokens_seen == 5
    assert out["cache"].cache_summary()["cache_population"] == "layer_projection_real"


def test_deepseek_sequential_debug_prefill_uses_forward_decode(monkeypatch):
    model = make_model(attention_type="csa", compression_factor=2)
    prompt = ids(batch=1, seq=5)
    calls = {"forward_decode": 0}
    original = model.forward_decode

    def counted_forward_decode(*args, **kwargs):
        calls["forward_decode"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward_decode", counted_forward_decode)
    out = prefill(
        model,
        prompt,
        inference_config=InferenceConfig(
            cache_mode="deepseek_decode",
            deepseek_prefill_mode="sequential_debug",
        ),
    )

    assert calls["forward_decode"] == prompt.shape[1]
    assert out["cache"].tokens_seen == prompt.shape[1]


def test_hca_parallel_prefill_cache_counts_from_full_sequence():
    model = make_model(
        attention_type="hca",
        hca_compression_factor=4,
        window_size=3,
        max_seq_len=32,
    )
    out = prefill(
        model,
        ids(batch=1, seq=17),
        inference_config=InferenceConfig(cache_mode="deepseek_decode"),
    )

    for layer_cache in out["cache"].layer_caches:
        assert layer_cache.compressed_kv.shape[1] == 4
        assert layer_cache.pending_c.shape[1] == 1
        assert layer_cache.local_c.shape[1] == 3
        assert layer_cache.tokens_seen == 17


def test_csa_parallel_prefill_cache_counts_from_full_sequence():
    model = make_model(
        attention_type="csa",
        compression_factor=4,
        window_size=3,
        max_seq_len=32,
    )
    out = prefill(
        model,
        ids(batch=1, seq=17),
        inference_config=InferenceConfig(cache_mode="deepseek_decode"),
    )

    for layer_cache in out["cache"].layer_caches:
        assert layer_cache.compressed_main.shape[1] == 4
        assert layer_cache.compressed_index.shape[1] == 4
        assert layer_cache.pending_a_c.shape[1] == 1
        assert layer_cache.local_c.shape[1] == 3
        assert layer_cache.tokens_seen == 17


@pytest.mark.parametrize("attention_type", ["hca", "csa"])
def test_deepseek_parallel_prefill_decode_matches_sequential_debug(attention_type):
    model = make_model(
        attention_type=attention_type,
        hca_compression_factor=2,
        compression_factor=2,
        max_seq_len=16,
        n_layers=1,
    )
    prompt = ids(batch=1, seq=6)
    next_token = ids(batch=1, seq=1)

    parallel = prefill(
        model,
        prompt,
        inference_config=InferenceConfig(
            cache_mode="deepseek_decode",
            deepseek_prefill_mode="parallel",
        ),
    )
    sequential = prefill(
        model,
        prompt,
        inference_config=InferenceConfig(
            cache_mode="deepseek_decode",
            deepseek_prefill_mode="sequential_debug",
        ),
    )

    parallel_step = decode_step(
        model,
        next_token,
        parallel["cache"],
        inference_config=InferenceConfig(cache_mode="deepseek_decode"),
    )
    sequential_step = decode_step(
        model,
        next_token,
        sequential["cache"],
        inference_config=InferenceConfig(cache_mode="deepseek_decode"),
    )

    assert torch.allclose(parallel_step["logits"], sequential_step["logits"], atol=1e-5, rtol=1e-5)


def test_decode_step_with_moe_mhc_and_mtp_enabled():
    moe = make_model(attention_type="mha", ffn_type="moe", router_type="learned")
    moe_cache = build_inference_cache(moe, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    assert decode_step(moe, ids(seq=1), moe_cache, inference_config=InferenceConfig())["logits"].shape == (2, 1, 64)

    mhc = make_model(attention_type="mha", use_mhc=True)
    mhc_cache = build_inference_cache(mhc, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    assert decode_step(mhc, ids(seq=1), mhc_cache, inference_config=InferenceConfig())["logits"].shape == (2, 1, 64)

    mtp = make_model(attention_type="mha", use_mtp=True)
    mtp_cache = build_inference_cache(mtp, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    assert decode_step(mtp, ids(seq=1), mtp_cache, inference_config=InferenceConfig(use_mtp_draft=False))["logits"].shape == (2, 1, 64)


def test_decode_step_rejects_wrong_shapes_and_bad_position_ids():
    model = make_model()
    cache = build_inference_cache(model, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

    with pytest.raises(ValueError, match="input_ids_t"):
        decode_step(model, torch.ones(2, dtype=torch.long), cache)

    with pytest.raises(ValueError, match="input_ids_t"):
        decode_step(model, torch.ones(2, 2, dtype=torch.long), cache)

    with pytest.raises(ValueError, match="position_ids"):
        decode_step(model, ids(seq=1), cache, position_ids_t=torch.arange(3))


@pytest.mark.parametrize("attention_type", ["mha", "hca", "csa"])
def test_generate_greedy_shape_valid_tokens_for_attention_types(attention_type):
    model = make_model(attention_type=attention_type)
    prompt = ids(batch=2, seq=4)

    out = generate(
        model,
        prompt,
        InferenceConfig(max_new_tokens=3, do_sample=False, return_cache_stats=True),
    )

    assert out["sequences"].shape == (2, 7)
    assert int(out["sequences"].min()) >= 0
    assert int(out["sequences"].max()) < model.config.vocab_size
    assert out["cache_stats"]["cache_memory_mb"] >= 0
    assert out["num_generated_tokens"] == 3


def test_generate_mha_decode_mode_runs_from_active_cache():
    model = make_model(attention_type="mha")
    prompt = ids(batch=2, seq=4)

    out = generate(
        model,
        prompt,
        InferenceConfig(cache_mode="mha_decode", max_new_tokens=3, do_sample=False, return_cache_stats=True),
    )

    assert out["sequences"].shape == (2, 7)
    assert out["cache_stats"]["cache_mode"] == "mha_decode"
    assert out["cache_stats"]["logits_from_cache"] is True
    assert out["cache"].tokens_seen == 7


@pytest.mark.parametrize("attention_type", ["hca", "csa"])
def test_generate_deepseek_decode_mode_runs(attention_type):
    model = make_model(attention_type=attention_type)
    prompt = ids(batch=2, seq=4)

    out = generate(
        model,
        prompt,
        InferenceConfig(cache_mode="deepseek_decode", max_new_tokens=3, do_sample=False, return_cache_stats=True),
    )

    assert out["sequences"].shape == (2, 7)
    assert int(out["sequences"].min()) >= 0
    assert int(out["sequences"].max()) < model.config.vocab_size
    assert out["cache_stats"]["cache_mode"] == "deepseek_decode"
    assert out["cache_stats"]["logits_from_cache"] is True


def test_generate_zero_new_tokens_returns_prompt_and_batch_attention_mask():
    model = make_model()
    prompt = ids(batch=2, seq=4)
    prompt[0, -1] = 0
    attention_mask = prompt.ne(0).long()

    out = generate(
        model,
        prompt,
        InferenceConfig(max_new_tokens=0, do_sample=False),
        attention_mask=attention_mask,
    )

    assert torch.equal(out["sequences"], prompt)
    assert out["num_generated_tokens"] == 0


def test_generate_sampling_modes_return_valid_tokens():
    model = make_model()
    prompt = ids(batch=2, seq=3)

    for cfg in [
        InferenceConfig(max_new_tokens=2, do_sample=True, top_k=5),
        InferenceConfig(max_new_tokens=2, do_sample=True, top_p=0.9),
        InferenceConfig(max_new_tokens=2, do_sample=True, temperature=0.7),
    ]:
        out = generate(model, prompt, cfg)
        assert out["sequences"].shape == (2, 5)
        assert int(out["sequences"].max()) < model.config.vocab_size


def test_generate_stops_on_eos_when_sampled(monkeypatch):
    model = make_model()
    prompt = ids(batch=2, seq=3)

    def fake_sample(logits, config, generated_ids=None):
        return torch.full((logits.shape[0], 1), 2, device=logits.device, dtype=torch.long)

    generate_module = importlib.import_module("inference.generate")
    monkeypatch.setattr(generate_module, "sample_next_token", fake_sample)
    out = generate(model, prompt, InferenceConfig(max_new_tokens=5, do_sample=False, eos_token_id=2))

    assert out["sequences"].shape[1] == prompt.shape[1] + 1
    assert out["num_generated_tokens"] == 1


def test_generate_complex_csa_moe_mhc_mtp_and_mtp_drafts():
    model = make_model(
        attention_type="csa",
        ffn_type="moe",
        use_mhc=True,
        use_mtp=True,
        mtp_depth=2,
    )

    out = generate(
        model,
        ids(batch=1, seq=4),
        InferenceConfig(max_new_tokens=2, do_sample=False, use_mtp_draft=True, return_cache_stats=True),
    )

    assert out["sequences"].shape == (1, 6)
    assert out["mtp_drafts"]
    assert out["mtp_drafts"][0]["draft_token_ids"].shape[-1] <= 2
    assert out["mtp_drafts"][0]["draft_confidence"].shape == out["mtp_drafts"][0]["draft_token_ids"].shape
    assert out["mtp_drafts"][0]["is_speculative_decode"] is False
    assert int(out["mtp_drafts"][0]["draft_token_ids"].max()) < model.config.vocab_size


@pytest.mark.parametrize("attention_type", ["mha", "hca", "csa"])
def test_full_vs_cached_logits_match(attention_type):
    model = make_model(attention_type=attention_type)
    prompt = ids(batch=1, seq=4)

    result = compare_full_vs_cached_logits(
        model,
        prompt,
        inference_config=InferenceConfig(do_sample=False),
        atol=1e-5,
        rtol=1e-5,
    )

    assert result["allclose"]
    assert result["max_abs_diff"] <= 1e-5


def test_full_vs_active_mha_cached_logits_match():
    model = make_model(attention_type="mha")
    prompt = ids(batch=1, seq=5)

    result = compare_full_vs_cached_logits(
        model,
        prompt,
        inference_config=InferenceConfig(cache_mode="mha_decode", do_sample=False),
        atol=1e-5,
        rtol=1e-5,
    )

    assert result["allclose"]
    assert result["cache_summary"]["logits_from_cache"] is True


def test_inference_autoregresive_and_audit_wrapper_list_prompt():
    model = make_model()

    out = inference_autoregresive(
        model,
        prompt=[1, 2, 3],
        max_new_tokens=2,
        do_sample=False,
        return_cache_stats=True,
    )
    audit = audit_inference_pipeline(
        model,
        prompt=[1, 2, 3],
        max_new_tokens=1,
        do_sample=False,
        compare_logits=True,
    )

    assert out["sequences"].shape == (1, 5)
    assert out["text"] is None
    assert audit["generation"]["sequences"].shape == (1, 4)
    assert audit["full_vs_cached"]["allclose"]


def test_audit_wrapper_accepts_text_prompt_with_tokenizer():
    class TinyTokenizer:
        def __init__(self):
            self.vocab = {"hello": 1, "world": 2}
            self.inv = {1: "hello", 2: "world"}

        def encode(self, text):
            return [self.vocab.get(part, 1) for part in text.split()]

        def decode(self, ids):
            return " ".join(self.inv.get(int(i), "<unk>") for i in ids)

    model = make_model(vocab_size=8)
    tokenizer = TinyTokenizer()

    out = audit_inference_pipeline(
        model,
        prompt="hello world",
        tokenizer=tokenizer,
        max_new_tokens=1,
        do_sample=False,
        compare_logits=True,
    )

    assert out["input_ids"].shape == (1, 2)
    assert isinstance(out["text"], str)


def test_prepare_input_ids_moves_to_model_device():
    model = make_model()
    prepared = prepare_input_ids(torch.tensor([[1, 2]]), model, InferenceConfig(device="cpu"))

    assert prepared.device.type == "cpu"
    assert prepared.dtype == torch.long


def test_cache_summary_function_accepts_none_and_cache():
    assert cache_summary(None) == {}

    model = make_model()
    cache = build_inference_cache(model, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    summary = cache_summary(cache)

    assert summary["tokens_seen"] == 0
    assert summary["num_layers"] == model.config.n_layers
