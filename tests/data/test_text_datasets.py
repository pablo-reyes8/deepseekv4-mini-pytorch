import pytest
import torch

from data.text_datasets import (
    CausalTextDataset,
    TextDataloaderConfig,
    available_hf_text_dataset_presets,
    create_text_dataloaders,
    resolve_hf_text_preset,
    train_byte_level_bpe_tokenizer,
)


def test_dataset_presets_include_expected_small_corpora():
    presets = set(available_hf_text_dataset_presets())
    assert {"wikitext2", "tinystories", "ag_news", "imdb", "minipile"}.issubset(presets)
    assert resolve_hf_text_preset("wikitext2").dataset_name == "Salesforce/wikitext"


def test_causal_text_dataset_builds_input_label_blocks():
    pytest.importorskip("tokenizers")

    texts = [
        "DeepSeek V4 mini tests causal language modeling.",
        "Compressed sparse attention and mixture of experts need clean data loaders.",
        "Training should be reproducible and small enough for CPU smoke tests.",
    ]
    tokenizer = train_byte_level_bpe_tokenizer(texts, vocab_size=128, min_frequency=1)

    dataset = CausalTextDataset(texts, tokenizer, block_size=12)
    sample = dataset[0]

    assert set(sample) == {"input_ids", "labels"}
    assert sample["input_ids"].shape == (12,)
    assert sample["labels"].shape == (12,)
    assert sample["input_ids"].dtype == torch.long
    assert sample["labels"].dtype == torch.long


def test_text_dataloader_config_forwards_to_hf_loader(monkeypatch):
    calls = {}

    def fake_create_hf_text_dataloaders(preset_name, **kwargs):
        calls["preset_name"] = preset_name
        calls.update(kwargs)
        return "train", "val", "tokenizer"

    monkeypatch.setattr(
        "data.text_datasets.create_hf_text_dataloaders",
        fake_create_hf_text_dataloaders,
    )

    cfg = TextDataloaderConfig(
        preset_name="ag_news",
        block_size=64,
        batch_size=4,
        max_train_documents=100,
        max_validation_documents=20,
    )
    train_loader, val_loader, tokenizer = create_text_dataloaders(cfg=cfg, use_mtp=True)

    assert (train_loader, val_loader, tokenizer) == ("train", "val", "tokenizer")
    assert calls["preset_name"] == "ag_news"
    assert calls["block_size"] == 64
    assert calls["batch_size"] == 4
    assert calls["max_train_documents"] == 100
