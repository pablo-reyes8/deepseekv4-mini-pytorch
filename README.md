<p align="center">
  <img src="assets\header_image.png" width="1000"/>
</p>



# DeepSeek-V4 Mini: A Paper-Faithful PyTorch Research Implementation

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Status](https://img.shields.io/badge/status-active_research-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An unofficial, ground-up PyTorch implementation of the core architectural ideas behind **DeepSeek-V4**. The project scales the system down for readable code, CPU-safe tests, controlled ablations, and fast research iteration.

This repository is not a toy Transformer wrapper or a production model clone. It implements the mechanisms that make the DeepSeek-V4 report technically interesting as a system: hybrid compressed attention, sparse long-context retrieval, Mixture-of-Experts routing, manifold-constrained hyper-connections, multi-token prediction, and Muon-based training.

> [NOTE]
> This project is not affiliated with DeepSeek-AI. It does not reproduce the official DeepSeek-V4 weights, training data, distributed infrastructure, or production kernels. Its goal is architectural transparency and research-oriented experimentation.

## Index

- [🎯 Why This Repo Exists](#-why-this-repo-exists)
- [Architecture Coverage](#architecture-coverage)
- [🏗️ Repository Layout](#️-repository-layout)
- [Installation](#installation)
- [Run Tests](#run-tests)
- [⚙️ Model Configs](#️-model-configs)
- [📚 Dataset Presets](#-dataset-presets)
- [🔬 Training A Tiny Model](#-training-a-tiny-model)
- [Training With Batches and Indexing](#training-with-batches-and-indexing)
- [Docker Support](#docker-support)
- [🛠️ Command Line Tools](#️-command-line-tools)
- [CI Strategy](#ci-strategy)
- [Notes on Scope](#notes-on-scope)
- [📖 References & Citation](#-references--citation)

## 🎯 Why This Repo Exists

DeepSeek-V4 pushes the Transformer in three directions that demand independent study:

1. **Context Limits:** Long context needs something better than naive full attention.
2. **Model Capacity:** Scaling requires sparse activation algorithms, not just dense parameter scaling.
3. **Training Stability:** Deep training stability necessitates complex residual routing and optimization machinery, not only a bigger model.

This project isolates those innovations into a mini implementation where each component can be tested, ablated, and trained on small corpora before scaling.

## Architecture Coverage

| Area | Implementation Status |
| :--- | :--- |
| **Causal Transformer** | Token embeddings, RMSNorm, RoPE, MHA, LM head |
| **HCA (Hybrid Context)**| Compressed KV branch, sliding window branch, causal tests |
| **CSA (Compressed Sparse)**| Compressed sparse block selection, local window, indexer, causal tests |
| **MoE (Mixture of Experts)**| Learned/hash routing, top-k experts, shared experts, balance metrics |
| **mHC (Hyper-Connections)**| Stream expansion, Sinkhorn mixing, modular block API |
| **MTP (Multi-Token)** | Auxiliary next-n-token heads and prediction loss |
| **Training Engine** | AdamW groups, Muon+AdamW, cosine schedule, AMP, EMA, checkpoints, metrics |
| **Data Pipelines** | Synthetic retrieval, TinyStories, WikiText-2, AG News, IMDB, MiniPile, FineWeb-Edu |

## 🏗️ Repository Layout

```text
src/                  # Core model components and architecture
src/transformer_modules/ # Isolated attention and MoE blocks
training/             # Training loop, schedulers, optimizers, metrics, checkpoints
data/                 # Dataset builders and causal LM dataloaders
tests/                # CPU-safe component unit tests
tests/training/       # Training-stack integration tests
config/               # YAML experiment profiles
.github/              # Path-aware CI and Dependabot configurations
paper/                # DeepSeek-V4 paper reference
proyect_structure/    # Project scope and implementation guide
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,data]"
```

Minimal install for inference only:
```bash
pip install -r requirements.txt
```

## Run Tests

The repository includes a comprehensive CPU-safe test suite. The CUDA-only checks correctly skip when no GPU is available.

Full local CPU suite:
```bash
pytest
```

Training-only tests:
```bash
pytest tests/training
```

Dataset loader tests:
```bash
pytest tests/data
```

*Current validation on CPU: `649 passed, 4 skipped`*

## ⚙️ Model Configs

Start from the YAML profiles in `config/model/`. These profiles allow you to seamlessly switch between standard dense models and full DeepSeek architectures.

| Config | Purpose |
| :--- | :--- |
| `deepseekv4_tiny.yaml` | CPU smoke model for CI and debugging |
| `deepseekv4_mini.yaml` | Default research model with Hybrid Attention + MoE + mHC + MTP |
| `deepseekv4_csa_moe_mhc_mtp.yaml` | Full-feature integration variant |

**Typical tiny model shape (`deepseekv4_tiny.yaml`):**
```yaml
model:
  vocab_size: 128
  d_model: 32
  n_layers: 1
  max_seq_len: 32
  attention_type: mha
  ffn_type: dense
```

**Mini research profile (`deepseekv4_mini.yaml`):**
```yaml
model:
  d_model: 256
  n_layers: 6
  attention_type: hybrid
  attention_pattern: [csa, hca]
  ffn_type: moe
  num_experts: 8
  top_k_experts: 2
  use_mhc: true
  use_mtp: true
```

## 📚 Dataset Presets

The project supports a robust set of small-to-medium text corpora through `data/text_datasets.py`:

| Preset | HF Dataset | Primary Use Case |
| :--- | :--- | :--- |
| `synthetic_long_context`| Local generator | Retrieval stress tests for CSA/HCA |
| `tinystories` | `roneneldan/TinyStories` | Tiny LM generation & curriculum training |
| `wikitext2` | `Salesforce/wikitext` | Classic language modeling benchmark |
| `ag_news` | `fancyzhx/ag_news` | Compact news-domain corpus |
| `imdb` | `stanfordnlp/imdb` | Longer review text and domain shift |
| `minipile` | `JeanKaddour/minipile` | Diverse small pretraining mix |
| `fineweb_edu_10bt_mincols`| `EliMC/fineweb-edu-10BT` | Educational web sample (local limits) |

The generic loader returns dict batches shaped for the training pipeline:

```python
from data.text_datasets import create_hf_text_dataloaders

train_loader, val_loader, tokenizer = create_hf_text_dataloaders(
    "wikitext2",
    block_size=256,
    batch_size=8,
    vocab_size=16_000,
    max_tokenizer_documents=50_000,
    max_train_documents=20_000,
    max_validation_documents=2_000,
)

# Batch structure:
# {
#     "input_ids": LongTensor[B, T],
#     "labels": LongTensor[B, T],
# }
```

## 🔬 Training A Tiny Model

The high-level API is `training.train_deepseek.train_deepseekv4`. A minimal CPU smoke run looks like this:

```python
from data.text_datasets import create_hf_text_dataloaders
from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig
from training.train_deepseek import train_deepseekv4

train_loader, val_loader, tokenizer = create_hf_text_dataloaders(
    "wikitext2",
    block_size=64,
    batch_size=4,
    vocab_size=4096,
    max_tokenizer_documents=1000,
    max_train_documents=1000,
    max_validation_documents=200,
)

model = DeepSeekV4LM(
    DeepSeekV4LMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=64,
        n_layers=2,
        max_seq_len=64,
        pad_token_id=tokenizer.token_to_id("<pad>"),
        attention_type="hca",
        n_heads=4,
        head_dim=16,
        rotary_dim=16,
        ffn_type="dense",
        mlp_hidden_dim=128,
    )
)

history = train_deepseekv4(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cpu",
    amp_enabled=False,
    optimizer_type="adamw",
    learning_rate=3e-4,
    epochs=1,
    max_batches_per_epoch=10,
    eval_max_batches=5,
    ckpt_dir="checkpoints/wikitext2_tiny",
)
```

## Training With Batches and Indexing

For quick iteration and architectural debugging, you can limit the number of documents used to build blocks:

```python
train_loader, val_loader, tokenizer = create_hf_text_dataloaders(
    "ag_news",
    block_size=128,
    batch_size=16,
    max_train_documents=5000,
    max_validation_documents=1000,
)

for step, batch in enumerate(train_loader):
    input_ids = batch["input_ids"]  # [B, T]
    labels = batch["labels"]        # [B, T]
    if step == 0:
        print(input_ids.shape, labels.shape)
    break
```

For component debugging, the **synthetic retrieval dataset** is highly recommended because it explicitly exposes controlled long-range key/value dependencies:

```python
from data.syntethic_long_context_retrieval import (
    SyntheticRetrievalConfig,
    create_synthetic_retrieval_dataloaders,
)

cfg = SyntheticRetrievalConfig(
    block_size=256,
    min_filler_tokens=64,
    max_filler_tokens=220,
    batch_size=8,
)

train_loader, val_loader, tokenizer = create_synthetic_retrieval_dataloaders(cfg)
```

## Docker Support

```bash
docker build -t deepseekv4-mini .
docker compose run --rm tests
```

## 🛠️ Command Line Tools

After installing with `pip install -e ".[dev,data]"`, the project exposes three transparent CLIs for immediate interaction:

```bash
deepseekv4-data presets
deepseekv4-data synthetic-inspect --block-size 32 --batch-size 2
deepseekv4-train smoke --attention hca --ffn dense --max-batches 2
deepseekv4-inspect model-summary --attention csa --ffn moe
deepseekv4-inspect module-tests csa --quiet
```

The same commands work natively without CLI installation through Python modules:

```bash
python -m scripts.data_cli synthetic-inspect --block-size 32 --batch-size 2
python -m scripts.train_cli smoke --attention mha --ffn dense --max-batches 1 --quiet
python -m scripts.inspect_cli module-tests training --quiet
```

**CLI Scope:**
- `data_cli`: List presets, inspect synthetic data, and download HF text presets.
- `train_cli`: Run a tiny synthetic training smoke test with configurable attention/FFN/mHC/MTP.
- `inspect_cli`: Summarize model parameter structure and execute targeted module tests.

## CI Strategy

Continuous Integration is strictly path-aware to ensure speed without losing critical coverage:
- Changes in `src/`, configs, or packaging trigger **model & component tests**.
- Changes in `training/` or `tests/training/` trigger **training-stack tests**.
- Changes in `data/` or `tests/data/` trigger **dataset loader tests**.
- *All* changes run a lightweight import smoke test.

## Notes on Scope

This project aims to be a faithful mini representation of the architectural ideas. It is **not** a claim of parity with production DeepSeek-V4 weights, highly-optimized custom CUDA kernels, distributed training frameworks, or data mixtures. The value lies in visibility: these components are transparent, rigorously tested, configurable, and easily trainable in small research regimes.

## 📖 References & Citation

- **Paper copy:** `paper/DeepSeek_V4.pdf`
- **Dataset cards:** WikiText, TinyStories, AG News, IMDB, MiniPile, FineWeb-Edu sample on Hugging Face


This implementation is based on the DeepSeek-V4 technical report:

```bibtex
@misc{deepseekai2026deepseekv4,
  author       = {{DeepSeek-AI}},
  title        = {DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/collections/deepseek-ai/deepseek-v4}},
  note         = {Technical report / preview paper}
}
```

If you use this implementation or adapt its modules for your research, please consider citing:

```bibtex
@misc{reyes2026deepseekv4mini,
  author       = {Reyes Granados, Pablo Alejandro},
  title        = {DeepSeek-V4 Mini: A Paper-Faithful PyTorch Research Implementation},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/pablo-reyes8/deepseek-v4-mini}}
}
```
