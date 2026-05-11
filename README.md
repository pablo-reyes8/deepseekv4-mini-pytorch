<p align="center">
  <img src="assets\header_image.png" width="1000"/>
</p>



# DeepSeek-V4 Mini: A Paper-Faithful From-Scratch PyTorch Implementation

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
- [Inference Status](#inference-status)
- [🏗️ Repository Layout](#️-repository-layout)
- [📘 Documentation](#-documentation)
- [Installation](#installation)
- [Run Tests](#run-tests)
- [⚙️ Model Configs](#️-model-configs)
- [📚 Dataset Presets](#-dataset-presets)
- [🔬 Training A Tiny Model](#-training-a-tiny-model)
- [Training With Batches and Indexing](#training-with-batches-and-indexing)
- [Ablation Suite](#ablation-suite)
- [Parallelism](#parallelism)
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
| **Ablation Suite** | Six high-level experiment suites for attention, compression, mHC, MoE, MTP, and full-stack composition |

## Inference Status

The repo now includes a complete pure-PyTorch inference path for the mini DeepSeek stack. It supports text prompts through a tokenizer, token-id prompts for debugging, greedy/sampling generation, MTP draft diagnostics, and three explicit cache modes: `audit`, `mha_decode`, and `deepseek_decode`.

The main DeepSeek path is:

```python
from inference import inference_autoregresive

out = inference_autoregresive(
    model=model,
    prompt="key key_1 is value_7 question : what is key_1 ? answer :",
    tokenizer=tokenizer,
    cache_mode="deepseek_decode",
    deepseek_prefill_mode="parallel",
    max_new_tokens=32,
    do_sample=False,
    return_cache_stats=True,
)
```

`deepseek_decode` builds real HCA/CSA layer caches. In `parallel` prefill mode, the prompt is processed once, each block captures its normalized attention input, and the HCA/CSA caches are populated with compressed global entries, pending tails, and local windows. Future tokens decode one at a time from that cache.

Console generation uses the same wrapper:

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

See [Inference Overview](docs/inference/overview.md), [Cache Modes](docs/inference/cache_modes.md), and [HCA/CSA KV Cache Mechanics](docs/inference/kv_cache.md).

## 🏗️ Repository Layout

```text
.
├── config/                         # YAML profiles for reproducible experiments
│   ├── data/                       # dataset presets: synthetic, TinyStories, WikiText-2, AG News, IMDB, MiniPile
│   ├── model/                      # model variants: tiny, mini, CSA+MoE+mHC+MTP
│   └── training/                   # CPU smoke and single-GPU training profiles
│
├── data/                           # causal LM datasets, tokenization, and dataloader inspection
│   ├── data_utils.py               # batch normalization helpers used by training/eval
│   ├── inspection.py               # tensor and dataloader summaries for CLI inspection
│   ├── syntethic_long_context_retrieval.py
│   │                                  # local synthetic retrieval task for long-context smoke tests
│   ├── text_datasets.py            # Hugging Face text presets and generic causal LM loader
│   └── tinystories_data.py         # TinyStories-specific tokenizer and dataloaders
│
├── src/                            # DeepSeek-V4 Mini architecture
│   ├── deepseek_csa_attention.py   # Compressed Sparse Attention
│   ├── deepseek_hca_attention.py   # Heavily Compressed Attention
│   ├── deepseek_moe.py             # DeepSeek-style routed/shared MoE layer
│   ├── mHC_residuals.py            # Manifold-Constrained Hyper-Connections
│   ├── deepseek_mtp.py             # Multi-Token Prediction head
│   ├── deepseek_block.py           # configurable Transformer block composition
│   ├── mini_deepseek_class.py      # DeepSeekV4LM wrapper
│   └── transformer_modules/        # baseline RMSNorm, RoPE, MHA, SwiGLU, embeddings, blocks
│
├── training/                       # train/eval stack and diagnostics
│   ├── train_deepseek.py           # high-level orchestration
│   ├── train_one_epoch.py          # single-epoch training loop
│   ├── eval_one_epoch.py           # evaluation and qualitative preview
│   ├── muon_optimizer.py           # Muon + AdamW hybrid optimizer
│   ├── scheduler.py                # warmup + cosine scheduler
│   ├── chekpoints.py               # checkpoint save/load utilities
│   └── *_metrics.py                # LM, MoE, mHC, MTP, and module diagnostics
│
├── ablations/                      # high-level experimental suites
│   ├── ablation_configs.py         # A1-A6 variant generation and memory-scaled configs
│   ├── model_factory.py            # DeepSeekV4LM and MiniCausalLM ablation factory
│   ├── data_factory.py             # synthetic/HF dataloader construction
│   ├── evaluate_ablation.py        # LM, retrieval, and inference/cache metrics
│   ├── run_ablation.py             # sequential runner with checkpointing and cache cleanup
│   ├── report.py                   # summary.csv and summary.md aggregation helpers
│   └── suites.py                   # ablation_1 ... ablation_6 public wrappers
│
├── inference/                      # generation, sampling, active decode, and cache implementations
│   ├── inference_config.py         # generation and cache-mode configuration
│   ├── prefill.py                  # audit, MHA, and DeepSeek prefill paths
│   ├── decode.py                   # one-token decode step and cache metadata
│   ├── hca_cache.py                # HCA compressed/global/local cache state
│   ├── csa_cache.py                # CSA main/index/local cache state
│   ├── deepseek_cache_builder.py   # full-prompt cache construction from layer inputs
│   ├── generate.py                 # autoregressive generation loop
│   └── audit.py                    # high-level audit wrapper and logit comparison
│
├── parallel/                       # PyTorch-native educational parallelism
│   ├── parallel_config.py          # DDP/model-parallel configuration object
│   ├── parallel_utils.py           # rank helpers, seeding, device moves, metric reduction
│   ├── data_parallel.py            # DDP setup, samplers, train/eval wrappers, save helpers
│   ├── model_parallel.py           # layerwise/blockwise DeepSeekV4LM placement
│   └── README.md                   # scope, limitations, and usage notes
│
├── scripts/                        # operational CLIs
│   ├── data_cli.py                 # list, download, tokenize, and inspect datasets
│   ├── train_cli.py                # tiny synthetic training smoke runs
│   ├── inspect_cli.py              # model summaries and module-level test runner
│   ├── ablation_cli.py             # run A1-A6 quick/full ablation suites
│   ├── inference_cli.py            # checkpoint loading and text generation with KV caches
│   └── parallel_cli.py             # DDP/model-parallel smoke tests and placement plans
│
├── docs/                           # architecture and configuration reference
│   ├── architecture/               # CSA, HCA, MoE, mHC, MTP, and model overview
│   ├── training/                   # pipeline, Muon, scheduler, autocast, metrics, EMA/checkpoints
│   ├── config_reference/           # hyperparameter reference by subsystem
│   ├── data/                       # dataset guide
│   ├── parallel/                   # DDP/model-parallel scope and limitations
│   └── cli/                        # command line reference
│
├── tests/                          # CPU-safe coverage for model behavior and causality
│   ├── data/                       # dataset preset and causal text loader tests
│   ├── training/                   # optimizer, scheduler, batch, and tiny-training tests
│   ├── parallel/                   # DDP/model-parallel CPU smoke and utility tests
│   ├── inference/                  # generation, sampling, MTP draft, and active-cache tests
│   ├── test_csa.py                 # CSA shape, causality, and gradient checks
│   ├── test_hca.py                 # HCA compression/local-window checks
│
│
├── notebooks/                      # interactive exploration
├── paper/                          # local DeepSeek-V4 paper reference
├── Dockerfile                      # containerized test/dev environment
├── docker-compose.yml              # compose entrypoint for tests/shell
├── LICENSE
└── README.md
```

## 📘 Documentation

The full technical documentation lives in [`docs/`](docs/README.md). It is organized around the two things that matter most for this project: understanding the architecture and knowing exactly which hyperparameters are configurable.

Recommended entry points:

- [Architecture Overview](docs/architecture/overview.md)
- [Attention Modules: MHA, HCA, CSA](docs/architecture/attention_modules.md)
- [HCA: Heavily Compressed Attention](docs/architecture/hca.md)
- [CSA: Compressed Sparse Attention](docs/architecture/csa.md)
- [MoE and Dense FFN](docs/architecture/moe_and_ffn.md)
- [mHC Residual Streams](docs/architecture/mhc.md)
- [MTP Auxiliary Prediction](docs/architecture/mtp.md)
- [Inference Overview](docs/inference/overview.md)
- [Inference Cache Modes](docs/inference/cache_modes.md)
- [HCA/CSA KV Cache Mechanics](docs/inference/kv_cache.md)
- [Training Pipeline](docs/training/pipeline.md)
- [Muon Optimizer](docs/training/muon.md)
- [Metrics and Diagnostics](docs/training/metrics.md)
- [Model Config Reference](docs/config_reference/model.md)
- [Training Config Reference](docs/config_reference/training.md)
- [Data Config Reference](docs/config_reference/data.md)
- [Parallelism Guide](docs/parallel/overview.md)
- [CLI Reference](docs/cli/reference.md)

## 🚀 Installation

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

*Current validation on CPU: `756 passed, 4 skipped`*

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

## Ablation Suite

The top-level `ablations/` package turns the repo into an experimental platform rather than only an implementation. Each suite has a high-level wrapper that accepts a dataset config, memory-oriented model limits, seeds, and training budget overrides. The runner executes variants sequentially, saves checkpoints before cleanup, and clears Python/Torch/CUDA cache before moving to the next model.

| Suite | Question | Main variants |
| :--- | :--- | :--- |
| `A1` Hybrid Attention Composition | Does CSA/HCA hybrid attention improve quality-memory trade-offs over MHA, HCA-only, and CSA-only? | `dense_mha_baseline`, `hca_only`, `csa_only`, `hybrid_csa_hca`, `hybrid_hca_csa` |
| `A2` Compression and Window Trade-off | How do compression factor, local window, and sparse top-k affect retrieval, perplexity, and cache cost? | HCA and CSA grids over `compression_factor`, `window_size`, `top_k_blocks` |
| `A3` mHC Utility | Does mHC improve stability in shallow and deeper low-token regimes? | MHA/hybrid with and without mHC at shallow and deeper depths |
| `A4` MoE Routing | Are routed experts, shared experts, and balance losses useful in the mini regime? | Dense FFN, MoE no shared, shared experts, no balance, hash routing |
| `A5` MTP Auxiliary Loss | Does MTP help convergence and generation, or distract next-token learning? | MTP off, depth/weight sweeps, weighted depth loss |
| `A6` System-Level Stack | Which DeepSeek-style components matter most when composed? | baseline, +compressed attention, +MoE, +mHC, +MTP, full-minus variants |

Python wrapper example:

```python
from ablations import ablation_1

results = ablation_1(
    data_config={
        "dataset": "synthetic_long_context",
        "block_size": 128,
        "batch_size": 4,
        "num_train_examples": 2_000,
        "num_val_examples": 300,
    },
    max_model={
        "d_model": 128,
        "n_layers": 4,
        "max_seq_len": 128,
        "n_heads": 4,
        "head_dim": 32,
    },
    training_config={
        "epochs": 1,
        "max_batches_per_epoch": 30,
        "eval_max_batches": 10,
        "optimizer_type": "adamw",
        "device": "cuda",
    },
    seeds=[1],
    quick=False,
)
```

CLI smoke run:

```bash
python -m scripts.ablation_cli --ablation A1 --quick --limit-variants 1 --device cpu
```

Outputs are written under `outputs/ablations/{ablation_id}/`, with one `final_metrics.json` per variant/seed plus suite-level `summary.csv` and `summary.md`.

## Parallelism

The repo includes a top-level `parallel/` package for PyTorch-native distributed experiments that stay close to the architecture without claiming custom runtime engineering from the paper.

Implemented now:

- **DDP data parallelism:** `torch.distributed`, `DistributedDataParallel`, `DistributedSampler`, rank-aware checkpoint saves, and scalar metric aggregation.
- **Layerwise model parallelism:** whole `DeepSeekV4LM` blocks are assigned to ordered devices and activations are moved between block boundaries.
- **CPU verification:** config validation, one-process `gloo` DDP, sampler behavior, metric reductions, model-parallel equivalence on CPU, and mHC wrapper compatibility.

V1 only accepts active model-parallel devices: every `balance` entry must be greater than zero, so `len(devices) <= n_layers`. For training, wrap the model before building the optimizer:

```python
model = DeepSeekV4LM(config)
model = wrap_model_parallel(model, devices=["cuda:0", "cuda:1"], balance=[8, 8])
optimizer = build_optimizer(model, train_config)
```

Not implemented by design: custom CUDA kernels, FP4/FP8 training kernels, NCCL topology scheduling, DualPipe, and true all-to-all expert parallelism.

```bash
python -m scripts.parallel_cli plan --n-layers 6 --devices cpu,cpu --balance 2,4
python -m scripts.parallel_cli model-parallel-smoke --devices cpu --n-layers 2
python -m scripts.parallel_cli ddp-smoke --backend gloo --n-layers 1
```

For the full scope and limitations, see [Parallelism Guide](docs/parallel/overview.md).

## Docker Support

```bash
docker build -t deepseekv4-mini .
docker compose run --rm tests
```

## 🛠️ Command Line Tools

After installing with `pip install -e ".[dev,data]"`, the project exposes four transparent CLIs for immediate interaction:

```bash
deepseekv4-data presets
deepseekv4-data synthetic-inspect --block-size 32 --batch-size 2
deepseekv4-train smoke --attention hca --ffn dense --max-batches 2
deepseekv4-inspect model-summary --attention csa --ffn moe
deepseekv4-inspect module-tests csa --quiet
deepseekv4-ablate --ablation A1 --quick --limit-variants 1 --device cpu
deepseekv4-parallel plan --n-layers 4 --devices cpu,cpu
deepseekv4-parallel model-parallel-smoke --devices cpu --n-layers 2
deepseekv4-parallel ddp-smoke --backend gloo --n-layers 1
```

The same commands work natively without CLI installation through Python modules:

```bash
python -m scripts.data_cli synthetic-inspect --block-size 32 --batch-size 2
python -m scripts.train_cli smoke --attention mha --ffn dense --max-batches 1 --quiet
python -m scripts.inspect_cli module-tests training --quiet
python -m scripts.ablation_cli --ablation A1 --quick --limit-variants 1 --device cpu
python -m scripts.parallel_cli tests --quiet
```

**CLI Scope:**
- `data_cli`: List presets, inspect synthetic data, and download HF text presets.
- `train_cli`: Run a tiny synthetic training smoke test with configurable attention/FFN/mHC/MTP.
- `inspect_cli`: Summarize model parameter structure and execute targeted module tests.
- `ablation_cli`: Generate and run A1-A6 ablation configs sequentially with checkpointing and cache cleanup.
- `parallel_cli`: Inspect model placement, run CPU-safe layerwise model-parallel smoke tests, run one-process `gloo` DDP smoke tests, and launch `tests/parallel`.

## CI Strategy

Continuous Integration is strictly path-aware to ensure speed without losing critical coverage:
- Changes in `src/`, configs, or packaging trigger **model & component tests**.
- Changes in `training/` or `tests/training/` trigger **training-stack tests**.
- Changes in `data/` or `tests/data/` trigger **dataset loader tests**.
- Changes in `parallel/`, `scripts/parallel_cli.py`, or `tests/parallel/` trigger **parallelism tests**.
- Changes in `ablations/`, `scripts/ablation_cli.py`, or `tests/experiments/` trigger **ablation-suite tests**.
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
  howpublished = {\url{https://github.com/pablo-reyes8/deepseek-v4-mini-pytorch}}
}
```

## License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for the full legal text.

Copyright (c) 2026 Pablo Reyes.
