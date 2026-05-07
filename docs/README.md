# DeepSeek-V4 Mini Documentation

This folder documents the model, training stack, data pipeline, and CLIs at the level needed to configure experiments without reading every source file first.

The docs are intentionally practical:

- Short description of each module.
- What role it plays in the architecture.
- Which hyperparameters can be configured.
- What each hyperparameter changes.
- Notes on safe tiny/CPU settings versus larger research settings.

## Architecture

- [Architecture Overview](architecture/overview.md)
- [Attention Modules: MHA, HCA, CSA](architecture/attention_modules.md)
- [HCA: Heavily Compressed Attention](architecture/hca.md)
- [CSA: Compressed Sparse Attention](architecture/csa.md)
- [MoE and Dense FFN](architecture/moe_and_ffn.md)
- [mHC Residual Streams](architecture/mhc.md)
- [MTP Auxiliary Prediction](architecture/mtp.md)

## Training System

- [Training Pipeline](training/pipeline.md)
- [Autocast and Precision](training/autocast_and_precision.md)
- [Scheduler](training/scheduler.md)
- [Muon Optimizer](training/muon.md)
- [Metrics and Diagnostics](training/metrics.md)
- [Checkpointing and EMA](training/checkpointing_and_ema.md)

## Inference System

- [Inference Overview](inference/overview.md)
- [Inference Cache Modes](inference/cache_modes.md)
- [HCA and CSA KV Cache Mechanics](inference/kv_cache.md)

## Configuration Reference

- [Model Config Reference](config_reference/model.md)
- [Attention Config Reference](config_reference/attention.md)
- [MoE and FFN Config Reference](config_reference/moe.md)
- [mHC Config Reference](config_reference/mhc.md)
- [MTP Config Reference](config_reference/mtp.md)
- [Training Config Reference](config_reference/training.md)
- [Optimizer and Scheduler Config Reference](config_reference/optimizer.md)
- [Data Config Reference](config_reference/data.md)
- [Logging, Evaluation, and Checkpointing](config_reference/logging_eval_checkpointing.md)

## Operational Docs

- [CLI Reference](cli/reference.md)
- [Dataset Guide](data/datasets.md)
- [Parallelism Guide](parallel/overview.md)
