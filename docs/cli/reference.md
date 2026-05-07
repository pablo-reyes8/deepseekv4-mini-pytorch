# CLI Reference

The project exposes five command groups.

After editable install:

```bash
deepseekv4-data
deepseekv4-train
deepseekv4-inspect
deepseekv4-infer
deepseekv4-parallel
```

Without installation:

```bash
python -m scripts.data_cli
python -m scripts.train_cli
python -m scripts.inspect_cli
python -m scripts.inference_cli
python -m scripts.parallel_cli
```

## Data CLI

List presets:

```bash
python -m scripts.data_cli presets
```

Inspect synthetic data:

```bash
python -m scripts.data_cli synthetic-inspect \
  --block-size 64 \
  --batch-size 2 \
  --num-train-examples 8
```

Show one HF preset:

```bash
python -m scripts.data_cli hf-info wikitext2
```

Prepare and inspect HF data:

```bash
python -m scripts.data_cli hf-prepare wikitext2 \
  --block-size 256 \
  --batch-size 8 \
  --max-tokenizer-documents 10000 \
  --max-train-documents 2000
```

## Train CLI

Run tiny CPU smoke training:

```bash
python -m scripts.train_cli smoke \
  --attention mha \
  --ffn dense \
  --max-batches 1 \
  --quiet
```

Try HCA:

```bash
python -m scripts.train_cli smoke \
  --attention hca \
  --ffn dense \
  --block-size 64 \
  --max-batches 2
```

Try MoE:

```bash
python -m scripts.train_cli smoke \
  --attention csa \
  --ffn moe \
  --num-experts 4 \
  --top-k-experts 2
```

## Inspect CLI

Model summary:

```bash
python -m scripts.inspect_cli model-summary --attention csa --ffn moe
```

Run tests for one module group:

```bash
python -m scripts.inspect_cli module-tests csa --quiet
python -m scripts.inspect_cli module-tests training --quiet
python -m scripts.inspect_cli module-tests data --quiet
python -m scripts.inspect_cli module-tests inference --quiet
```

## Inference CLI

Generate from the bundled manual checkpoint with DeepSeek caches:

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

Use raw token ids when no tokenizer is available:

```bash
python -m scripts.inference_cli generate \
  --checkpoint outputs/deepseekv4_mini_muon_last_manual.pt \
  --config-json outputs/deepseekv4_mini_muon_last_manual.json \
  --prompt-ids 1,4,5,6 \
  --cache-mode deepseek_decode \
  --max-new-tokens 8 \
  --no-do-sample
```

## Parallel CLI

Inspect a layer/device placement plan:

```bash
python -m scripts.parallel_cli plan \
  --n-layers 6 \
  --devices cpu,cpu \
  --balance 2,4
```

Run a CPU-safe model-parallel forward pass:

```bash
python -m scripts.parallel_cli model-parallel-smoke \
  --devices cpu \
  --n-layers 2
```

Run a one-process DDP smoke check with `gloo`:

```bash
python -m scripts.parallel_cli ddp-smoke \
  --backend gloo \
  --n-layers 1
```

Run only the parallelism tests:

```bash
python -m scripts.parallel_cli tests --quiet
```
