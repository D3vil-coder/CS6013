# Week05 Compression_20 Submission01

Base dtype: `torch.bfloat16`. Compressed dtype: `torch.int8`
(group-wise symmetric, group size 128) for selected MLP projections;
all other tensors remain `torch.bfloat16`.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Compress

```bash
python compress.py --model_name Qwen-3.5-4B \
  --checkpoint_path <base_checkpoint> --output_path <compressed_dir>
```

## Decompress (offline, no downloads)

```bash
python decompress.py --model_name Qwen-3.5-4B \
  --checkpoint_path <compressed_dir> --output_path <decompressed_dir>
```

The decompressed checkpoint is a standard single-shard
`model.safetensors` (bf16) plus the copied config/tokenizer files.
