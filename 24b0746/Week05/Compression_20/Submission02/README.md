# Week05 Compression_20 Submission02 — RESP (MLP width, decode-only grads)

Base: `torch.bfloat16` → compressed: `torch.bfloat16` pruned (MLP `9216→7372`, 20% width).

## Setup
```bash
uv sync
source .venv/bin/activate
```

## Compress
```bash
python compress.py --model_name Qwen-3.5-4B --checkpoint_path <base> --output_path <out>
```

## Decompress
```bash
python decompress.py --model_name Qwen-3.5-4B --checkpoint_path <comp> --output_path <decomp>
```
