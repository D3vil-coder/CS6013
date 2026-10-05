Requires Python 3.12. Run `uv sync` from this directory, with `uv.lock` alongside `pyproject.toml`.

For offline setup, NumPy 2.2.6 and safetensors 0.8.0 must be available in the uv cache. Check with `uv sync --offline --locked`.

Compress the local original checkpoint:

```bash
uv run python compress.py --model_name Qwen-3.5-4B --checkpoint_path /path/to/original --output_path /path/to/packed
```

The default compression profile is `ffn_int8`. The output directory must be empty.

Reconstruct the downloaded packed checkpoint offline:

```bash
uv run python decompress.py --model_name Qwen-3.5-4B --checkpoint_path /path/to/packed --output_path /path/to/reconstructed
```

Base checkpoint tensors are BF16, apart from small FP32 control/norm tensors. The course size denominator is the FP16 baseline. Packed payload tensors are U8. Any quantization scales inside them are FP16. All reconstructed model tensors are BF16. Original FP32 control/norm values are cast once to BF16. `model_name` is an informational label; the code reads only `checkpoint_path`.
