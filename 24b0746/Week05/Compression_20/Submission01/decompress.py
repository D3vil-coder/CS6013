#!/usr/bin/env python3
"""Decompress to fp16/bf16 (offline; no downloads).

Usage:
    python decompress.py --model_name Qwen-3.5-4B \
        --checkpoint_path <compressed_dir> --output_path <decompressed_dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compression import GROUP_SIZE
from decompression import dequantize_groupwise_int8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    src = Path(args.checkpoint_path)
    dst = Path(args.output_path)
    weights = src / "model.safetensors"
    scales_f = src / "scales.safetensors"
    meta_f = src / "compression_meta.json"
    if not weights.exists():
        print(f"ERROR: {weights} missing", file=sys.stderr)
        return 1

    group_size = GROUP_SIZE
    if meta_f.exists():
        group_size = int(json.loads(meta_f.read_text(encoding="utf-8")).get("group_size", GROUP_SIZE))

    print("[decompress] loading scales...", flush=True)
    scales: dict[str, torch.Tensor] = {}
    if scales_f.exists():
        with safe_open(str(scales_f), framework="pt", device="cpu") as h:
            for k in h.keys():
                scales[k] = h.get_tensor(k)

    print("[decompress] reconstructing bf16 weights...", flush=True)
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(weights), framework="pt", device="cpu") as h:
        for k in h.keys():
            t = h.get_tensor(k)
            if t.dtype == torch.int8 and (k + ".qscale") in scales:
                out[k] = dequantize_groupwise_int8(
                    t, scales[k + ".qscale"], group_size=group_size
                ).contiguous()
            else:
                out[k] = t.to(torch.bfloat16).contiguous()

    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst / "model.safetensors"))
    for item in src.iterdir():
        if item.is_file() and item.suffix.lower() not in {
            ".safetensors", ".bin", ".pt", ".pth",
        } and item.name not in {"compression_meta.json"}:
            shutil.copy2(item, dst / item.name)
    print(f"[decompress] wrote {dst} ({len(out)} tensors)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
