#!/usr/bin/env python3
"""Compress Qwen-3.5-4B via selective group-wise INT8 quantization (Track 1+2).

Usage:
    python compress.py --model_name Qwen-3.5-4B \
        --checkpoint_path <base_checkpoint> --output_path <compressed_dir>
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

from compression import GROUP_SIZE, build_plan, meta_for_plan
from decompression import dequantize_groupwise_int8, quantize_groupwise_int8


def dequantize_probe(
    q: torch.Tensor, s: torch.Tensor, shape: torch.Size
) -> torch.Tensor:
    flat = q.reshape(-1).to(torch.float32)
    rep = s.to(torch.float32).repeat_interleave(GROUP_SIZE)[: flat.numel()]
    return (flat * rep).reshape(shape)

TARGET_FRAC = 0.80  # 20% size reduction target.
ORIGINAL_TEXT_GB = 8.0585
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def shard_files(ckpt: Path) -> list[Path]:
    index = ckpt / "model.safetensors.index.json"
    if index.exists():
        meta = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set(meta.get("weight_map", {}).values()))
        return [ckpt / n for n in names]
    files = sorted(ckpt.glob("*.safetensors"))
    if files:
        return files
    raise FileNotFoundError(f"No safetensors weights under {ckpt}")


def load_tensor_shapes(files: list[Path]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in h.keys():
                shapes[k] = tuple(h.get_tensor(k).shape)
    return shapes


def copy_sidecars(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_dir() or item.suffix.lower() in WEIGHT_SUFFIXES:
            continue
        if item.name in {"model.safetensors.index.json", "compression_meta.json"}:
            continue
        shutil.copy2(item, dst / item.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--target_frac", type=float, default=TARGET_FRAC)
    args = ap.parse_args()

    src = Path(args.checkpoint_path)
    dst = Path(args.output_path)
    if not src.exists():
        print(f"ERROR: checkpoint not found: {src}", file=sys.stderr)
        return 1

    files = shard_files(src) if src.is_dir() else [src]
    print(f"[compress] reading shapes from {len(files)} shard(s)...", flush=True)
    shapes = load_tensor_shapes(files)

    plan = build_plan(shapes, target_frac=args.target_frac)
    print(
        f"[compress] quantizing {len(plan.to_quantize)} tensors "
        f"(projected text frac={plan.projected_frac:.4f})",
        flush=True,
    )
    quant_set = set(plan.to_quantize)

    out_tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    err_sum, err_max, err_n = 0.0, 0.0, 0
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in h.keys():
                w = h.get_tensor(k)
                if k in quant_set:
                    q, s = quantize_groupwise_int8(w, group_size=GROUP_SIZE)
                    # Fidelity probe (original still in hand): rel L2 error.
                    with torch.no_grad():
                        rec = dequantize_probe(q, s, w.shape)
                        rel = (rec.float() - w.float()).norm() / w.float().norm().clamp_min(1e-12)
                    err_sum += float(rel)
                    err_max = max(err_max, float(rel))
                    err_n += 1
                    out_tensors[k] = q.contiguous()
                    scales[k + ".qscale"] = s.contiguous()
                else:
                    out_tensors[k] = w.to(torch.bfloat16).contiguous()
        print(f"[compress] processed {f.name}", flush=True)
    print(f"[compress] quant fidelity: mean_rel={err_sum / max(err_n, 1):.6f} max_rel={err_max:.6f}", flush=True)

    dst.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(dst / "model.safetensors"))
    save_file(scales, str(dst / "scales.safetensors"))
    (dst / "compression_meta.json").write_text(
        json.dumps(meta_for_plan(plan, target_frac=args.target_frac), indent=2),
        encoding="utf-8",
    )
    if src.is_dir():
        copy_sidecars(src, dst)

    # Size report (mirrors measure_checkpoint_bits.py: numel * dtype bits).
    text_bits = 0
    for k, t in list(out_tensors.items()):
        if "visual." in k.lower():
            continue
        text_bits += t.numel() * t.element_size() * 8
    for t in scales.values():
        text_bits += t.numel() * t.element_size() * 8
    frac = text_bits / 8 / 1024**3 / ORIGINAL_TEXT_GB
    print(f"[compress] text size frac = {frac:.4f} (target <={args.target_frac})", flush=True)
    print(f"[compress] wrote {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
