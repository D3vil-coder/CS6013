#!/usr/bin/env python3
"""Decompress RESP — already bf16, just copy."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()
    src = Path(args.checkpoint_path)
    dst = Path(args.output_path)
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)
    print(f"[decompress] copied {src} -> {dst}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
