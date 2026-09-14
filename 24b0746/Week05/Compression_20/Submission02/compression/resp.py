"""RESP-style width pruning for Qwen3.5-4B (MLP neurons, decode-only grads)."""
from __future__ import annotations

import re
from dataclasses import dataclass

GROUP_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")

PROTECTED = {0, 1, 30, 31}
H, I = 2560, 9216

def layer_of(k: str) -> int | None:
    m = GROUP_RE.match(k)
    return int(m.group(1)) if m else None

def eligible_keys(keys: list[str]) -> list[str]:
    out=[]
    for k in keys:
        L=layer_of(k)
        if L is not None and L not in PROTECTED:
            out.append(k)
    return out

@dataclass
class PrunePlan:
    keep_idx: dict[int, list[int]]  # layer -> kept neuron indices
    pruned: int
    kept: int

def build_plan(importance: dict[int, list[float]], keep_ratio: float = 0.8) -> PrunePlan:
    keep={}
    pruned=kept=0
    for L, scores in importance.items():
        n=len(scores)
        k=int(n*keep_ratio)
        # keep top-k by importance
        idx=sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
        idx=sorted(idx)
        keep[L]=idx
        pruned+=n-k
        kept+=k
    return PrunePlan(keep, pruned, kept)
