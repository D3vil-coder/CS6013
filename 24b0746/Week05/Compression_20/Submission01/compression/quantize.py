"""Group-wise symmetric INT8 quantization policy for Qwen3.5-4B.

Target-aware selection: quantize MLP blocks (gate/up/down) in middle layers
first (protecting early/late layers that matter most for reasoning), adding
layers until the projected compressed size meets the reduction target.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

GROUP_SIZE = 128
SCALE_DTYPE = "float16"

# Layers protected from quantization (early parsing + late output layers).
PROTECTED_LAYERS = {0, 1, 30, 31}

# Only these text-tower tensors are eligible for INT8 quantization.
MLP_PROJ_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$"
)


def is_visual_key(name: str) -> bool:
    n = name.replace("\\", "/").lower()
    if "visual." in n:
        return True
    return any(p in {"visual", "vision", "vision_tower", "vision_model"} for p in n.split("."))


def layer_of(name: str) -> int | None:
    m = MLP_PROJ_RE.match(name)
    return int(m.group(1)) if m else None


@dataclass
class QuantPlan:
    """Ordered list of tensor names to quantize + size projection."""

    to_quantize: list[str] = field(default_factory=list)
    projected_text_bytes: int = 0
    projected_frac: float = 1.0


def build_plan(
    tensor_shapes: dict[str, tuple[int, ...]],
    *,
    target_frac: float,
    original_text_bytes: float = 8.0585 * 1024**3,
) -> QuantPlan:
    """Select MLP tensors (middle layers first) until target size is met."""
    eligible: list[tuple[int, str, int]] = []  # (priority, name, numel)
    text_bytes = 0
    for name, shape in tensor_shapes.items():
        numel = 1
        for d in shape:
            numel *= d
        if is_visual_key(name):
            continue
        text_bytes += numel * 2  # bf16 baseline
        layer = layer_of(name)
        if layer is None:
            continue
        # Priority: distance from protected edges (middle layers first).
        priority = min(layer - 1, 30 - layer) if layer not in PROTECTED_LAYERS else -1
        eligible.append((priority, name, numel))

    eligible.sort(key=lambda t: (-t[0], t[1]))
    # Never quantize protected layers in this recipe.
    eligible = [e for e in eligible if e[0] >= 0]

    chosen: list[str] = []
    saved = 0.0
    for _, name, numel in eligible:
        n_groups = (numel + GROUP_SIZE - 1) // GROUP_SIZE
        # int8 (1 byte/elem) + fp16 scale per group vs bf16 (2 bytes/elem).
        new_bytes = numel * 1 + n_groups * 2
        old_bytes = numel * 2
        saved += old_bytes - new_bytes
        chosen.append(name)
        if (text_bytes - saved) / original_text_bytes <= target_frac:
            break

    projected = text_bytes - saved
    return QuantPlan(
        to_quantize=chosen,
        projected_text_bytes=int(projected),
        projected_frac=projected / original_text_bytes,
    )


def meta_for_plan(plan: QuantPlan, *, target_frac: float) -> dict:
    return {
        "method": "selective_groupwise_int8",
        "group_size": GROUP_SIZE,
        "scale_dtype": SCALE_DTYPE,
        "target_frac": target_frac,
        "quantized_tensors": sorted(plan.to_quantize),
        "projected_text_bytes": plan.projected_text_bytes,
        "projected_frac": plan.projected_frac,
    }
