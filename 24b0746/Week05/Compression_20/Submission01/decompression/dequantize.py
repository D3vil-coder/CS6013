"""Dequantization helpers (INT8 groups -> bf16)."""

import torch


def dequantize_groupwise_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct fp tensor from int8 codes + per-group fp16 scales."""
    flat = q.reshape(-1).to(torch.float32)
    n_groups = (flat.numel() + group_size - 1) // group_size
    s = scale.reshape(-1).to(torch.float32)
    assert s.numel() == n_groups, f"scale/groups mismatch: {s.numel()} vs {n_groups}"
    scales = s.repeat_interleave(group_size)[: flat.numel()]
    return (flat * scales).reshape(q.shape).to(out_dtype)


def quantize_groupwise_int8(
    w: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group INT8 quantization. Returns (codes int8, scales fp16)."""
    flat = w.detach().to(torch.float32).reshape(-1)
    n_groups = (flat.numel() + group_size - 1) // group_size
    pad = n_groups * group_size - flat.numel()
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.float32)])
    groups = flat.reshape(n_groups, group_size)
    amax = groups.abs().amax(dim=1).clamp_min(1e-12)
    scales = amax / 127.0
    codes = torch.round(groups / scales[:, None]).clamp(-128, 127).to(torch.int8)
    codes = codes.reshape(-1)[: w.numel()].reshape(w.shape)
    return codes, scales.to(torch.float16)
