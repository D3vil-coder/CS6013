#!/usr/bin/env python3
"""RESP 20% — self-generated CoT + decode-only grad pruning (MLP width)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText

from compression.resp import build_plan, eligible_keys, layer_of

# 20% prune -> keep 80%
KEEP_RATIO = 0.8
N_CALIB = 64
MAX_NEW = 256

MATH_PROMPTS = [
 "Janet ducks lay 16 eggs per day. She eats 3 and bakes with 4. She sells the rest at $2 each. How much does she make?",
 "A farmer has 17 sheep, all but 9 die. How many remain?",
 "If 3x + 5 = 20, what is x?",
 "What is 15% of 200?",
 "A rectangle has length 8 and width 5. What is its area?",
 "A store has 10 apples, sells 3, buys 5 more. How many now?",
 "Solve 2x + 3 = 7",
 "What is 7 * 8?",
 "A train travels 60 km/h for 2 hours. Distance?",
 "If a+b=5 and a-b=1, find a",
] * 8  # 64 prompts for self-generated CoT calibration

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    src = Path(args.checkpoint_path)
    dst = Path(args.output_path)
    dst.mkdir(parents=True, exist_ok=True)

    print("[resp] loading tokenizer/model...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(str(src), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.train()
    # enable grad for importance
    for p in model.parameters():
        p.requires_grad = True

    # 1) self-generated CoT traces
    print(f"[resp] generating {N_CALIB} traces...", flush=True)
    traces = []
    model.eval()
    with torch.no_grad():
        for q in MATH_PROMPTS[:N_CALIB]:
            msgs = [{"role": "user", "content": q + " Reason step by step, then give \\boxed{X}."}]
            try:
                enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", enable_thinking=True)
            except TypeError:
                enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            # handle BatchEncoding vs tensor
            if isinstance(enc, dict):
                inp = enc["input_ids"].to(model.device)
            elif hasattr(enc, "input_ids"):
                inp = enc.input_ids.to(model.device)
            else:
                inp = enc.to(model.device)
            out = model.generate(inp, max_new_tokens=MAX_NEW, do_sample=False)
            txt = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
            traces.append((q, txt))
            print(f" trace {len(traces)}: {txt[:80]!r}", flush=True)
    # 2) decode-only grad importance
    print("[resp] computing decode-only importance...", flush=True)
    model.train()
    # collect eligible layers
    state = model.state_dict()
    keys = eligible_keys(list(state.keys()))
    # importance per layer: mean |grad * weight| per neuron
    imp: dict[int, list[float]] = {}
    # we need to map weight to neuron: gate/up have shape [9216,2560] rows = neurons, down has [2560,9216] cols = neurons
    # For down, we treat column importance as mean over rows.
    param_map = dict(model.named_parameters())
    for q, trace in traces[:16]:  # use subset for grad to save time (16 traces)
        full = q + " " + trace
        enc = tok(full, return_tensors="pt")
        if isinstance(enc, dict):
            input_ids = enc["input_ids"].to(model.device)
        elif hasattr(enc, "input_ids"):
            input_ids = enc.input_ids.to(model.device)
        else:
            input_ids = enc.to(model.device)
        # supervise only generated-token positions; prompt positions carry no loss
        prompt_len = len(tok(q, add_special_tokens=False)["input_ids"])
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        model.zero_grad(set_to_none=True)
        # forward
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        for k in keys:
            L = layer_of(k)
            if L is None:
                continue
            p = param_map.get(k)
            if p is None or p.grad is None:
                continue
            g = p.grad
            w = state[k].to(g.device).float()
            # per-neuron score: mean |g*w|
            if "down_proj" in k:
                # shape [2560,9216] -> per col
                prod = (g.float() * w).abs().mean(dim=0)  # [9216]
            else:
                prod = (g.float() * w).abs().mean(dim=1)  # [9216]
            prod = prod.detach().cpu().float().tolist()
            if L not in imp:
                imp[L] = [0.0]*len(prod)
            for i,v in enumerate(prod):
                imp[L][i] += v / 16

    # build plan
    plan = build_plan(imp, keep_ratio=KEEP_RATIO)
    # Every text layer must end up at the same width, otherwise the saved
    # checkpoint cannot be loaded with a single intermediate_size. Fill any
    # layer missing gradient importance with a weight-magnitude fallback.
    text_cfg_probe = getattr(model.config, "text_config", model.config)
    num_layers = int(text_cfg_probe.num_hidden_layers)
    old_inter = int(text_cfg_probe.intermediate_size)
    keep_n = int(old_inter * KEEP_RATIO)
    for L in range(num_layers):
        if L in plan.keep_idx:
            continue
        scores = None
        for suffix, dim in (("gate_proj", 1), ("up_proj", 1), ("down_proj", 0)):
            kk = f"model.language_model.layers.{L}.mlp.{suffix}.weight"
            if kk not in state:
                continue
            s = state[kk].detach().to(torch.float32).abs().mean(dim=dim)
            s = s.detach().cpu().float().tolist()
            scores = s if scores is None else [a + b for a, b in zip(scores, s)]
        if scores is None:
            continue
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep_n]
        plan.keep_idx[L] = sorted(idx)
        plan.pruned += len(scores) - keep_n
        plan.kept += keep_n
        print(f"[resp] layer {L}: magnitude fallback (no grads)", flush=True)
    assert set(plan.keep_idx.keys()) == set(range(num_layers)), "plan must cover all layers"
    print(f"[resp] pruning {plan.pruned} neurons, keeping {plan.kept}", flush=True)

    def _is_visual_key(name: str) -> bool:
        n = name.replace("\\", "/").lower()
        if "visual." in n:
            return True
        return any(p in {"visual", "vision", "vision_tower", "vision_model"} for p in n.split("."))

    def _set_param(root: torch.nn.Module, full_name: str, tensor: torch.Tensor) -> None:
        *mods, attr = full_name.split(".")
        m = root
        for x in mods:
            m = getattr(m, x)
        setattr(m, attr, torch.nn.Parameter(tensor))

    # 3) apply pruning in place so tied weights remain handled by transformers
    orig_text_params = 0
    saved_params = 0
    for k, v in state.items():
        if _is_visual_key(k):
            continue
        orig_text_params += v.numel()
        L = layer_of(k)
        if L is not None and L in plan.keep_idx:
            keep_idx = torch.tensor(plan.keep_idx[L], device=v.device)
            if "down_proj" in k:
                # [2560,9216] -> keep cols
                pruned = v[:, keep_idx].contiguous()
            else:
                pruned = v[keep_idx, :].contiguous()
            saved_params += v.numel() - pruned.numel()
            _set_param(model, k, pruned.to(v.dtype))
    # fix config: update intermediate_size to the actual kept width
    text_cfg = getattr(model.config, "text_config", model.config)
    old = int(text_cfg.intermediate_size)
    new = len(next(iter(plan.keep_idx.values())))
    text_cfg.intermediate_size = new
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        try:
            model.language_model.config.intermediate_size = new
        except Exception:
            pass
    print(f"[resp] intermediate {old} -> {new}", flush=True)
    # release calibration memory before saving
    model.zero_grad(set_to_none=True)
    model.eval()
    del state, param_map, imp, traces
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # save a standard sharded HF checkpoint; save_pretrained handles tied weights
    model.save_pretrained(str(dst), safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(str(dst))
    # meta
    new_text_params = orig_text_params - saved_params
    frac = (new_text_params * 2) / (8.0585 * 1024**3)
    (dst / "compression_meta.json").write_text(json.dumps({"method": "resp_mlp_width", "keep_ratio": KEEP_RATIO, "intermediate_old": old, "intermediate_new": new, "intermediate_pruned": plan.pruned, "size_frac": frac}, indent=2))
    print(f"[resp] done frac={frac:.4f}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
