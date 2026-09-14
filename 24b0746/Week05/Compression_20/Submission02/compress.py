#!/usr/bin/env python3
"""RESP 20% — self-generated CoT + decode-only grad pruning (MLP width)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from safetensors.torch import save_file
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
] * 7  # 70 prompts, we use first N_CALIB

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
    # init per neuron accum
    from collections import defaultdict
    acc = defaultdict(list)
    # we need to map weight to neuron: gate/up have shape [9216,2560] rows = neurons, down has [2560,9216] cols = neurons
    # For down, we treat column importance as mean over rows.
    for q, trace in traces[:16]:  # use subset for grad to save time (16 traces)
        full = q + " " + trace
        enc = tok(full, return_tensors="pt")
        if isinstance(enc, dict):
            input_ids = enc["input_ids"].to(model.device)
        elif hasattr(enc, "input_ids"):
            input_ids = enc.input_ids.to(model.device)
        else:
            input_ids = enc.to(model.device)
        # forward
        out = model(input_ids=input_ids, labels=input_ids)
        loss = out.loss
        model.zero_grad()
        loss.backward()
        # build param map once
        param_map = dict(model.named_parameters())
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
    print(f"[resp] pruning {plan.pruned} neurons, keeping {plan.kept}", flush=True)
    # 3) apply pruning: create new state with pruned weights
    new_state = {}
    for k, v in state.items():
        L = layer_of(k)
        if L is not None and L in plan.keep:
            keep_idx = torch.tensor(plan.keep[L], device=v.device)
            if "down_proj" in k:
                # [2560,9216] -> keep cols
                new_state[k] = v[:, keep_idx].contiguous()
            else:
                new_state[k] = v[keep_idx, :].contiguous()
        else:
            new_state[k] = v
    # fix config: update intermediate_size
    # need to copy config and adjust
    import json as js
    cfg_path = Path(src) / "config.json"
    cfg = js.loads(cfg_path.read_text())
    # text_config intermediate_size
    if "text_config" in cfg and "intermediate_size" in cfg["text_config"]:
        old = cfg["text_config"]["intermediate_size"]
        new = int(old * KEEP_RATIO)
        cfg["text_config"]["intermediate_size"] = new
        print(f"[resp] intermediate {old} -> {new}", flush=True)
    # also need to update model.safetensors + config
    # save
    # we save as single shard for simplicity (fits)
    save_file({k: v.to(torch.bfloat16) for k,v in new_state.items()}, str(dst / "model.safetensors"))
    (dst / "config.json").write_text(js.dumps(cfg, indent=2))
    # copy tokenizer files
    for p in Path(src).glob("*.json"):
        if p.name not in {"config.json"}:
            shutil.copy2(p, dst / p.name)
    for p in Path(src).glob("*.py"):
        shutil.copy2(p, dst / p.name)
    # meta
    (dst / "compression_meta.json").write_text(json.dumps({"method":"resp_mlp_width","keep_ratio":KEEP_RATIO,"intermediate_pruned":plan.pruned}, indent=2))
    # size report
    text_bits = sum(v.numel()*16 for k,v in new_state.items() if "visual." not in k.lower())
    frac = text_bits/8/1024**3/8.0585
    print(f"[resp] done frac={frac:.4f}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
