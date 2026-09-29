#!/usr/bin/env python3
"""Group-level DPO of King 127 on `vinhable_single` / `vinhable_double`, 8xH200, DeepSpeed ZeRO-3.

Built on what the lab's runs taught (2026-09-25/26):
  * Train only attention, linear attention, the shared expert and the norms (~1.4B); routed experts,
    routers, embeddings, lm_head, vision and MTP stay frozen. Full-weight runs hit OOM on the Adam
    moments and still moved like noise; with experts frozen every rank touches the same trainable
    tensors on every forward, which keeps ZeRO-3's collectives in step.
  * No DeepSpeed op that needs `nvcc` (Shadeform images have none): the optimizer is torch AdamW.
  * Reference and policy log-probs go through one code path; a step-0 gate compares them before any
    update and stops the run when they disagree.
  * Every rank runs the same number of forwards and backwards per step (dummy work pads the short
    ranks): a rank that skips one deadlocks ZeRO-3.
  * Weight drift is measured on the fp32 master against the Adam noise line sqrt(sum lr^2): the
    bf16 export rounds small moves away.
  * Keys are row uids, checked unique; failures carry a traceback per rank (`@record`).

Unit of training: a dataset row = one behaviour-group prefix k of a pair; its DPO term is over the
assistant turns of group k (vinhable_dpo_data.py renders each as its own production-exact sequence):

    logp_side = sum of the turns' target log-probs          (reference: same, King weights)
    h         = beta * N0 * [ (logp_c - ref_c)/n_c - (logp_r - ref_r)/n_r ]   (--agg mean, default)
              = beta * [ (logp_c - ref_c) - (logp_r - ref_r) ]                 (--agg sum)
    loss      = dpo_weight * [-(1-ls) logsig(h) - ls logsig(-h)] + nll_weight * (-logp_c / n_c)
                (--dpo-weight 0 is plain SFT on the chosen turns)

Sequences of a row are not held in one graph: pass A computes every sequence's log-prob without
grad, the loss's derivative w.r.t. each side's log-prob is formed, and pass B back-propagates each
sequence scaled by that derivative (exact gradient, one sequence in memory at a time).

Phases (same flags):  reference | train | export  — see run_vinhable_dpo_8xh200.sh.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vinhable_dpo_data import Renderer, RowExample, batches, char_length, load_rows, row_example  # noqa: E402

KING_MODEL = "dendriteholdings/albedo-qwen3.6-35b-king-CXXVII"
KING_REVISION = "e920362b460ae6b2a33c9cb298aa7f14a38d5584"
TRAINABLE = re.compile(
    r"^model\.language_model\.("
    r"layers\.\d+\.(self_attn|linear_attn|mlp\.shared_expert|mlp\.shared_expert_gate)\."
    r"|layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight$"
    r"|norm\.weight$)")
FAMILIES = ("linear_attn", "self_attn", "shared_expert", "norm")


def family(name: str) -> str:
    for f in FAMILIES[:3]:
        if f in name:
            return f
    return "norm"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("reference", "train", "export"), required=True)
    p.add_argument("--data", type=Path, required=True, help="folder with train.jsonl and dev.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, help="reference log-probs (default <output>/reference.jsonl)")
    p.add_argument("--base-model", default=KING_MODEL, help="HF id or local folder of the King")
    p.add_argument("--base-revision", default=KING_REVISION)
    p.add_argument("--tokenizer-dir", type=Path, default=None)
    p.add_argument("--max-seq-tokens", type=int, default=65536, help="rows with a longer sequence are skipped")
    p.add_argument("--tokens-per-forward", type=int, default=131072,
                   help="micro-batch budget: sequences x longest (right-padded) per forward")
    p.add_argument("--targets-per-forward", type=int, default=8192,
                   help="target tokens per forward; bounds the single lm_head call's fp32 logits")
    p.add_argument("--reference-rows-per-call", type=int, default=4)
    p.add_argument("--attn-implementation", default="sdpa")
    # objective
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--agg", choices=("mean", "sum"), default="mean")
    p.add_argument("--norm-tokens", type=float, default=512.0, help="N0 for --agg mean")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--nll-weight", type=float, default=0.1)
    p.add_argument("--dpo-weight", type=float, default=1.0,
                   help="0 = pure SFT on the chosen side (rejected still scored for the metrics, never trained)")
    # optimisation
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--rows-per-rank", type=int, default=2, help="rows per rank per optimizer step")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260927)
    # checks and outputs
    p.add_argument("--gate-rows", type=int, default=4, help="rows per rank for the step-0 gate")
    p.add_argument("--gate-max-diff", type=float, default=1e-3, help="max mean |policy-ref| per target token")
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--drift-every", type=int, default=10)
    p.add_argument("--export-steps", default="", help="comma list of steps to export slim weights at")
    p.add_argument("--limit-rows", type=int, default=0, help="smoke test: first N train rows")
    p.add_argument("--limit-dev-rows", type=int, default=0, help="smoke test: first N dev rows")
    p.add_argument("--longest-first", action="store_true",
                   help="smoke test: rows sorted longest first, steps taken in that order (no shuffle)")
    p.add_argument("--local-cpu", action="store_true", help="CPU/tiny-model test path without DeepSpeed")
    return p.parse_args(argv)


PAD_ID = 248044  # <|endoftext|>; right padding never reaches a real token of a causal model


def target_logp(model, ids, lengths: list[int], n_targets: list[int]):
    """Per-sequence sums of the target tokens' log-probs for a right-padded batch `ids` [B, T]
    (sequence b is ids[b, :lengths[b]], its last n_targets[b] tokens are the targets). Returns [B] fp32.

    `lm_head` runs exactly once per call, on the target positions only (checkpointed under grad).
    Under ZeRO-3 each lm_head call is a collective: a data-dependent number of calls per forward
    (the first version chunked by 2,048 targets) sets the ranks' collectives out of step and hangs
    NCCL. A turn has at most ~4.1K targets, so one call's logits stay a few GB."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    hidden = model.model.language_model(input_ids=ids, use_cache=False).last_hidden_state
    rows, positions, owner = [], [], []
    for b, (length, n) in enumerate(zip(lengths, n_targets)):
        start = length - n - 1  # position t predicts token t+1
        rows.append(torch.full((n,), b, dtype=torch.long, device=ids.device))
        positions.append(torch.arange(start, start + n, device=ids.device))
        owner.append(torch.full((n,), b, dtype=torch.long, device=ids.device))
    rows_t, pos_t = torch.cat(rows), torch.cat(positions)
    selected = hidden[rows_t, pos_t]
    targets = ids[rows_t, pos_t + 1]

    def head(h, t):
        logits = model.lm_head(h).float()
        return torch.gather(F.log_softmax(logits, dim=-1), 1, t[:, None])[:, 0]

    token_logps = checkpoint(head, selected, targets, use_reentrant=False) if torch.is_grad_enabled() \
        else head(selected, targets)
    sums = torch.zeros(len(lengths), dtype=torch.float32, device=ids.device)
    return sums.index_add(0, torch.cat(owner), token_logps)


def make_sequence_module(model):
    """The module handed to DeepSpeed: forward(ids, lengths, n_targets) -> [B] summed target log-probs.

    Every forward must go through the engine. ZeRO-3 advances its micro-step (so the next
    backward adds to the gradient instead of overwriting it) and registers its backward hooks
    inside `engine.forward`; calling the model's submodules directly would keep only the last
    micro-batch's gradient of a step."""
    import torch

    class SequenceLogp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, ids, lengths, n_targets):
            return target_logp(self.model, ids, list(lengths), list(n_targets))

    return SequenceLogp()


def run_batch(runner, seqs, device):
    """A micro-batch of sequences through `runner` (the DeepSpeed engine, or the bare module on CPU)."""
    import torch

    lengths = [s.length for s in seqs]
    ids = torch.full((len(seqs), max(lengths)), PAD_ID, dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        ids[b, :s.length] = torch.tensor(s.prompt_ids + s.target_ids, dtype=torch.long, device=device)
    return runner(ids, lengths, [len(s.target_ids) for s in seqs])


def sequence_logp(runner, seq, device):
    """One sequence (a batch of one): 0-dim log-prob sum."""
    return run_batch(runner, [seq], device)[0]


def pack(items: list, length_of, max_tokens: int, max_targets: int, targets_of) -> list[list]:
    """Micro-batches of right-padded sequences: longest first, while batch x longest <= max_tokens
    and the targets stay <= max_targets (they bound the one lm_head call's logits)."""
    batches, current, longest, targets = [], [], 0, 0
    for item in sorted(items, key=length_of, reverse=True):
        n, t = length_of(item), targets_of(item)
        if current and (max(longest, n) * (len(current) + 1) > max_tokens or targets + t > max_targets):
            batches.append(current)
            current, longest, targets = [], 0, 0
        current.append(item)
        longest, targets = max(longest, n), targets + t
    if current:
        batches.append(current)
    return batches


def lr_at(step: int, total: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, total - args.warmup_steps)
    floor = args.lr * args.min_lr_ratio
    return floor + (args.lr - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def dpo_terms(pc, pr, rc, rr, nc, nr, args) -> dict[str, float]:
    """Loss, its derivatives w.r.t. the policy log-prob sums, and metrics (plain floats)."""
    if args.agg == "mean":
        dc, dr = (pc - rc) / nc, (pr - rr) / nr
        scale = args.beta * args.norm_tokens
        dh_dpc, dh_dpr = scale / nc, -scale / nr
    else:
        dc, dr = pc - rc, pr - rr
        scale = args.beta
        dh_dpc, dh_dpr = scale, -scale
    h = scale * (dc - dr)
    ls = args.label_smoothing
    sig = lambda x: 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
    logsig = lambda x: -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(math.exp(x))
    w = args.dpo_weight
    loss = w * (-(1 - ls) * logsig(h) - ls * logsig(-h)) + args.nll_weight * (-pc / nc)
    dloss_dh = w * (-(1 - ls) * sig(-h) + ls * sig(h))
    return {
        "loss": loss, "h": h,
        "coef_chosen": dloss_dh * dh_dpc - args.nll_weight / nc,
        "coef_rejected": dloss_dh * dh_dpr,  # exactly 0 with --dpo-weight 0: pass B then skips rejected
        "chosen_logratio": (pc - rc) / nc, "rejected_logratio": (pr - rr) / nr,
        "chosen_nll": -pc / nc, "accuracy": float(h > 0),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    ref_path = args.reference or args.output / "reference.jsonl"

    # ------------------------------------------------------------------ distributed + model
    if args.local_cpu:
        rank, world, device = 0, 1, torch.device("cpu")
        dist = None
    else:
        import deepspeed
        import torch.distributed as dist

        deepspeed.init_distributed(dist_backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)

    def log(msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with (args.output / f"rank{rank}.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {msg}\n")
        if rank == 0:
            print(f"{stamp} {msg}", flush=True)

    renderer = Renderer(args.tokenizer_dir) if args.tokenizer_dir else Renderer()
    module, engine = load_model(args, torch, rank, world, log)
    core = module.model  # the King: parameter names match its checkpoint
    runner = engine if engine is not None else module

    from vinhable_dpo_data import Sequence
    dummy = Sequence(prompt_ids=renderer.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"),
                     target_ids=[renderer.im_end])

    def padded_count(n: int) -> int:
        if dist is None:
            return n
        t = torch.tensor([n], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    def micro_batches(jobs: list[tuple]) -> list[list[tuple]]:
        """Jobs whose last element is a Sequence, packed into right-padded micro-batches."""
        return pack(jobs, lambda j: j[-1].length, args.tokens_per_forward, args.targets_per_forward,
                    lambda j: len(j[-1].target_ids))

    def forward_rows(rows: list[RowExample]) -> list[dict[str, float]]:
        """No-grad log-prob sums per row side. Every rank runs the same number of forwards (a dummy
        micro-batch pads the short ones) and each forward has the same collectives."""
        jobs = [(i, side, s) for i, r in enumerate(rows) for side in ("chosen", "rejected") for s in getattr(r, side)]
        out = [{"chosen": 0.0, "rejected": 0.0} for _ in rows]
        batches = micro_batches(jobs)
        with torch.no_grad():
            for k in range(padded_count(len(batches))):
                if k < len(batches):
                    sums = run_batch(runner, [j[-1] for j in batches[k]], device).tolist()
                    for (i, side, _), value in zip(batches[k], sums):
                        out[i][side] += value
                else:
                    run_batch(runner, [dummy], device)
        return out

    def sum_all(values: dict[str, float]) -> dict[str, float]:
        if dist is None:
            return values
        keys = sorted(values)
        t = torch.tensor([values[k] for k in keys], dtype=torch.float64, device=device)
        dist.all_reduce(t)
        return dict(zip(keys, t.tolist()))

    # ------------------------------------------------------------------ data
    train_rows = load_rows(args.data, "train")
    dev_rows = load_rows(args.data, "dev")
    if args.longest_first:  # smoke test: the most expensive rows decide whether memory holds
        train_rows.sort(key=char_length, reverse=True)
    if args.limit_rows:
        train_rows = train_rows[: args.limit_rows]
    if args.limit_dev_rows:
        dev_rows = dev_rows[: args.limit_dev_rows]

    def examples(rows: list[dict[str, Any]]) -> list[RowExample]:
        kept = []
        for row in rows:
            ex = row_example(renderer, row)
            if ex.chosen and ex.rejected and ex.max_length <= args.max_seq_tokens:
                kept.append(ex)
        return kept

    def my_share(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows[rank::world]

    # ------------------------------------------------------------------ reference phase
    if args.phase == "reference":
        started = time.time()
        part = args.output / f"reference.rank{rank}.jsonl"
        with part.open("w", encoding="utf-8") as handle:
            mine = examples(my_share(train_rows + dev_rows))
            per_call = max(1, args.reference_rows_per_call)
            calls = padded_count(math.ceil(len(mine) / per_call))
            done = 0
            for c in range(calls):
                batch = mine[c * per_call:(c + 1) * per_call]
                sums = forward_rows(batch)
                for ex, s in zip(batch, sums):
                    handle.write(json.dumps({"uid": ex.uid, "chosen": s["chosen"], "rejected": s["rejected"],
                                             "n_chosen": ex.tokens("chosen"), "n_rejected": ex.tokens("rejected")}) + "\n")
                handle.flush()
                done += len(batch)
                log(f"reference rank {rank}: {done}/{len(mine)} rows, {time.time() - started:.0f}s")
        if dist is not None:
            dist.barrier()
        if rank == 0:
            merged = {}
            for r in range(world):
                for line in (args.output / f"reference.rank{r}.jsonl").open(encoding="utf-8"):
                    row = json.loads(line)
                    if row["uid"] in merged:
                        raise RuntimeError(f"duplicate reference uid {row['uid']}")
                    merged[row["uid"]] = row
            with ref_path.open("w", encoding="utf-8") as handle:
                for row in merged.values():
                    handle.write(json.dumps(row) + "\n")
            log(f"reference: {len(merged)} rows in {time.time() - started:.0f}s -> {ref_path}")
        return 0

    # ------------------------------------------------------------------ train / export
    reference = {json.loads(l)["uid"]: json.loads(l) for l in ref_path.open(encoding="utf-8") if l.strip()}
    train_rows = [r for r in train_rows if r["sample_uid"] in reference]
    dev_rows = [r for r in dev_rows if r["sample_uid"] in reference]
    trainable = [(n, p) for n, p in core.named_parameters() if p.requires_grad]
    optimizer = engine.optimizer if engine is not None else torch.optim.AdamW(
        [p for _, p in trainable], lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    def full_fp32(p):
        if engine is None:
            return p.detach().float().cpu()
        from deepspeed.utils import safe_get_full_fp32_param
        value = safe_get_full_fp32_param(p)  # collective: every rank calls it for every tensor
        return value.detach().float().cpu() if rank == 0 else None

    def snapshot_trainable() -> dict[str, Any]:
        return {n: full_fp32(p) for n, p in trainable}

    def export(step: int, weights: dict[str, Any] | None = None) -> None:
        weights = weights or snapshot_trainable()
        if rank != 0:
            return
        from safetensors.torch import save_file
        folder = args.output / f"export-step{step}"
        folder.mkdir(exist_ok=True)
        save_file({n: w.to(torch.bfloat16).contiguous() for n, w in weights.items()}, str(folder / "trained.safetensors"),
                  metadata={"base_model": args.base_model, "base_revision": args.base_revision, "step": str(step)})
        (folder / "manifest.json").write_text(json.dumps({"step": step, "tensors": len(weights),
                                                          "base_model": args.base_model,
                                                          "base_revision": args.base_revision}, indent=1))
        log(f"export step {step}: {len(weights)} trained tensors -> {folder}")

    if args.phase == "export":
        export(0)
        return 0

    per_step = args.rows_per_rank * world
    steps_per_epoch = math.ceil(len(train_rows) / per_step)
    total_steps = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    export_steps = {int(s) for s in args.export_steps.split(",") if s.strip()}
    # every epoch end is exported too: the first run peaked on dev around epoch 1 and only its
    # final (overfit, degenerate) step had been kept
    export_steps |= {e * steps_per_epoch for e in range(1, math.ceil(args.epochs) + 1) if e * steps_per_epoch <= total_steps}
    log(f"train rows {len(train_rows)}, dev rows {len(dev_rows)}, trainable tensors {len(trainable)} "
        f"({sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel() for _, p in trainable) / 1e9:.3f}B), "
        f"{steps_per_epoch} steps/epoch, {total_steps} steps")

    def row_terms(ex: RowExample, sums: dict[str, float]) -> dict[str, float]:
        ref = reference[ex.uid]
        return dpo_terms(sums["chosen"], sums["rejected"], ref["chosen"], ref["rejected"],
                         max(1, ref["n_chosen"]), max(1, ref["n_rejected"]), args)

    def evaluate(tag: str) -> dict[str, float]:
        mine = examples(my_share(dev_rows))
        totals = {"rows": 0.0, "accuracy": 0.0, "loss": 0.0, "margin": 0.0, "chosen_logratio": 0.0,
                  "rejected_logratio": 0.0}
        for ex, s in zip(mine, forward_rows(mine)):
            t = row_terms(ex, s)
            totals["rows"] += 1
            totals["accuracy"] += t["accuracy"]
            totals["loss"] += t["loss"]
            totals["margin"] += t["h"]
            totals["chosen_logratio"] += t["chosen_logratio"]
            totals["rejected_logratio"] += t["rejected_logratio"]
        totals = sum_all(totals)
        n = max(1.0, totals.pop("rows"))
        result = {f"dev_{k}": v / n for k, v in totals.items()}
        result["dev_rows"] = n
        log(f"eval {tag}: " + json.dumps({k: round(v, 5) for k, v in result.items()}))
        return result

    # step-0 gate: the policy must reproduce the reference before any update
    gate = examples(my_share(train_rows)[: args.gate_rows])
    sums = forward_rows(gate)
    diffs = {"abs_diff": 0.0, "tokens": 0.0}
    for ex, s in zip(gate, sums):
        ref = reference[ex.uid]
        diffs["abs_diff"] += abs(s["chosen"] - ref["chosen"]) + abs(s["rejected"] - ref["rejected"])
        diffs["tokens"] += ref["n_chosen"] + ref["n_rejected"]
    diffs = sum_all(diffs)
    per_token = diffs["abs_diff"] / max(1.0, diffs["tokens"])
    log(f"step-0 gate: mean |policy - reference| per target token = {per_token:.2e} (limit {args.gate_max_diff})")
    if per_token > args.gate_max_diff and not args.skip_gate:
        raise SystemExit("step-0 gate failed: reference and policy paths disagree; stopping before any update")

    metrics_path = args.output / "metrics.jsonl"

    def write(record: dict[str, Any]) -> None:
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            tmp = args.output / "progress.json.tmp"
            tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
            os.replace(tmp, args.output / "progress.json")

    write({"step": 0, "gate_per_token_diff": per_token, **evaluate("step0")})
    initial = snapshot_trainable() if args.drift_every else None
    last_snapshot, lr_sq_sum, lr_sq_at_last = initial, 0.0, 0.0
    started = time.time()
    step = 0
    epoch = 0
    while step < total_steps:
        schedule = ([train_rows[i:i + per_step] for i in range(0, len(train_rows), per_step)] if args.longest_first
                    else batches(train_rows, per_step, args.seed, epoch))
        for step_rows in schedule:
            if step >= total_steps:
                break
            lr = lr_at(step, total_steps, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            ordered = step_rows if args.longest_first else sorted(step_rows, key=lambda r: r["sample_uid"])
            mine = examples(ordered[rank::world])
            # pass A: policy log-prob sums, then the loss derivative per side
            sums = forward_rows(mine)
            terms = [row_terms(ex, s) for ex, s in zip(mine, sums)]
            # pass B: back-propagate each micro-batch, every sequence scaled by its side's derivative
            scale = 1.0 / args.rows_per_rank  # DeepSpeed averages over ranks: the step is a row mean
            jobs = [(t[f"coef_{side}"] * scale, s) for ex, t in zip(mine, terms)
                    for side in ("chosen", "rejected") for s in getattr(ex, side) if t[f"coef_{side}"] != 0.0]
            packed = micro_batches(jobs)
            n_batches = padded_count(len(packed))
            for k in range(n_batches):
                if k < len(packed):
                    coefs = torch.tensor([c for c, _ in packed[k]], dtype=torch.float32, device=device)
                    loss = (coefs * run_batch(runner, [s for _, s in packed[k]], device)).sum()
                else:
                    loss = 0.0 * run_batch(runner, [dummy], device).sum()
                if engine is not None:
                    engine.set_gradient_accumulation_boundary(k == n_batches - 1)
                    engine.backward(loss, scale_wrt_gas=False)
                else:
                    loss.backward()
            if engine is not None:
                engine.step()
                grad_norm = float(engine.get_global_grad_norm() or 0.0)
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            lr_sq_sum += lr * lr
            agg = sum_all({"rows": float(len(terms)), "loss": sum(t["loss"] for t in terms),
                           "accuracy": sum(t["accuracy"] for t in terms), "margin": sum(t["h"] for t in terms),
                           "chosen_logratio": sum(t["chosen_logratio"] for t in terms),
                           "rejected_logratio": sum(t["rejected_logratio"] for t in terms),
                           "chosen_nll": sum(t["chosen_nll"] for t in terms),
                           "sequences": float(len(jobs)),
                           "tokens": float(sum(s.length for _, s in jobs))})
            n = max(1.0, agg["rows"])
            elapsed = time.time() - started
            record = {"step": step, "epoch": epoch, "lr": lr, "grad_norm": grad_norm,
                      **{k: agg[k] / n for k in ("loss", "accuracy", "margin", "chosen_logratio",
                                                  "rejected_logratio", "chosen_nll")},
                      "rows": agg["rows"], "sequences": agg["sequences"], "tokens": agg["tokens"],
                      "tokens_per_s": agg["tokens"] / max(1e-6, elapsed / step),
                      "elapsed_s": round(elapsed, 1), "eta_s": round(elapsed / step * (total_steps - step), 1)}
            if not args.local_cpu:
                record["peak_vram_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
            if args.eval_every and step % args.eval_every == 0:
                record.update(evaluate(f"step{step}"))
            if args.drift_every and step % args.drift_every == 0:
                now = snapshot_trainable()
                if rank == 0:
                    record.update(drift_report(now, initial, last_snapshot, lr_sq_sum, lr_sq_sum - lr_sq_at_last))
                last_snapshot, lr_sq_at_last = now, lr_sq_sum
                if step in export_steps:
                    export(step, now)
            elif step in export_steps:
                export(step)
            write(record)
            log(f"step {step}/{total_steps} " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                                              for k, v in record.items()
                                                              if k not in ("elapsed_s",)}))
        epoch += 1
    final = snapshot_trainable()
    if rank == 0 and initial is not None:
        write({"step": step, "final": True, **drift_report(final, initial, last_snapshot, lr_sq_sum,
                                                           lr_sq_sum - lr_sq_at_last), **evaluate("final")})
    else:
        evaluate("final")
    if step not in export_steps:
        export(step, final)
    if dist is not None:
        dist.barrier()
    return 0


def drift_report(now: dict, initial: dict, previous: dict, lr_sq: float, lr_sq_increment: float) -> dict[str, float]:
    """RMS fp32 drift from the King per family, against the Adam noise line sqrt(sum lr^2).

    Adam moves every element by about lr per step whatever the gradient, so a random walk drifts
    sqrt(sum lr^2); a directed update grows with sum lr and pulls clear of that line. The cosine
    between the last increment and the total drift is ~0 for noise."""
    import torch

    out: dict[str, float] = {}
    sq = {f: 0.0 for f in FAMILIES}
    cnt = {f: 0 for f in FAMILIES}
    dot = norm_inc = norm_tot = 0.0
    for name, value in now.items():
        f = family(name)
        delta = value - initial[name]
        inc = value - previous[name]
        sq[f] += float((delta * delta).sum())
        cnt[f] += delta.numel()
        dot += float((delta * inc).sum())
        norm_inc += float((inc * inc).sum())
        norm_tot += float((delta * delta).sum())
    noise = math.sqrt(lr_sq)
    total_sq, total_cnt = sum(sq.values()), sum(cnt.values())
    out["drift_rms"] = math.sqrt(total_sq / max(1, total_cnt))
    out["noise_line"] = noise
    out["drift_over_noise"] = out["drift_rms"] / noise if noise else float("nan")
    for f in FAMILIES:
        if cnt[f]:
            out[f"drift_over_noise_{f}"] = math.sqrt(sq[f] / cnt[f]) / noise if noise else float("nan")
    out["increment_rms"] = math.sqrt(norm_inc / max(1, total_cnt))
    out["increment_over_noise"] = out["increment_rms"] / math.sqrt(lr_sq_increment) if lr_sq_increment else float("nan")
    out["increment_cos_total"] = dot / math.sqrt(max(1e-30, norm_inc * norm_tot))
    return out


def load_model(args, torch, rank: int, world: int, log):
    """King 127 as Qwen3_5MoeForConditionalGeneration (names match the checkpoint), trainable set frozen
    in place, wrapped in a ZeRO-3 engine with a torch optimizer (no nvcc-built DeepSpeed op)."""
    from transformers import AutoConfig, Qwen3_5MoeForConditionalGeneration

    if args.local_cpu:
        config = AutoConfig.from_pretrained(args.base_model)
        model = Qwen3_5MoeForConditionalGeneration(config) if (Path(args.base_model) / "random_init").exists() \
            else Qwen3_5MoeForConditionalGeneration.from_pretrained(args.base_model)
        model = model.to(torch.bfloat16)  # the GPU run is bf16 throughout; so is this check
        engine = None
    else:
        import deepspeed
        from transformers.integrations import HfDeepSpeedConfig

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            # One step spans a variable number of backwards; the boundary is set explicitly before
            # each one (set_gradient_accumulation_boundary) and backward never scales by this.
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1_000_000_000,
            "gradient_clipping": args.max_grad_norm,
            "bf16": {"enabled": True},
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 3, "overlap_comm": True, "contiguous_gradients": True,
                "reduce_bucket_size": 2e8, "stage3_prefetch_bucket_size": 2e8,
                "stage3_param_persistence_threshold": 1e6, "stage3_max_live_parameters": 1e9,
                "stage3_gather_16bit_weights_on_model_save": False,
            },
            "wall_clock_breakdown": False,
        }
        keepalive = HfDeepSpeedConfig(ds_config)  # noqa: F841 - must exist while loading (zero.Init)
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            args.base_model, revision=None if Path(args.base_model).exists() else args.base_revision,
            dtype=torch.bfloat16, attn_implementation=args.attn_implementation,
            # asked explicitly: transformers silently falls back to a per-expert loop otherwise
            experts_implementation="grouped_mm")
    model.config.use_cache = False
    for name, p in model.named_parameters():
        p.requires_grad = bool(TRAINABLE.match(name))
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("no trainable tensor matched; check the parameter names")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    module = make_sequence_module(model)
    if args.local_cpu:
        return module, None
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    engine, _, _, _ = deepspeed.initialize(model=module, optimizer=optimizer, config=ds_config)
    log(f"loaded {args.base_model} on {world} ranks")
    return module, engine


if __name__ == "__main__":
    try:
        from torch.distributed.elastic.multiprocessing.errors import record
        main = record(main)
    except ImportError:
        pass
    raise SystemExit(main())
