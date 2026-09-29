#!/usr/bin/env python3
"""King's NLL on a lab-packed pair set, from the lab trainer's reference pass (a gate-risk proxy).

The dedup gate needs a directed weight change. How far a pair set can pull the King is bounded by how
unlikely its targets already are under the King: GLM text started at NLL ~0.8/token (v3 passed the
gate at only 1.1x), while the King's own rollouts should sit near its sampling entropy. This reads the
per-token log p_ref that `lab_train/train.py --mode ref` writes and reports, per side, the
token-weighted NLL overall / thinking / reply, and the per-row spread.

    # 1. a small subset of each set, so the reference pass takes minutes (on the box)
    python scripts/measure_king_nll.py subset --data DATA/lab --out DATA/nll --rows 300
    # 2. reference pass (8 GPUs), writes OUT/ref/{train,dev}-rank*.pt
    python -m torch.distributed.run --nproc-per-node 8 LAB/scripts/lab_train/train.py \
        --model KING_DIR --data DATA/nll --out RUN --mode ref
    # 3. the numbers
    python scripts/measure_king_nll.py summarize --data DATA/nll --ref RUN/ref --label king_selfpairs
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path


def subset(args: argparse.Namespace) -> None:
    import torch

    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        records = torch.load(args.data / f"{split}.pt", weights_only=False)
        random.Random(args.seed).shuffle(records)
        keep = records[: args.rows]
        torch.save(keep, args.out / f"{split}.pt")
        print(split, len(records), "->", len(keep))


def summarize(args: argparse.Namespace) -> None:
    import torch

    records = {}
    for split in ("train", "dev"):
        path = args.data / f"{split}.pt"
        if path.is_file():
            for r in torch.load(path, weights_only=False):
                records[r["meta"]["pair_id"]] = r
    ref = {}
    for f in sorted(args.ref.glob("*-rank*.pt")):
        ref.update(torch.load(f, weights_only=False))
    missing = [k for k in records if k not in ref]
    totals = {s: {"all": [0.0, 0], "think": [0.0, 0], "reply": [0.0, 0]} for s in ("chosen", "rejected")}
    per_row = {"chosen": [], "rejected": []}
    for pid, rec in records.items():
        if pid not in ref:
            continue
        for side, lp in zip(("chosen", "rejected"), ref[pid]):
            part = rec[side]["part"][rec[side]["target_mask"].bool()]
            if len(part) != len(lp):
                raise ValueError(f"{pid} {side}: {len(lp)} log-probs for {len(part)} targets")
            nll = -lp.double()
            for name, mask in (("all", torch.ones_like(part, dtype=torch.bool)), ("think", part == 1), ("reply", part == 2)):
                totals[side][name][0] += float(nll[mask].sum())
                totals[side][name][1] += int(mask.sum())
            per_row[side].append(float(nll.mean()))
    out = {"label": args.label, "pairs": len(records) - len(missing), "missing_ref": len(missing)}
    for side in ("chosen", "rejected"):
        out[side] = {name: {"nll_per_token": round(s / max(n, 1), 4), "tokens": n} for name, (s, n) in totals[side].items()}
        rows = sorted(per_row[side])
        if rows:
            out[side]["row_nll"] = {"median": round(st.median(rows), 4), "p10": round(rows[len(rows) // 10], 4),
                                    "p90": round(rows[9 * len(rows) // 10], 4)}
    print(json.dumps(out, indent=1))
    if args.save:
        args.save.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--rows", type=int, default=300)
    s.add_argument("--seed", type=int, default=20260929)
    m = sub.add_parser("summarize")
    m.add_argument("--data", type=Path, required=True)
    m.add_argument("--ref", type=Path, required=True)
    m.add_argument("--label", default="")
    m.add_argument("--save", type=Path)
    args = parser.parse_args()
    subset(args) if args.stage == "subset" else summarize(args)


if __name__ == "__main__":
    main()
