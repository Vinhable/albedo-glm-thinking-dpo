#!/usr/bin/env python3
"""Convert `vinhable_single` / `vinhable_double` rows to the lab trainer's branch-packed records.

The lab trainer (`albedo-lab-dpo/scripts/lab_train`, FSDP2, one forward per side) packs a side as
trunk pieces + one branch per supervised turn (render.py). Its "thinking" mode takes one entry per
assistant turn: the reasoning of a supervised turn, or None for a context-only turn. A row here
supervises only its last behaviour group (`loss` flags), so each side becomes

    thinking[t] = reasoning of turn t   if vinhable_dpo_data.supervised(turn t)   else None
    content[t]  = the action after "\\n</think>\\n\\n" for supervised turns (the lab rebuilds the
                  generation as reasoning + "\\n</think>\\n\\n" + action + <|im_end|>)

Every record is checked token by token against our own per-turn rendering (vinhable_dpo_data.py,
the path train_vinhable_dpo.py trains on): for each supervised turn, the trunk before the branch
plus "<think>\\n" must equal our prompt ids, and the branch targets must equal our target ids.
A row that fails is dropped and counted; the run fails if any row fails unless --allow-mismatch.

Output: <out>/train.pt, <out>/dev.pt (list of {"meta", "chosen", "rejected"}, tensors as prep.py
writes them, meta.pair_id = our sample_uid) and <out>/prep-summary.json.

    python scripts/prep_vinhable_for_lab.py --data DATA_DIR --lab-dir /path/to/albedo-lab-dpo --out OUT
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vinhable_dpo_data import Renderer, load_rows, side_sequences, supervised  # noqa: E402

SEPARATOR = "\n</think>\n\n"
STATE: dict[str, Any] = {}


class Mismatch(ValueError):
    pass


def split_turn(content: str) -> tuple[str, str]:
    """reasoning, action such that reasoning + SEPARATOR + action == content."""
    head, sep, tail = content.partition(SEPARATOR)
    if not sep or "</think>" in head:
        raise Mismatch("turn does not close its reasoning with '\\n</think>\\n\\n'")
    return head, tail


def lab_side(prompt: list[dict], side: list[dict]) -> tuple[list[dict], list[str | None]]:
    messages, thinking = [], []
    for m in side:
        if m["role"] == "assistant":
            if supervised(m):
                reasoning, action = split_turn(m["content"])
                messages.append({"role": "assistant", "content": action})
                thinking.append(reasoning)
            else:
                messages.append({"role": "assistant", "content": m["content"]})
                thinking.append(None)
        else:
            messages.append({"role": m["role"], "content": m["content"]})
    return messages, thinking


def check(ps, ours) -> None:
    """Lab packed side vs our per-turn sequences, turn by turn."""
    render = STATE["render"]
    think_len = len(render.encode(render.THINK_OPEN))
    if ps.n_turns != len(ours):
        raise Mismatch(f"{ps.n_turns} lab branches vs {len(ours)} supervised turns")
    starts = {}
    for i, seg in enumerate(ps.segment):
        if seg >= 0 and seg not in starts:
            starts[seg] = i
    for (seg, start), seq in zip(sorted(starts.items(), key=lambda kv: kv[1]), ours):
        trunk = [tok for tok, s in zip(ps.input_ids[:start], ps.segment[:start]) if s == -1]
        branch = [i for i in range(start, len(ps.segment)) if ps.segment[i] == seg]
        context = trunk + [ps.input_ids[i] for i in branch[:think_len]]
        targets = [ps.input_ids[i] for i in branch if ps.target_mask[i]]
        if context != seq.prompt_ids:
            raise Mismatch(f"turn {seg}: context differs ({len(context)} vs {len(seq.prompt_ids)} tokens)")
        if targets != seq.target_ids:
            raise Mismatch(f"turn {seg}: targets differ ({len(targets)} vs {len(seq.target_ids)} tokens)")
        positions = [ps.position_ids[i] for i in branch]
        if positions != list(range(len(trunk), len(trunk) + len(branch))):
            raise Mismatch(f"turn {seg}: branch positions do not continue the trunk")


def pack(prompt: list[dict], side: list[dict]):
    """Plain lists (tensors are built in the parent: torch tensors sent back from pool workers pass file
    descriptors and ran out of them on the box, 'received 0 items of ancdata')."""
    render, ours_renderer = STATE["render"], STATE["ours"]
    ours = side_sequences(ours_renderer, prompt, side)
    messages, thinking = lab_side(prompt, side)
    ps = render.render_side(prompt, messages, max_tokens=STATE["max_tokens"] or None, measure_verbatim=False,
                            thinking=thinking)
    if ps.capped:
        raise Mismatch("side exceeds --max-tokens")
    check(ps, ours)
    lists = {"input_ids": ps.input_ids, "position_ids": ps.position_ids, "segment": ps.segment,
             "target_mask": ps.target_mask, "part": ps.part}
    per_turn = sum(s.length for s in ours)
    return lists, {"packed": len(ps.input_ids), "per_turn": per_turn, "targets": int(sum(ps.target_mask)),
                     "turns": ps.n_turns}


def work(line: str):
    row = json.loads(line)
    try:
        chosen, cs = pack(row["prompt"], row["chosen"])
        rejected, rs = pack(row["prompt"], row["rejected"])
    except (Mismatch, ValueError) as error:  # render.RenderError is a ValueError
        return None, {"error": f"{type(error).__name__}: {error}", "uid": row["sample_uid"]}
    meta = {"pair_id": row["sample_uid"], "task_pair_id": row["pair_id"], "direction": row["direction"],
            "prefix_groups": row.get("prefix_groups"), "split": row["split"]}
    return {"meta": meta, "chosen": chosen, "rejected": rejected}, {"c": cs, "r": rs}


def init(lab_dir: str, max_tokens: int) -> None:
    sys.path.insert(0, str(Path(lab_dir) / "scripts" / "lab_train"))
    import render  # the lab's module (transformers tokenizer + the genesis chat template)

    STATE.update(render=render, ours=Renderer(), max_tokens=max_tokens)


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="dataset dir with train.jsonl / dev.jsonl")
    parser.add_argument("--lab-dir", type=Path, required=True, help="checkout of the lab repo (albedo-lab-dpo)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="cap on a packed side (0: none). A packed side can exceed our per-turn cap while every "
                             "turn fits, so the default keeps exactly the rows train_vinhable_dpo.py trains on")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()
    DTYPES = {"input_ids": torch.int32, "position_ids": torch.int32, "segment": torch.int16,
              "target_mask": torch.bool, "part": torch.int8}  # as the lab's prep.py writes them

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"args": {k: str(v) for k, v in vars(args).items()}}
    failed = 0
    for split in ("dev", "train"):
        rows = load_rows(args.data, split)  # checks sample_uid uniqueness
        lines = [json.dumps(r) for r in rows][: args.limit or None]
        records, errors = [], {}
        totals = {"packed": 0, "per_turn": 0, "targets": 0, "turns": 0}
        longest = 0
        with mp.Pool(args.workers, initializer=init, initargs=(str(args.lab_dir), args.max_tokens)) as pool:
            for record, stats in pool.imap(work, lines, chunksize=4):
                if record is None:
                    errors.setdefault(stats["error"], []).append(stats["uid"])
                    continue
                for side in ("chosen", "rejected"):
                    record[side] = {k: torch.tensor(v, dtype=DTYPES[k]) for k, v in record[side].items()}
                records.append(record)
                for side in ("c", "r"):
                    for k in totals:
                        totals[k] += stats[side][k]
                    longest = max(longest, stats[side]["packed"])
        torch.save(records, args.out / f"{split}.pt")
        failed += sum(len(v) for v in errors.values())
        summary[split] = {
            "rows": len(lines), "records": len(records),
            "mismatches": {k: {"count": len(v), "first": v[:3]} for k, v in errors.items()},
            **totals, "per_turn_over_packed": round(totals["per_turn"] / max(1, totals["packed"]), 3),
            "longest_packed_side": longest,
        }
        print(split, json.dumps(summary[split]), flush=True)
    (args.out / "prep-summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if failed and not args.allow_mismatch:
        print(f"{failed} rows failed the token-level check; rerun with --allow-mismatch to drop them", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
