#!/usr/bin/env python3
"""Build `king_selfpairs`: King 127's better vs worse production rollout of the same task (no API).

Direction 1 of the self-improvement plan: best-of-2 as the policy-improvement step. Source is the
graded crawl; both rollouts are the King's own generations, scored by the graded judge on the same
checklist, so the pair differs only in what the King did.

Pairs
    A graded task with two usable King rollouts and |r1 - r2| >= --margin. Chosen = the higher score.
    Skipped: the winner has amputated thinking, the task was already used (GLM batches, pilots, duel).
Rows
    The same behaviour-group rows as `vinhable_single` (build_vinhable_glm_pairs.grouped_rows): both
    sides >= 4 assistant turns, min(6, turns) groups, row k supervises group k. The same `loss` rules:
    no loss on observations, harness notices, empty think blocks, turns over 4,096 tokens.
Split: task-level hash, --dev-share to dev.

    python scripts/build_king_selfpairs.py --out E:/albedo-storage-temp/king-selfpairs-20260929
Then `prep_vinhable_for_lab.py --data OUT/rows --lab-dir ../albedo-lab-dpo --out OUT/lab` for the lab trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from analyze_king_pair_pool import used_samples  # noqa: E402
from build_gen_vinhable import rollouts, run_index  # noqa: E402
from build_vinhable_glm_pairs import Turns, grouped_rows  # noqa: E402
from vinhable_dpo_data import supervised  # noqa: E402

SCHEMA = "albedo-king-selfpairs-v1"


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crawl-root", type=Path, default=Path("E:/albedo-storage-temp/raw-rollouts-graded-20260927"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--dev-share", type=float, default=0.10)
    args = parser.parse_args()

    used = used_samples()
    turns = Turns()
    counts: Counter[str] = Counter()
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run in run_index([args.crawl_root]):
        for sid, info, found in rollouts(run, counts):
            if info.get("scoring_mode") != "graded_20":
                counts["skip:not_graded"] += 1
                continue
            kings = sorted((r for r in found if r["side"] == "king"), key=lambda r: r["replica"])
            if len(kings) < 2:
                counts["skip:fewer_than_2_king"] += 1
                continue
            if sid in used:
                counts["skip:already_used"] += 1
                continue
            if sid in seen:  # the same sample in a later run: keep the first (newest) run only
                counts["skip:repeat_sample"] += 1
                continue
            best, worst = sorted(kings[:2], key=lambda r: -r["score"])
            margin = best["score"] - worst["score"]
            if margin < args.margin:
                counts["skip:below_margin"] += 1
                continue
            if best.get("amputated"):
                counts["skip:amputated_winner"] += 1
                continue
            if [m["content"] for m in best["context"]] != [m["content"] for m in worst["context"]]:
                counts["skip:contexts_differ"] += 1
                continue
            seen.add(sid)
            task_id = f"{run['eval_run_id']}:{sid}"
            chosen_id, rejected_id = f"king#r{best['replica']}", f"king#r{worst['replica']}"
            pairs.append({
                "schema": SCHEMA,
                "pair_id": sha(f"{task_id}|{chosen_id}|{rejected_id}")[:32],
                "task_id": task_id, "eval_run_id": run["eval_run_id"], "sample_id": sid,
                "split": "dev" if int(sha(sid)[:8], 16) % 1000 < args.dev_share * 1000 else "train",
                "source": sid.split("/", 1)[0], "sample_phase": info.get("sample_phase"),
                "question_mode": info.get("question_mode"), "king_version": run["king_version"],
                "direction": "king_best_over_worst", "chosen_side": chosen_id, "rejected_side": rejected_id,
                "chosen_score": round(best["score"], 6), "rejected_score": round(worst["score"], 6),
                "margin": round(margin, 6),
                "prompt": [{"role": m["role"], "content": m["content"], "loss": False} for m in best["context"]],
                "chosen": turns.side(best["completion"], king=True),
                "rejected": turns.side(worst["completion"], king=True),
            })
            counts["pair"] += 1

    folder = args.out / "rows"
    folder.mkdir(parents=True, exist_ok=True)
    group_counts: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    kept_pairs: Counter[str] = Counter()
    with (folder / "train.jsonl").open("w", encoding="utf-8") as train, \
            (folder / "dev.jsonl").open("w", encoding="utf-8") as dev:
        for pair in pairs:
            produced = []
            for row in grouped_rows(pair, group_counts):
                # a `loss` turn that never closes </think> is not trained (vinhable_dpo_data.supervised);
                # a side left with no trained turn gives DPO nothing to compare (the lab renderer rejects it)
                if all(any(supervised(m) for m in row[side]) for side in ("chosen", "rejected")):
                    produced.append(row)
                else:
                    group_counts["grouped_rows_dropped:side_without_trained_turn"] += 1
            if produced:
                kept_pairs[pair["split"]] += 1
            for row in produced:
                by_split[row["split"]] += 1
                (dev if row["split"] == "dev" else train).write(json.dumps(row, ensure_ascii=False) + "\n")
    loss_turns = Counter()
    for pair in pairs:
        for side in ("chosen", "rejected"):
            loss_turns[side] += sum(1 for t in pair[side] if t["role"] == "assistant" and t["loss"])
    summary = {
        "schema": SCHEMA, "margin_min": args.margin, "pairs": len(pairs),
        "pairs_by_split": dict(Counter(p["split"] for p in pairs)),
        "by_phase": dict(Counter(p["sample_phase"] for p in pairs)),
        "by_source": dict(Counter(p["source"] for p in pairs)),
        "margin_median": round(st.median(p["margin"] for p in pairs), 4) if pairs else None,
        "chosen_score_mean": round(st.mean(p["chosen_score"] for p in pairs), 4) if pairs else None,
        "rejected_score_mean": round(st.mean(p["rejected_score"] for p in pairs), 4) if pairs else None,
        "loss_turns_before_grouping": dict(loss_turns),
        "grouped": {"rows": sum(by_split.values()), "rows_by_split": dict(by_split),
                    "pairs_with_rows": dict(kept_pairs), "counts": dict(group_counts)},
        "build_counts": dict(counts),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
