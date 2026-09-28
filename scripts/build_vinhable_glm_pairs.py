#!/usr/bin/env python3
"""Build `vinhable_single` and `vinhable_double`: GLM-5.2-with-thinking vs King 127 DPO pairs.

Source: a GLM regeneration batch merged with its reruns (`merge_glm_regen_results.py`) plus the
King's two production rollouts of each task from the crawled artifacts.

Pairs (margin >= --margin on the graded 0-1 scale)
    A task's GLM rollout is compared with each of the King's two rollouts. The pair's winner is the
    higher score: GLM chosen / King rejected, or King chosen / GLM rejected.
    single  one pair per task: the qualifying comparison with the largest gap.
    double  every qualifying comparison, so up to two pairs per task (both share the GLM side).
Both sets use the same task-level train/dev split, so they can be compared directly.

Row = one trajectory-level pair
    `prompt`   production context up to the cut point (loss false)
    `chosen`, `rejected`   the two continuations: assistant turns and environment observations
Assistant `content` is the text after `<|im_start|>assistant\\n<think>\\n`: reasoning, `</think>`,
action (the King's own generation; GLM's reasoning and action assembled the same way).
`loss: false` on observations, harness notices, turns whose think block is empty, and any assistant
turn longer than 4,096 Qwen3.6 tokens (production's per-turn limit).

    py -3 scripts/build_vinhable_glm_pairs.py --batch E:/albedo-storage-temp/glm-thinking-batch300-20260927 \
        --out E:/albedo-storage-temp/vinhable-glm-pairs-20260927
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from build_gen_vinhable import rollouts, run_index  # noqa: E402
from build_joint_behavior_groups import validate_groups  # noqa: E402
from build_joint_behavior_groups_v2 import atomic_units, grouped  # noqa: E402

SCHEMA = "albedo-glm-thinking-dpo-v1"
TURN_TOKEN_LIMIT = 4096
TOKENIZER = ROOT / "assets" / "tokenizers" / "Qwen3.6-35B-A3B" / "tokenizer.json"


# GLM was steered by a generation-only instruction to reason through five labelled points; on ~8% of
# turns it also wrote them into the visible answer. The stored prompt has no such instruction, so
# those answers would teach headers out of nowhere: they stay as context without loss.
_NUDGE_HEADER = re.compile(r"(Observation|Evidence|Open questions|Options|Decision)\s*(\*\*)?\s*[:：]", re.I)


def nudge_echo(action: str) -> bool:
    return len({m.group(1).lower() for m in _NUDGE_HEADER.finditer(action)}) >= 2


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Turns:
    def __init__(self) -> None:
        self.tok = Tokenizer.from_file(str(TOKENIZER))

    def tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids) + 1  # + <|im_end|>

    def assistant(self, content: str, *, harness: bool = False) -> dict[str, Any]:
        n = self.tokens(content)
        empty_think = content.lstrip().startswith("</think>")
        turn = {"role": "assistant", "content": content, "think_closed": "</think>" in content, "tokens": n,
                "loss": not harness and not empty_think and bool(content.strip()) and n <= TURN_TOKEN_LIMIT}
        if harness:
            turn["harness_text"] = True
        if empty_think:  # never teach an empty think block: Albedo halves such rollouts
            turn["empty_think"] = True
        if n > TURN_TOKEN_LIMIT:
            turn["over_turn_limit"] = True
        return turn

    def side(self, turns: list[dict[str, Any]], *, king: bool) -> list[dict[str, Any]]:
        out = []
        for t in turns:
            if t["role"] == "assistant":
                harness = bool(t.get("harness_text") if king else t.get("truncated"))
                turn = self.assistant(t["content"], harness=harness)
                if not king and nudge_echo(t.get("action") or ""):
                    turn["loss"] = False
                    turn["nudge_echo"] = True
                out.append(turn)
            else:
                out.append({"role": "user", "content": t["content"], "loss": False})
        return out


def grouped_rows(pair: dict[str, Any], counts: Counter[str], *, first_group_match: bool = False) -> list[dict[str, Any]]:
    """`gen_vinhable`'s turn-group split of one trajectory-level pair.

    Both sides need >= 4 assistant turns; each is cut into the same number of behaviour groups,
    min(6, turns). Row k holds groups 1..k of each side and supervises only group k's assistant
    turns (a turn's own `loss: false` stays). `gen_vinhable` also required the first groups to share
    their dominant behaviour; here that is off unless `first_group_match` (the two sides are
    different models, so their openings legitimately differ)."""
    context = [{"role": m["role"], "content": m["content"]} for m in pair["prompt"]]
    units = {side: atomic_units(pair[side], context) for side in ("chosen", "rejected")}
    turns = min(len(units["chosen"]), len(units["rejected"]))
    if turns < 4:
        counts["grouped_drop:fewer_than_four_turns"] += 1
        return []
    total = min(6, turns)
    groups = {side: grouped(units[side], total) for side in ("chosen", "rejected")}
    if first_group_match and groups["chosen"][0]["dominant_behavior"] != groups["rejected"][0]["dominant_behavior"]:
        counts["grouped_drop:first_group_mismatch"] += 1
        return []
    for side in ("chosen", "rejected"):
        validate_groups(pair[side], groups[side], total)

    def flatten(side: str, upto: int) -> list[dict[str, Any]]:
        out = []
        for index, group in enumerate(groups[side][:upto], 1):
            for message in group["messages"]:
                item = dict(message)
                item["loss"] = bool(message.get("loss")) and message["role"] == "assistant" and index == upto
                out.append(item)
        return out

    aligned = sum(a["dominant_behavior"] == b["dominant_behavior"] for a, b in zip(groups["chosen"], groups["rejected"]))
    base = {k: v for k, v in pair.items() if k not in ("chosen", "rejected")}
    rows = []
    for k in range(1, total + 1):
        chosen, rejected = flatten("chosen", k), flatten("rejected", k)
        if not any(m["loss"] for m in chosen) or not any(m["loss"] for m in rejected):
            counts["grouped_rows_dropped:no_supervised_turn"] += 1
            continue
        rows.append({**base, "sample_uid": f"{pair['pair_id']}#k{k}", "prefix_groups": k, "group_count": total,
                     "is_full_trajectory": k == total, "chosen": chosen, "rejected": rejected,
                     "aligned_groups": aligned, "aligned_fraction": round(aligned / total, 6),
                     "chosen_behavior_sequence": [g["dominant_behavior"] for g in groups["chosen"][:k]],
                     "rejected_behavior_sequence": [g["dominant_behavior"] for g in groups["rejected"][:k]]})
    counts["grouped_pairs"] += 1
    return rows


def king_rollouts(crawl_root: Path, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], dict[int, dict]]:
    found: dict[tuple[str, str], dict[int, dict]] = {}
    for run in run_index([crawl_root]):
        for sid, info, rolls in rollouts(run, Counter()):
            key = (run["eval_run_id"], sid)
            if key in wanted:
                found[key] = {r["replica"]: {**r, "info": info, "finished_at": run["finished_at"]}
                              for r in rolls if r["side"] == "king"}
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--crawl-root", type=Path, default=Path("E:/albedo-storage-temp/raw-rollouts-graded-20260927"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--dev-share", type=float, default=0.10)
    parser.add_argument("--first-group-match", action="store_true",
                        help="gen_vinhable's rule: drop pairs whose first behaviour groups differ (off by default)")
    args = parser.parse_args()

    tasks = {t["task_id"]: t for t in read_jsonl(args.batch / "pilot-input.jsonl")}
    merged = [r for r in read_jsonl(args.batch / "merged.jsonl")
              if r["glm_score"] is not None and r["stop"] != "truncated" and not r.get("glm_amputated")]
    kings = king_rollouts(args.crawl_root, {(t["task_id"].split(":", 1)[0], t["sample_id"]) for t in merged})
    turns = Turns()
    counts: Counter[str] = Counter()
    sets: dict[str, list[dict[str, Any]]] = {"single": [], "double": []}

    for r in merged:
        task = tasks[r["task_id"]]
        eval_id = r["task_id"].split(":", 1)[0]
        king_by_replica = kings.get((eval_id, r["sample_id"]), {})
        if len(king_by_replica) < 2:
            counts["skip:king_rollouts_missing"] += 1
            continue
        glm_side = turns.side(r["turns"], king=False)
        comparisons = []
        for replica, king in sorted(king_by_replica.items()):
            gap = r["glm_score"] - king["score"]
            if abs(gap) < args.margin:
                continue
            if gap > 0 and king.get("amputated"):
                pass  # a rejected King rollout may carry the flag
            if gap < 0 and king.get("amputated"):
                counts["skip:amputated_king_winner"] += 1
                continue
            comparisons.append((abs(gap), replica, king, gap > 0))
        if not comparisons:
            counts["task_tie"] += 1
            continue
        split = "dev" if int(sha(r["sample_id"])[:8], 16) % 1000 < args.dev_share * 1000 else "train"
        prompt = [{"role": m["role"], "content": m["content"], "loss": False} for m in task["context"]]
        rows = []
        for gap, replica, king, glm_wins in sorted(comparisons, key=lambda c: -c[0]):
            king_side = turns.side(king["completion"], king=True)
            chosen, rejected = (glm_side, king_side) if glm_wins else (king_side, glm_side)
            chosen_id, rejected_id = ("glm", f"king#r{replica}") if glm_wins else (f"king#r{replica}", "glm")
            rows.append({
                "schema": SCHEMA,
                "pair_id": sha(f"{r['task_id']}|{chosen_id}|{rejected_id}")[:32],
                "task_id": r["task_id"], "eval_run_id": eval_id, "sample_id": r["sample_id"],
                "split": split, "source": task["source"], "sample_phase": task["sample_phase"],
                "question_mode": task.get("question_mode"), "king_version": 127,
                "task_role": task.get("role", "main"),
                "direction": "glm_over_king" if glm_wins else "king_over_glm",
                "chosen_side": chosen_id, "rejected_side": rejected_id,
                "glm_score": round(r["glm_score"], 6), "king_score": round(king["score"], 6),
                "king_scores_both_rollouts": [round(k["score"], 6) for _, k in sorted(king_by_replica.items())],
                "margin": round(gap, 6),
                "glm_scored_with_tolerant_logprobs": bool(r.get("tolerant_logprobs") and r.get("rejudged")),
                "prompt": prompt, "chosen": chosen, "rejected": rejected,
            })
        sets["single"].append(rows[0])
        sets["double"].extend(rows)
        counts["task_with_pair"] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        folder = args.out / name
        folder.mkdir(exist_ok=True)
        summary = summarize(name, rows, counts, args)
        group_counts: Counter[str] = Counter()
        grouped_by_split: Counter[str] = Counter()
        grouped_pairs: Counter[str] = Counter()
        # released as behaviour-group prefix rows only; trajectory-level pairs are the intermediate
        with (folder / "train.jsonl").open("w", encoding="utf-8") as train, \
                (folder / "dev.jsonl").open("w", encoding="utf-8") as dev:
            for pair in rows:
                produced = grouped_rows(pair, group_counts, first_group_match=args.first_group_match)
                if produced:
                    grouped_pairs[pair["direction"]] += 1
                for row in produced:
                    grouped_by_split[row["split"]] += 1
                    (dev if row["split"] == "dev" else train).write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["grouped"] = {"rows": sum(grouped_by_split.values()), "rows_by_split": dict(grouped_by_split),
                              "pairs": sum(grouped_pairs.values()), "pairs_by_direction": dict(grouped_pairs),
                              "counts": dict(group_counts)}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ("name", "pairs", "tasks", "by_split", "by_direction", "grouped")}, indent=1))


def summarize(name: str, rows: list[dict[str, Any]], counts: Counter, args) -> dict[str, Any]:
    def assistant_turns(side: str, who: str):
        return [t for row in rows for t in row[side]
                if t["role"] == "assistant" and (row[f"{side}_side"] == "glm") == (who == "glm")]

    def reasoning_tokens(turn):
        return turn["tokens"] if not turn["think_closed"] else None

    audit = {}
    for who in ("glm", "king"):
        ts = [t for side in ("chosen", "rejected") for t in assistant_turns(side, who)]
        ts_unique = {id(t): t for t in ts}.values()
        ts = list(ts_unique)
        audit[who] = {
            "assistant_turns": len(ts),
            "think_closed_rate": round(sum(t["think_closed"] for t in ts) / max(1, len(ts)), 4),
            "empty_think_turns": sum(1 for t in ts if t["content"].lstrip().startswith("</think>")),
            "turn_tokens_median": st.median(t["tokens"] for t in ts) if ts else None,
            "over_turn_limit": sum(1 for t in ts if t.get("over_turn_limit")),
            "nudge_echo_turns": sum(1 for t in ts if t.get("nudge_echo")),
            "supervised_turns": sum(1 for t in ts if t["loss"]),
        }
    return {
        "name": name, "schema": SCHEMA, "pairs": len(rows), "tasks": len({r["task_id"] for r in rows}),
        "margin_min": args.margin,
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_direction": dict(Counter(r["direction"] for r in rows)),
        "by_task_role": dict(Counter(r["task_role"] for r in rows)),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_phase": dict(Counter(r["sample_phase"] for r in rows)),
        "margin_median": st.median(abs(r["margin"]) for r in rows) if rows else None,
        "glm_score_mean": round(st.mean(r["glm_score"] for r in rows), 4) if rows else None,
        "king_score_mean": round(st.mean(r["king_score"] for r in rows), 4) if rows else None,
        "tolerant_logprob_scored_pairs": sum(r["glm_scored_with_tolerant_logprobs"] for r in rows),
        "thinking_audit": audit,
        "build_counts": dict(counts),
    }


if __name__ == "__main__":
    main()
