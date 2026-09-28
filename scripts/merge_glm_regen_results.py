#!/usr/bin/env python3
"""Merge a GLM regeneration batch with its rerun and rejudge outputs into one row per task.

Later sources win: batch `results.jsonl` < its `rejudge.jsonl` (scored rows) < rerun
`results.jsonl` < the rerun's `rejudge.jsonl`. Writes `merged.jsonl` in the batch folder and prints
the pair counts (GLM vs each King rollout, margin >= --margin).

    py -3 scripts/merge_glm_regen_results.py --batch E:/albedo-storage-temp/glm-thinking-batch300-20260927 \
        --rerun E:/albedo-storage-temp/glm-thinking-batch300-rerun-trunc-20260927
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def overlay_rejudge(merged: dict[str, dict], folder: Path) -> None:
    for r in rows(folder / "rejudge.jsonl"):
        if r["glm_score"] is not None and r["task_id"] in merged:
            merged[r["task_id"]] = {**merged[r["task_id"]], "glm_score": r["glm_score"],
                                    "glm_amputated": r["glm_amputated"], "rejudged": True,
                                    "tolerant_logprobs": r.get("tolerant_logprobs", False)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, nargs="*", default=[])
    parser.add_argument("--margin", type=float, default=0.15)
    args = parser.parse_args()

    roles = {r["task_id"]: r.get("role", "main") for r in rows(args.batch / "pilot-input.jsonl")}
    merged = {r["task_id"]: r for r in rows(args.batch / "results.jsonl") if r.get("status") == "ok"}
    overlay_rejudge(merged, args.batch)
    for folder in args.rerun:
        for r in rows(folder / "results.jsonl"):
            if r.get("status") == "ok":
                merged[r["task_id"]] = {**r, "rerun": True}
        overlay_rejudge(merged, folder)

    usable = [r for r in merged.values()
              if r["glm_score"] is not None and r["stop"] != "truncated" and not r.get("glm_amputated")]
    with (args.batch / "merged.jsonl").open("w", encoding="utf-8") as handle:
        for r in merged.values():
            handle.write(json.dumps({**r, "role": roles.get(r["task_id"], "main")}, ensure_ascii=False) + "\n")

    m = args.margin
    glm_tasks = [r for r in usable if any(r["glm_score"] >= k + m for k in r["king_scores"])]
    king_tasks = [r for r in usable if any(k >= r["glm_score"] + m for k in r["king_scores"])]
    report = {
        "tasks": len(merged),
        "unscored": sum(1 for r in merged.values() if r["glm_score"] is None),
        "truncated": sum(1 for r in merged.values() if r["stop"] == "truncated"),
        "usable": len(usable),
        "glm_mean_usable_main": round(st.mean(r["glm_score"] for r in usable if roles[r["task_id"]] == "main"), 3),
        "king_mean_usable_main": round(st.mean(r["king_mean"] for r in usable if roles[r["task_id"]] == "main"), 3),
        "tasks_glm_beats_king": len(glm_tasks),
        "pairs_glm_beats_king_both_rollouts": sum(sum(r["glm_score"] >= k + m for k in r["king_scores"]) for r in glm_tasks),
        "tasks_king_beats_glm": len(king_tasks),
        "king_beats_glm_by_role": {role: sum(roles[r["task_id"]] == role for r in king_tasks) for role in ("main", "contrast")},
        "tasks_with_a_pair": len({r["task_id"] for r in glm_tasks + king_tasks}),
        "ties": len(usable) - len({r["task_id"] for r in glm_tasks + king_tasks}),
        "scored_by_tolerant_reading": sum(1 for r in merged.values() if r.get("tolerant_logprobs") and r.get("rejudged")),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
