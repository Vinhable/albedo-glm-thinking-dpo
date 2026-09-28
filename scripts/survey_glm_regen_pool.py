#!/usr/bin/env python3
"""How many crawled graded-era tasks could feed GLM-with-thinking regeneration, and how many pairs.

Reads every crawled graded run (King 127) and the GLM reference scores from HF
`dendriteholdings/albedo`, and counts tasks per King band, with pair-yield estimates taken from
pilot v5 (GLM beat the King by >= 0.15 on 8/15 weak-King tasks). A task has two scored King
rollouts, so it can give up to two pairs (GLM vs each King rollout it beats by the margin).

    py -3 scripts/survey_glm_regen_pool.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from build_gen_vinhable import rollouts, run_index  # noqa: E402
from pilot_glm_thinking_teacher import reference_scores  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crawl-root", type=Path, default=Path("E:/albedo-storage-temp/raw-rollouts-graded-20260927"))
    parser.add_argument("--hf-root", type=Path, default=Path("E:/albedo-storage-temp/hf-albedo-reference-20260927"))
    parser.add_argument("--held-out", nargs="*", default=["719dbe80", "3a48801e"])
    parser.add_argument("--out", type=Path, default=Path("E:/albedo-storage-temp/glm-regen-pool-survey.json"))
    args = parser.parse_args()

    glm = reference_scores(args.hf_root)
    counts: Counter[str] = Counter()
    tasks = []
    runs = run_index([args.crawl_root])
    for run in runs:
        if run["eval_run_id"].startswith(tuple(args.held_out)):
            counts["run_held_out"] += 1
            continue
        for sid, info, found in rollouts(run, counts):
            counts["samples"] += 1
            if info.get("scoring_mode") != "graded_20":
                counts["skip_not_graded"] += 1
                continue
            kings = sorted(r["score"] for r in found if r["side"] == "king")
            if len(kings) < 2:
                counts["skip_king_rollouts_lt2"] += 1
                continue
            ref = glm.get((run["eval_run_id"], sid), [])
            tasks.append({"eval": run["eval_run_id"], "sample_id": sid, "phase": info.get("sample_phase"),
                          "source": sid.split("/", 1)[0], "king": kings, "king_mean": st.mean(kings),
                          "glm_ref": ref, "finished_at": run["finished_at"]})
    unique = {t["sample_id"] for t in tasks}
    finished = sorted(t["finished_at"] for t in tasks if t["finished_at"])

    def band(t):
        k = t["king_mean"]
        return "<0.5" if k < 0.5 else "0.5-0.7" if k < 0.7 else "0.7-0.85" if k < 0.85 else ">=0.85"

    by_band = Counter(band(t) for t in tasks)
    glm_high = Counter(band(t) for t in tasks if t["glm_ref"] and max(t["glm_ref"]) >= 0.9)
    # pair potential if a regenerated GLM rollout scored like the reference's best run
    pot = Counter()
    for t in tasks:
        if not t["glm_ref"]:
            continue
        g = max(t["glm_ref"])
        pot["glm_beats_both_king_by_0.15"] += all(g >= k + 0.15 for k in t["king"])
        pot["glm_beats_one_king_by_0.15"] += sum(g >= k + 0.15 for k in t["king"]) == 1
        pot["king_best_beats_glm_by_0.15"] += max(t["king"]) >= g + 0.15
    report = {
        "runs_used": len(runs) - counts["run_held_out"],
        "graded_tasks_with_two_king_rollouts": len(tasks),
        "unique_sample_ids": len(unique),
        "finished_range": [finished[0], finished[-1]] if finished else None,
        "king_mean_band": dict(by_band),
        "king_mean_band_with_glm_ref_best_ge_0.9": dict(glm_high),
        "pair_potential_if_glm_matches_its_reference": dict(pot),
        "counts": dict(counts),
        "by_source": dict(Counter(t["source"] for t in tasks)),
        "by_phase": dict(Counter(t["phase"] for t in tasks)),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
