#!/usr/bin/env python3
"""Export a `duel_checkpoints_vs_king.py` run as one complete round in the able_e6_400_rollouts layout.

    py -3 scripts/export_duel_rounds.py --duel E:/albedo-storage-temp/duel-sft-dpo-vs-king-20260928 \
        --dest C:/Users/VINH/Desktop/able_e6_400_rollouts/sft-dpo-step70-vs-cxxvii-20260929

Layout (every JSONL has one row per task, in the same task order):
    challenger/<label>-step70-r1.jsonl, -r2.jsonl   per policy label
    king/cxxvii-king-r1.jsonl, -r2.jsonl            production King rollouts, rejudged locally
    duel-manifest.jsonl                             all trajectories of a task with their scores
    evaluation/                                     summary, per-trajectory scores, unscored readings
    duel-summary.json

King turns come from the crawled production artifacts (the duel input keeps only the judged document).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from albedo_eval_service.remote.generation import format_scored_trajectory  # noqa: E402
from build_gen_vinhable import rollouts, run_index  # noqa: E402
from duel_checkpoints_vs_king import first_sample  # noqa: E402
from pilot_glm_thinking_teacher import read_jsonl  # noqa: E402

SCHEMA = "sft-dpo-step70-vs-cxxvii-king-v1"
PROTOCOL = "cxxvii_graded20_local_rejudge_glm52_one_reading"
KING_MODEL = "dendriteholdings/albedo-qwen3.6-35b-king-CXXVII@e920362b460ae6b2a33c9cb298aa7f14a38d5584"
CHECKPOINTS = {
    "sft": "single-lab-sft-cont/snapshots/step0070-ep1.0 (SFT continued from step 50, not uploaded)",
    "dpo": "single-lab-v3/snapshots/step0070-ep1.0 (DPO + NLL 50); HF private "
           "vinhable/albedo-hessian-backup-20260921 checkpoints/single-lab-v3-step0070-ep1.0-delta-vs-king127",
}
RESULT_BAND = 0.05  # win/tie/loss band, as in the duel summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluation(r: dict, score_key: str) -> dict:
    return {"protocol": PROTOCOL, "score_key": score_key, "score": r.get("score"),
            "scored": r.get("score") is not None, "provider": r.get("provider"), "parse_ok": r.get("parse_ok"),
            "judge_retries": r.get("judge_retries"), "judge_error": r.get("judge_error"),
            "zero_reason": r.get("zero_reason"), "amputated_thinking": r.get("amputated")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duel", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--crawl-root", type=Path, default=Path("E:/albedo-storage-temp/raw-rollouts-graded-20260927"))
    args = parser.parse_args()

    tasks = read_jsonl(args.duel / "duel-input.jsonl")
    generated = {(r["label"], r["task_id"], r["replica"]): r for r in read_jsonl(args.duel / "generated.jsonl")
                 if r.get("status") == "generated"}
    results = {(r["label"], r["task_id"], r["replica"]): r for r in read_jsonl(args.duel / "results.jsonl")
               if r.get("status") == "ok"}
    labels = sorted({label for label, _, _ in generated})

    # production King turns, from the crawled run each task came from
    runs = {run["eval_run_id"]: run for run in run_index([args.crawl_root])}
    king_turns: dict[tuple[str, int], list[dict]] = {}
    counts: Counter[str] = Counter()
    for eval_run_id in sorted({t["eval_run_id"] for t in tasks}):
        wanted = {t["sample_id"] for t in tasks if t["eval_run_id"] == eval_run_id}
        for sid, _, found in rollouts(runs[eval_run_id], counts):
            if sid in wanted:
                for r in found:
                    if r["side"] == "king":
                        king_turns[(f"{eval_run_id}:{sid}", r["replica"])] = (
                            [dict(m) for m in r["context"]] + [dict(t) for t in r["completion"]])

    shards: dict[str, list[dict]] = {}
    manifest, scores, unscored = [], [], []
    for position, task in enumerate(tasks, 1):
        tid = task["task_id"]
        duel_id = hashlib.sha256(tid.encode()).hexdigest()
        checklist_sha = hashlib.sha256(json.dumps(task["questions"], sort_keys=True).encode()).hexdigest()
        common = {"duel_schema": SCHEMA, "duel_id": duel_id, "sample_id": task["sample_id"],
                  "task_coordinate": task["sample_id"].rsplit(":", 1)[0], "source": task["source"],
                  "sample_phase": task["sample_phase"], "horizon": task["horizon"],
                  "production_eval_run_id": task["eval_run_id"]}
        files, trajectory_scores = {}, {}
        for label in labels:
            for replica in (1, 2):
                g, r = generated[(label, tid, replica)], results.get((label, tid, replica), {})
                file = f"challenger/{label}-step70-r{replica}.jsonl"
                key = f"{task['sample_id']}|{label}|r{replica}"
                shards.setdefault(file, []).append({
                    "schema_version": 1, **common, "duel_side": "challenger", "participant": f"{label}_step70",
                    "checkpoint": CHECKPOINTS[label], "king_model": KING_MODEL, "replica": f"r{replica}",
                    "eval_run_id": f"duel-{label}:{task['eval_run_id']}",
                    "prompt": first_sample(task, replica).prompt,
                    "submit_marker": task["submit_marker"], "submit_command": task["submit_command"],
                    "candidate_output": g["document"], "candidate_turns": g["turns"], "candidate_error": None,
                    "truncated": g["truncated"], "abandoned": g["abandoned"],
                    "retry_feedbacks": g["retry_feedbacks"], "assistant_turns": g["assistant_turns"],
                    "questions": task["questions"], "checklist_sha256": checklist_sha,
                    "trajectory_score": r.get("score"), "score_provider": r.get("provider"),
                    "evaluation": evaluation(r, key)})
                files[f"{label}_r{replica}"] = file
                trajectory_scores[f"{label}_r{replica}"] = r.get("score")
                scores.append({"score_key": key, "duel_id": duel_id, **{k: r.get(k) for k in (
                    "label", "replica", "score", "provider", "parse_ok", "judge_error", "zero_reason")}})
                if r.get("score") is None:
                    unscored.append(scores[-1])
        for rollout in task["king_rollouts"]:
            replica = rollout["replica"]
            r = results.get(("king", tid, replica), {})
            turns = king_turns.get((tid, replica))
            if turns is None:
                raise SystemExit(f"King turns missing for {tid} r{replica}")
            rebuilt = format_scored_trajectory(
                [dict(m) for m in task["context"]]
                + [{**t, "score_target": True} if t["role"] == "assistant" else {**t, "environment_observation": True}
                   for t in turns[len(task["context"]):]])
            if rebuilt != rollout["document"]:
                raise SystemExit(f"crawled King turns do not reproduce the judged document: {tid} r{replica}")
            file = f"king/cxxvii-king-r{replica}.jsonl"
            key = f"{task['sample_id']}|king|r{replica}"
            shards.setdefault(file, []).append({
                "schema_version": 1, **common, "duel_side": "king", "participant": "cxxvii_king",
                "king_model": KING_MODEL, "replica": f"r{replica}", "eval_run_id": task["eval_run_id"],
                "submit_marker": task["submit_marker"], "submit_command": task["submit_command"],
                "king_output": rollout["document"], "king_turns": turns,
                "assistant_turns": rollout["assistant_turns"], "questions": task["questions"],
                "checklist_sha256": checklist_sha, "trajectory_score": r.get("score"),
                "production_score": rollout["score"], "score_provider": r.get("provider"),
                "evaluation": {**evaluation(r, key), "production_score": rollout["score"],
                               "production_amputated": rollout["amputated"]}})
            files[f"king_r{replica}"] = file
            trajectory_scores[f"king_r{replica}"] = r.get("score")
            scores.append({"score_key": key, "duel_id": duel_id, "label": "king", "replica": replica,
                           "score": r.get("score"), "production_score": rollout["score"],
                           **{k: r.get(k) for k in ("provider", "parse_ok", "judge_error", "zero_reason")}})
            if r.get("score") is None:
                unscored.append(scores[-1])

        # the duel summary's rules: a policy mean needs both replicas scored, the King mean any scored one
        king_scored = [trajectory_scores[f"king_r{i}"] for i in (1, 2) if trajectory_scores.get(f"king_r{i}") is not None]
        king_mean = st.mean(king_scored) if king_scored else None
        row = {"round": 1, "round_position": position, "duel_id": duel_id, "task_id": tid,
               "sample_id": task["sample_id"], "task_coordinate": common["task_coordinate"],
               "production_eval_run_id": task["eval_run_id"], "king_model": KING_MODEL, "files": files,
               "evaluation": {"trajectory_scores": trajectory_scores, "king_mean": king_mean,
                              "king_production_mean": task["king_mean"], "source": task["source"],
                              "sample_phase": task["sample_phase"], "horizon": task["horizon"],
                              "result_band": RESULT_BAND}}
        for label in labels:
            pair = [trajectory_scores[f"{label}_r{i}"] for i in (1, 2)]
            mean = st.mean(pair) if None not in pair else None
            delta = mean - king_mean if mean is not None and king_mean is not None else None
            result = None if delta is None else ("win" if delta > RESULT_BAND else "loss" if delta < -RESULT_BAND else "tie")
            row["evaluation"].update({f"{label}_mean": mean, f"{label}_delta": delta, f"{label}_result": result})
            row.update({f"{label}_r1_score": pair[0], f"{label}_r2_score": pair[1], f"{label}_mean_score": mean,
                        f"{label}_score_delta": delta, f"{label}_duel_result": result})
        row.update({"king_r1_score": trajectory_scores.get("king_r1"), "king_r2_score": trajectory_scores.get("king_r2"),
                    "king_mean_score": king_mean})
        manifest.append(row)

    dest = args.dest
    for file, rows in shards.items():
        assert len(rows) == len(tasks), file
        write_jsonl(dest / file, rows)
    write_jsonl(dest / "duel-manifest.jsonl", manifest)
    write_jsonl(dest / "evaluation" / "trajectory-scores.jsonl", scores)
    write_jsonl(dest / "evaluation" / "unscored-readings.jsonl", unscored)
    shutil.copy2(args.duel / "summary.json", dest / "evaluation" / "summary.json")
    shutil.copy2(args.duel / "select-summary.json", dest / "evaluation" / "select-summary.json")
    summary = json.loads((args.duel / "summary.json").read_text(encoding="utf-8"))
    duel_summary = {
        "schema": SCHEMA, "tasks": len(tasks), "trajectories": sum(len(r) for r in shards.values()),
        "files": sorted(shards), "king_model": KING_MODEL, "checkpoints": CHECKPOINTS,
        "manifest_results": {label: dict(Counter(m[f"{label}_duel_result"] for m in manifest)) for label in labels},
        "headline": {k: summary[k] for k in ["king", *labels]},
        "unscored_readings": len(unscored),
    }
    (dest / "duel-summary.json").write_text(json.dumps(duel_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dest": str(dest), "files": {f: len(r) for f, r in shards.items()},
                      "manifest": len(manifest), "unscored": len(unscored),
                      "manifest_results": duel_summary["manifest_results"]}, indent=1))


if __name__ == "__main__":
    main()
