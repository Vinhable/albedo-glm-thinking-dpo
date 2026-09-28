#!/usr/bin/env python3
"""Build provisional balanced best-of-pool SFT data; no API calls or training.

Unknown individual scores stay unknown. Selection uses evidence heuristics, not a
cross-protocol ranking of historical scores. Raw artifacts are never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import build_training_catalog as audit
import filter_teacher_rollouts as teacher_filter
from dataset_creator.extract import parse_trajectory
from extract_teacher_rollouts import _reference_entries

from albedo_eval_service.shared.edit_detection import edited_in_turn
from albedo_eval_service.shared.observation_format import action_blocks, detect_format
from generated_sample_fields import side_output  # noqa: E402

FLAGS = (
    "malformed_bash",
    "loop",
    "broad_search_without_inspect",
    "unsupported_completion_claim",
    "environment_error",
    "output_truncated",
    "too_long",
)
CHAT = teacher_filter.CHAT_BLOCK_RE


def sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        for line, text in enumerate(handle, 1):
            if text.strip():
                yield line, json.loads(text)


def dump(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def messages(value):
    return [{"role": m["role"], "content": m.get("content", "")} for m in value]


def prompt_messages(text):
    result = [{"role": r, "content": c} for r, c in CHAT.findall(text)]
    if len(result) < 2 or [r["role"] for r in result[:2]] != ["system", "user"]:
        raise ValueError("invalid prompt")
    return result


def context_hash(turns):
    return sha([(m["role"], m["content"].strip()) for m in turns])


def split_turns(turns):
    first = next(
        (
            i
            for i, t in enumerate(turns)
            if t.get("role") == "assistant" and t.get("score_target") is True
        ),
        None,
    )
    if first is None:
        raise ValueError("missing explicit score_target cut point")
    return messages(turns[:first]), messages(turns[first:])


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def coordinate(sample_id):
    parsed = audit.parse_sample_id(sample_id)
    if parsed:
        shard, row, cut = parsed
        return f"{shard}:{row}", shard, row, cut
    return sample_id, None, None, None


def phase_from(scoring):
    phases = {(r.get("question_source") or {}).get("sample_phase") for _, r in scoring}
    return next(iter(phases)) if len(phases) == 1 and None not in phases else "unknown"


def score_for(scoring, side):
    # A missing repeat cannot silently shift another repeat's score onto this trajectory.
    if len(scoring) != 1:
        return None
    value = scoring[0][1].get(side + "_score")
    return float(value) if isinstance(value, (float, int)) else None


def evidence(completion, text, error, max_chars):
    result = audit.audit_trajectory(
        completion,
        supplemental_text=text,
        explicit_error=error or "",
        max_trajectory_chars=max_chars,
    )
    assistants = [m["content"] for m in completion if m["role"] == "assistant"]
    edits = [i for i, t in enumerate(assistants) if edited_in_turn(t)]
    tests = [
        i
        for i, t in enumerate(assistants)
        if any(audit.rollout_audit.TEST_RE.search(c) for c in action_blocks(t))
    ]
    result["made_edit"] = bool(edits)
    result["test_command_after_last_edit"] = bool(edits and any(i > max(edits) for i in tests))
    result["targeted_inspect"] = teacher_filter._has_targeted_inspect(completion)
    # Test commands and observations are evidence, not a verified external test execution.
    result["behavior"] = (
        "edit_test"
        if result["test_command_after_last_edit"]
        else "edit_inspect"
        if result["workflow_complete"]
        else "edit_other"
        if edits
        else "grounded_navigation"
        if result["targeted_inspect"]
        else "other"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=120000)
    parser.add_argument("--per-context", type=int, default=3)
    parser.add_argument("--per-task", type=int, default=6)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1 or min(args.per_context, args.per_task) < 1:
        raise SystemExit("invalid selection limits")
    root, dest = args.corpus.resolve(), args.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "summary.json").exists() or (dest / "candidates.jsonl").exists():
        raise SystemExit("Use a fresh output directory; existing builds are not overwritten.")
    latest = json.loads((root / "dashboard-snapshots/latest.json").read_text(encoding="utf-8"))
    current_king = next(m["model_uri"] for m in latest["reign"]["members"] if m.get("is_king"))
    groups, counts, metadata, dedup, seen_runs = Groups(), Counter(), [], {}, set()
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        (dest / "candidates.jsonl").open("wb") as spool,
        (dest / "audit.jsonl").open("w", encoding="utf-8") as audit_out,
    ):
        pending = []

        def drain():
            for future, inputs in pending:
                process_candidate(*inputs, quality=future.result())
            pending.clear()

        def add(raw, prompt, completion, text, error, historical_score):
            inputs = (raw, prompt, completion, text, error, historical_score)
            future = pool.submit(evidence, completion, text, error, args.max_chars)
            pending.append((future, inputs))
            if len(pending) >= 64:
                drain()

        def process_candidate(raw, prompt, completion, text, error, historical_score, quality):
            counts["raw_candidates"] += 1
            reasons = [f for f in FLAGS if quality[f]]
            if not completion or not any(t["role"] == "assistant" for t in completion):
                reasons.append("empty_completion")
            if historical_score is not None and historical_score < 0.65:
                reasons.append("known_score_below_0.65")
            if raw["origin"] == "teacher" and "REFERENCE STEP " not in text:
                reasons.append("unrecognized_reference_format")
            task_coord, shard, row, cut = coordinate(raw["sample_id"])
            task_text = audit._task_text(prompt)
            task_text_hash = sha(" ".join(task_text.split()).lower())
            groups.union("coord:" + task_coord, "text:" + task_text_hash)
            key = sha([context_hash(prompt), completion])
            status = "rejected" if reasons else "duplicate" if key in dedup else "candidate"
            brief = {
                **raw,
                "trajectory_id": key,
                "status": status,
                "reasons": reasons,
                "audit": quality,
                "historical_score": historical_score,
            }
            dump(audit_out, brief)
            counts[status] += 1
            counts.update("reject:" + f for f in reasons)
            if reasons:
                return
            if key in dedup:
                counts["duplicate_provenance_preserved_in_audit"] += 1
                return
            ctx = context_hash(prompt)
            payload = {
                **raw,
                "trajectory_id": key,
                "prompt_messages": prompt,
                "completion_messages": completion,
                "stored_output": text,
                "observation_format": detect_format(raw["sample_id"], prompt),
                "audit": quality,
                "historical_score": historical_score,
                "score_status": "unknown" if historical_score is None else "historical_only",
                "task_coordinate": task_coord,
                "task_text_sha256": task_text_hash,
                "shard": shard,
                "row_index": row,
                "cut_point": cut,
                "source": (shard or raw["sample_id"]).split("/")[0],
                "repo": None,
                "language": None,
                "instance_id": None,
                "license": "unknown-public-artifact-not-a-license",
                "snapshot_hash": None,
                "context_sha256": ctx,
                "status": "provisional_pre_eval",
                "method": "SFT",
            }
            encoded = (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode()
            offset = spool.tell()
            spool.write(encoded)
            rank = (
                quality["test_command_after_last_edit"],
                quality["workflow_complete"],
                quality["targeted_inspect"],
                quality["patch_source_file"],
                -quality["trajectory_chars"],
                key,
            )
            item = {
                "id": key,
                "offset": offset,
                "length": len(encoded),
                "rank": rank,
                "task": "coord:" + task_coord,
                "context": ctx,
                "origin": raw["origin"],
                "phase": raw["sample_phase"],
                "source": payload["source"],
                "behavior": quality["behavior"],
                "eval_ready": raw["eval_ready"],
                "protocol": raw["protocol"],
            }
            metadata.append(item)
            dedup[key] = item

        for mp in sorted((root / "runs").glob("*/*/run-metadata.json")):
            run = json.loads(mp.read_text(encoding="utf-8"))["dashboard_run"]
            run_id = run["eval_run_id"]
            if run_id in seen_runs:
                continue
            seen_runs.add(run_id)
            if args.limit_runs and counts["runs"] >= args.limit_runs:
                break
            gp, sp = mp.parent / "generated-samples.jsonl", mp.parent / "scoring-results.jsonl"
            if not gp.is_file() or not sp.is_file():
                counts["missing_artifact_run"] += 1
                continue
            generated, scoring = defaultdict(list), defaultdict(list)
            for line, row in read_rows(gp):
                generated[str(row["sample_id"]).split("#r")[0]].append((line, row))
            for line, row in read_rows(sp):
                scoring[str(row["sample_id"]).split("#r")[0]].append((line, row))
            for sid, gens in generated.items():
                scores = scoring.get(sid, [])
                try:
                    prompts = [prompt_messages(g["prompt"]) for _, g in gens]
                except (ValueError, KeyError):
                    counts["unparseable_prompt_samples"] += 1
                    continue
                if len({context_hash(p) for p in prompts}) != 1:
                    counts["ambiguous_prefix_samples"] += 1
                    continue
                prompt = prompts[0]
                sources = [(r.get("question_source") or {}) for _, r in scores]
                protocol = (
                    "multi_reference"
                    if any(s.get("reference_trajectories") for s in sources)
                    else "legacy"
                )
                checklist_ok = (
                    bool(scores)
                    and all(r.get("questions") for _, r in scores)
                    and len({sha(r["questions"]) for _, r in scores}) == 1
                )
                king_uri = (run.get("king") or {}).get("model_uri")
                baseline = [
                    {
                        "generated_line": line,
                        "rollout_sample_id": g["sample_id"],
                        "error": g.get("king_error"),
                    }
                    for line, g in gens
                ]
                ready = (
                    checklist_ok
                    and king_uri == current_king
                    and all(
                        side_output(g, "previous_king") and not g.get("king_error") for _, g in gens
                    )
                    and len(gens) == (2 if protocol == "multi_reference" else 1)
                )
                common = {
                    "eval_run_id": run_id,
                    "sample_id": sid,
                    "sample_phase": phase_from(scores),
                    "protocol": protocol,
                    "generated_source": str(gp),
                    "scoring_source": str(sp),
                    "scoring_lines": [line for line, _ in scores],
                    "checklist_sha256": sha(scores[0][1]["questions"]) if checklist_ok else None,
                    "baseline_king_uri": king_uri,
                    "baseline_rollouts": baseline,
                    "eval_ready": ready,
                }
                seen_refs = set()
                for scoring_line, score_row in scores:
                    entries, _ = _reference_entries(score_row.get("question_source") or {})
                    for entry in entries:
                        refkey = sha([entry["trajectory"], score_row.get("questions")])
                        if refkey in seen_refs:
                            continue
                        seen_refs.add(refkey)
                        raw = {
                            **common,
                            "origin": "teacher",
                            "model_uri": entry["model"],
                            "reference_index": entry["reference_index"],
                            "reference_scoring_line": scoring_line,
                            "generated_line": gens[0][0],
                        }
                        add(
                            raw,
                            prompt,
                            parse_trajectory(entry["trajectory"]),
                            entry["trajectory"],
                            "",
                            entry["self_score"],
                        )
                for generated_line, gen in gens:
                    for origin, prefix in (("king", "previous_king"), ("challenger", "challenger")):
                        try:
                            prefix_msgs, completion = split_turns(gen.get(prefix + "_turns") or [])
                        except ValueError:
                            counts["missing_target_boundary"] += 1
                            continue
                        if context_hash(prefix_msgs) != context_hash(prompt):
                            counts["prefix_turn_mismatch"] += 1
                            continue
                        model_uri = king_uri if origin == "king" else run.get("model_uri")
                        raw = {
                            **common,
                            "origin": origin,
                            "model_uri": model_uri,
                            "generated_line": generated_line,
                            "rollout_sample_id": gen["sample_id"],
                            "eval_ready": ready and model_uri != current_king,
                        }
                        add(
                            raw,
                            prefix_msgs,
                            completion,
                            gen.get(prefix + "_output") or "",
                            gen.get("king_error" if origin == "king" else "chal_error"),
                            score_for(scores, origin) if len(gens) == 1 else None,
                        )
            drain()
            counts["runs"] += 1
            if counts["runs"] % 10 == 0:
                print(json.dumps(dict(counts)), flush=True)

    # Split connected task groups before selection; all cut points/references remain together.
    by_task = defaultdict(list)
    for m in metadata:
        m["task_group"] = groups.find(m["task"])
        by_task[m["task_group"]].append(m)
    selected = []
    for group, items in sorted(by_task.items()):
        contexts, origins = Counter(), Counter()
        remaining = sorted(items, key=lambda m: m["rank"], reverse=True)
        for _ in range(args.per_task):
            eligible = [m for m in remaining if contexts[m["context"]] < args.per_context]
            if not eligible:
                break
            # Round-robin origins where available; no forced corpus-wide teacher percentage.
            pick = max(eligible, key=lambda m: (-origins[m["origin"]], m["rank"]))
            remaining.remove(pick)
            contexts[pick["context"]] += 1
            origins[pick["origin"]] += 1
            pick["split"] = (
                "dev"
                if int(sha(["sft-c-v1", group])[:8], 16) / 2**32 < args.dev_fraction
                else "train"
            )
            selected.append(pick)
    strata = Counter((m["split"], m["source"], m["phase"], m["behavior"]) for m in selected)
    weights = {key: 1 / math.sqrt(n) for key, n in strata.items()}
    split_counts = Counter(m["split"] for m in selected)
    norm = {
        s: sum(
            weights[(m["split"], m["source"], m["phase"], m["behavior"])]
            for m in selected
            if m["split"] == s
        )
        / n
        for s, n in split_counts.items()
    }
    output_counts = Counter()
    with (
        (dest / "candidates.jsonl").open("rb") as spool,
        (dest / "sft-c-train.jsonl").open("w", encoding="utf-8") as train,
        (dest / "sft-c-dev.jsonl").open("w", encoding="utf-8") as dev,
        (dest / "eval-c-frozen.jsonl").open("w", encoding="utf-8") as eval_out,
    ):
        for m in sorted(selected, key=lambda v: v["id"]):
            spool.seek(m["offset"])
            row = json.loads(spool.read(m["length"]))
            weight = (
                weights[(m["split"], m["source"], m["phase"], m["behavior"])] / norm[m["split"]]
            )
            row.update(
                split=m["split"],
                task_group=m["task_group"],
                sampling_weight=max(0.25, min(4.0, weight)),
            )
            dump(train if m["split"] == "train" else dev, row)
            for key in (
                "selected",
                "split:" + m["split"],
                "origin:" + m["origin"],
                "phase:" + m["phase"],
                "behavior:" + m["behavior"],
                "protocol:" + m["protocol"],
            ):
                output_counts[key] += 1
            if row["eval_ready"]:
                dump(
                    eval_out,
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("prompt_messages", "completion_messages", "stored_output")
                    },
                )
                output_counts["eval_ready"] += 1
                output_counts["eval_protocol:" + m["protocol"]] += 1
    summary = {
        "counts": dict(counts),
        "output_counts": dict(output_counts),
        "task_groups": len(by_task),
        "current_king": current_king,
        "corpus": str(root),
        "policy": vars(args) | {"corpus": str(root), "output": str(dest)},
        "status": "PROVISIONAL: quality heuristics passed, not live-evaluated",
        "balancing": (
            "max 3/context, 6/task; origin diversity; "
            "inverse-sqrt source/phase/behavior weights clipped 0.25..4"
        ),
        "limitations": [
            "Unknown repo/language/license/snapshot metadata remains null/unknown",
            "Observed scores are not comparable across teacher self-score and candidate score",
            "Weights must be consumed by a sampler; writing JSONL does not apply them",
            "Test/patch heuristics do not certify real repository correctness",
            "Near-duplicate paraphrases are capped per task, not semantically certified",
            "Frozen-checklist evaluation has reference and selection bias",
        ],
    }
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
