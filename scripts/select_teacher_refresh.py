#!/usr/bin/env python3
"""Select one provisional teacher per exact context, protecting reserved tasks.

No judge/model calls. Selection is heuristic, never a correctness label.
Source artifacts and existing C2 are immutable; write into a fresh output directory.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import build_sft_c_dataset as base
import crawl_rollouts as crawler
import review_c3_candidates as scanner
from dataset_creator.extract import parse_trajectory
from extract_teacher_rollouts import _reference_entries
from generated_sample_fields import side_output  # noqa: E402


def dump(f, row):
    f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def identities(sid, prompt):
    coord = base.coordinate(sid.split("#r")[0])[0]
    task = " ".join(base.audit._task_text(prompt).split()).lower()
    if not task:
        raise ValueError("Missing task text")
    return ["coord:" + coord, "task:" + base.sha(task), "context:" + base.context_hash(prompt)]


def connect(groups, keys):
    for key in keys[1:]:
        groups.union(keys[0], key)


def grouped_run(path):
    generated, scores = defaultdict(list), defaultdict(list)
    for line, row in base.read_rows(path / "generated-samples.jsonl"):
        generated[str(row["sample_id"]).split("#r")[0]].append((line, row))
    for line, row in base.read_rows(path / "scoring-results.jsonl"):
        scores[str(row["sample_id"]).split("#r")[0]].append((line, row))
    for sid, gens in generated.items():
        prompts = [base.prompt_messages(g["prompt"]) for _, g in gens]
        # Fail closed: even r1/r2 must have the same exact full prefix.
        if len({base.sha(p) for p in prompts}) != 1:
            yield sid, None, gens, scores.get(sid, [])
        else:
            yield sid, prompts[0], gens, scores.get(sid, [])


def numeric_score(value):
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
        else None
    )


def candidate(prompt, entry, common):
    text = entry["trajectory"]
    if "REFERENCE STEP " not in text:
        raise ValueError("Unrecognized reference format")
    completion = parse_trajectory(text)
    if not completion or completion[0]["role"] != "assistant":
        raise ValueError("Missing assistant continuation")
    messages = [
        {**m, "loss": i >= len(prompt) and m["role"] == "assistant"}
        for i, m in enumerate(prompt + completion)
    ]
    tid = base.sha([prompt, completion])
    row = {
        **common,
        "trajectory_id": tid,
        "messages": messages,
        "completion_start": len(prompt),
        "split": "candidate",
        "origin": "teacher",
        "model_uri": entry["model"],
        "reference_index": entry["reference_index"],
        "reference_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "reference_self_score": numeric_score(entry["self_score"]),
        "observation_format": base.detect_format(common["sample_id"], prompt),
        "status": "provisional_teacher_refresh_not_C3",
        "license": "unknown-public-artifact-not-a-license",
    }
    diag = scanner.analyze_row(row)
    quality = base.evidence(completion, text, "", 10**9)
    # Environment errors remain review-only; no edit/test requirement for no-op tasks.
    rejects = [f["flag"] for f in diag["invalid_reasons"] + diag["policy_flags"]]
    rejects += [k for k in ("malformed_bash", "output_truncated") if quality[k]]
    flags = Counter(f["flag"] for f in diag["risk_flags"])
    risk = sum(
        flags[k]
        for k in (
            "stagnation_retry",
            "near_loop",
            "environment_detour",
            "observation_command_mismatch",
        )
    )
    # Simple comparison signals, not successful test/edit labels.
    rank = [
        -risk,
        -int(quality["broad_search_without_inspect"]),
        -int(quality["unsupported_completion_claim"]),
        int(quality["targeted_inspect"]),
    ]
    row["selection_signals"] = {
        "rank_without_self_score": rank,
        "risk_flags": dict(flags),
        "scanner_disposition": diag["suggested_disposition"],
        "targeted_inspect": quality["targeted_inspect"],
        "historical_self_score_only": True,
        "correctness": "unknown",
    }
    return row, diag, rejects, rank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--c2", type=Path, required=True)
    p.add_argument("--used-eval", type=Path, action="append", default=[])
    a = p.parse_args()
    plan = json.loads((a.report / "crawl-plan.json").read_text(encoding="utf-8"))
    crawl = json.loads((a.report / "crawl-summary.json").read_text(encoding="utf-8"))
    if crawl["failures"] or crawl["runs"] != len(plan["holdout_runs"]) + len(
        plan["new_training_runs"]
    ):
        raise ValueError("Crawl is incomplete")
    a.output.mkdir(parents=True, exist_ok=False)
    root = Path(plan["output"])
    groups = base.Groups()
    counts = Counter()
    provenance = {
        "crawl_plan_sha256": crawler._sha256(a.report / "crawl-plan.json"),
        "scanner_sha256": crawler._sha256(Path(scanner.__file__)),
        "selector_sha256": crawler._sha256(Path(__file__)),
        "c2_sha256": {},
        "used_eval_sha256": {},
    }
    crawler._write_json_atomic(
        a.output / "selection-policy.json",
        {
            "version": "teacher-refresh-best-one-v1",
            "group": "exact full prefix/context, dedup r1/r2 reference copies",
            "hard_exclusions": ["schema", "canonical_loop", "malformed_bash", "output_truncated"],
            "rank": [
                "fewer stagnation/near-loop/detour/mismatch cues",
                "no broad search without inspect",
                "no unsupported completion claim",
                "targeted inspection",
                "aligned historical self score only when all alternatives have scores",
                "stable trajectory hash tie break",
            ],
            "unknowns": "No new judge scores; quality flags and self scores are not correctness.",
            "holdout": "All reserved task/context connected groups excluded before ranking",
            "no_train_export": "Candidate pool only; token lengths and C3 recipes still pending",
        },
    )
    c2_keys, dev_keys, prior_keys = set(), set(), set()
    for split in ("train", "dev"):
        path = a.c2 / f"{split}.message-loss.jsonl"
        provenance["c2_sha256"][str(path)] = crawler._sha256(path)
        for _, row in base.read_rows(path):
            keys = identities(row["sample_id"], row["messages"][: row["completion_start"]])
            connect(groups, keys)
            c2_keys.add(keys[0])
            if split == "dev":
                dev_keys.add(keys[0])
    for path in a.used_eval:
        provenance["used_eval_sha256"][str(path)] = crawler._sha256(path)
        for _, row in base.read_rows(path):
            prompt = (
                base.prompt_messages(row["prompt"])
                if isinstance(row.get("prompt"), str)
                else row.get("messages")
            )
            if not prompt:
                raise ValueError(f"Missing prefix in prior eval {path}")
            keys = identities(row["sample_id"], prompt)
            connect(groups, keys)
            prior_keys.add(keys[0])
    # Identity-only pass through all refresh prefixes closes transitive aliases before split.
    held = []
    held_keys = set()
    for lane, key in [("holdout", "holdout_runs"), ("teacher", "new_training_runs")]:
        for run in plan[key]:
            directory = root / "runs" / lane / run["eval_run_id"]
            for sid, prompt, gens, scores in grouped_run(directory):
                if prompt is None:
                    counts[lane + "_ambiguous_prefix"] += 1
                    # Reserve every conflicting prefix too; never let it into training.
                    for _, g in gens:
                        keys = identities(sid, base.prompt_messages(g["prompt"]))
                        connect(groups, keys)
                        if lane == "holdout":
                            held_keys.add(keys[0])
                    continue
                keys = identities(sid, prompt)
                connect(groups, keys)
                if lane != "holdout":
                    continue
                held_keys.add(keys[0])
                questions = {base.sha(s.get("questions")) for _, s in scores if s.get("questions")}
                ready = (
                    len(gens) == 2
                    and len({g["sample_id"] for _, g in gens}) == 2
                    and all(
                        side_output(g, "previous_king") and not g.get("king_error") for _, g in gens
                    )
                    and bool(scores)
                    and all(s.get("questions") for _, s in scores)
                    and len(questions) == 1
                )
                held.append(
                    {
                        "eval_run_id": run["eval_run_id"],
                        "sample_id": sid,
                        "task_key": keys[0],
                        "context_sha256": base.sha(prompt),
                        "king_model_uri": (run.get("king") or {}).get("model_uri"),
                        "generated_source": str(directory / "generated-samples.jsonl"),
                        "scoring_source": str(directory / "scoring-results.jsonl"),
                        "generated_lines": [n for n, _ in gens],
                        "scoring_lines": [n for n, _ in scores],
                        "questions_sha256": next(iter(questions)) if len(questions) == 1 else None,
                        "paired_two_rollouts_available": ready,
                    }
                )
    held_roots = {groups.find(k) for k in held_keys}
    c2_roots = {groups.find(k) for k in c2_keys}
    dev_roots = {groups.find(k) for k in dev_keys}
    prior_roots = {groups.find(k) for k in prior_keys}
    with (a.output / "holdout-inventory.jsonl").open("w", encoding="utf-8") as f:
        clean = set()
        for row in held:
            group = groups.find(row["task_key"])
            row.update(
                {
                    "task_group": group,
                    "overlap_c2": group in c2_roots,
                    "overlap_prior_eval": group in prior_roots,
                }
            )
            row["eligible_future_eval"] = (
                row["paired_two_rollouts_available"] and group not in c2_roots | prior_roots
            )
            counts["holdout_samples"] += 1
            counts["holdout_overlap_c2"] += row["overlap_c2"]
            counts["holdout_overlap_prior_eval"] += row["overlap_prior_eval"]
            counts["holdout_pairable"] += row["paired_two_rollouts_available"]
            counts["holdout_clean_pairable"] += row["eligible_future_eval"]
            if row["eligible_future_eval"]:
                clean.add(group)
            dump(f, row)
        counts["holdout_clean_unique_tasks"] = len(clean)
    crawler._write_json_atomic(
        a.output / "excluded-task-groups.json",
        {
            "holdout": sorted(held_roots),
            "c2_dev": sorted(dev_roots),
            "prior_eval": sorted(prior_roots),
            "alias_to_group": {k: groups.find(k) for k in groups.parent},
            "rule": "Future C3 must exclude all reserved connected groups from all sources",
        },
    )
    best = {}
    spool_path = a.output / "per-run-selected.jsonl"
    with (
        spool_path.open("wb") as spool,
        (a.output / "selection-audit.jsonl").open("w", encoding="utf-8") as audit,
    ):
        for run in plan["new_training_runs"]:
            directory = root / "runs" / "teacher" / run["eval_run_id"]
            for sid, prompt, gens, scores in grouped_run(directory):
                counts["training_samples"] += 1
                if prompt is None:
                    counts["skip_ambiguous_prefix"] += 1
                    continue
                keys = identities(sid, prompt)
                group = groups.find(keys[0])
                blocked = (
                    "reserved_holdout"
                    if group in held_roots
                    else "c2_dev"
                    if group in dev_roots
                    else "prior_eval"
                    if group in prior_roots
                    else None
                )
                if blocked:
                    counts["excluded_" + blocked] += 1
                    dump(
                        audit,
                        {
                            "eval_run_id": run["eval_run_id"],
                            "sample_id": sid,
                            "excluded_before_selection": blocked,
                            "task_group": group,
                        },
                    )
                    continue
                # Distinct reference/checklist sets are ambiguous; never merge loose passes.
                sets = set()
                options = {}
                for line, s in scores:
                    entries, _ = _reference_entries(s.get("question_source") or {})
                    if entries:
                        sets.add(base.sha([[e["trajectory"] for e in entries], s.get("questions")]))
                    for entry in entries:
                        options.setdefault(base.sha(entry["trajectory"]), (line, s, entry))
                if len(sets) != 1:
                    counts["skip_missing_or_ambiguous_reference_set"] += 1
                    continue
                candidates = []
                for line, s, entry in options.values():
                    counts["raw_distinct_references"] += 1
                    common = {
                        "eval_run_id": run["eval_run_id"],
                        "sample_id": sid,
                        "task_group": group,
                        "task_coordinate": keys[0][6:],
                        "source": sid.split("/")[0],
                        "sample_phase": base.phase_from(scores),
                        "context_sha256": base.sha(prompt),
                        "reference_scoring_line": line,
                        "generated_source": str(directory / "generated-samples.jsonl"),
                        "generated_line": gens[0][0],
                        "scoring_source": str(directory / "scoring-results.jsonl"),
                        "questions_sha256": base.sha(s.get("questions")),
                    }
                    try:
                        row, diag, rejects, rank = candidate(prompt, entry, common)
                    except ValueError as exc:
                        counts["reference_parse_failures"] += 1
                        dump(
                            audit,
                            {
                                **common,
                                "reference_index": entry["reference_index"],
                                "rejected": str(exc),
                            },
                        )
                        continue
                    record = {
                        **common,
                        "trajectory_id": row["trajectory_id"],
                        "reference_index": entry["reference_index"],
                        "rejected": rejects,
                        "signals": row["selection_signals"],
                        "diagnostics": diag,
                        "reference_self_score": row["reference_self_score"],
                    }
                    candidates.append((row, rejects, rank, record))
                valid = [c for c in candidates if not c[1]]
                all_scored = bool(valid) and all(
                    c[0]["reference_self_score"] is not None for c in valid
                )

                def rank_key(c):
                    return (
                        *c[2],
                        c[0]["reference_self_score"] if all_scored else -1,
                        c[0]["trajectory_id"],
                    )

                chosen = max(valid, key=rank_key) if valid else None
                for c in candidates:
                    c[3]["selected_in_run_context"] = c is chosen
                    c[3]["self_score_used"] = all_scored
                    dump(audit, c[3])
                if chosen is None:
                    counts["no_usable_reference"] += 1
                    continue
                row = chosen[0]
                row["selection_signals"]["self_score_used"] = all_scored
                row["alternatives_in_run_context"] = len(candidates)
                payload = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode()
                offset = spool.tell()
                spool.write(payload)
                # No cross-run historical score comparison. Use quality cues then hash.
                cross_rank = (*chosen[2], row["trajectory_id"])
                context = row["context_sha256"]
                if context not in best or cross_rank > best[context]["rank"]:
                    best[context] = {"rank": cross_rank, "offset": offset, "size": len(payload)}
                counts["selected_run_contexts"] += 1
            print(
                json.dumps(
                    {"runs_processed": counts["runs_processed"] + 1, "selected_contexts": len(best)}
                ),
                flush=True,
            )
            counts["runs_processed"] += 1
    sources, phases, dispositions = Counter(), Counter(), Counter()
    unique_tasks = set()
    with (
        spool_path.open("rb") as spool,
        (a.output / "teacher-best-one.message-loss.jsonl").open("wb") as out,
    ):
        for context, pointer in sorted(best.items()):
            spool.seek(pointer["offset"])
            encoded = spool.read(pointer["size"])
            row = json.loads(encoded)
            assert row["task_group"] not in held_roots | dev_roots | prior_roots
            assert not scanner.validate_row(row)
            out.write(encoded)
            sources[row["source"]] += 1
            phases[row["sample_phase"]] += 1
            dispositions[row["selection_signals"]["scanner_disposition"]] += 1
            unique_tasks.add(row["task_group"])
    for path, old_hash in provenance["c2_sha256"].items():
        if crawler._sha256(Path(path)) != old_hash:
            raise ValueError("C2 changed during selection")
    counts["selected_unique_contexts"] = len(best)
    counts["selected_unique_tasks"] = len(unique_tasks)
    crawler._write_json_atomic(
        a.output / "summary.json",
        {
            "counts": dict(counts),
            "source_counts": dict(sources),
            "phase_counts": dict(phases),
            "selected_scanner_dispositions": dict(dispositions),
            "provenance": provenance,
            "limitations": [
                "Provisional best-of heuristic, correctness not verified",
                "No semantic near-duplicate certification",
                "No real tokenizer length check or C3 recipe export yet",
                "Future use of older C2 needs the reserved-group exclusion manifest",
            ],
            "output_hashes": {
                p.name: crawler._sha256(p) for p in a.output.iterdir() if p.is_file()
            },
        },
    )
    print(json.dumps(dict(counts)), flush=True)


if __name__ == "__main__":
    main()
