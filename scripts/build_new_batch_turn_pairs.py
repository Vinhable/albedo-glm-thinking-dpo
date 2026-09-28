#!/usr/bin/env python3
"""Pair one teacher with one King rollout per sample of a fresh crawl, ready for re-segmentation.

Output rows use the `albedo-functional-turn-pairs-v1` shape that
`build_joint_behavior_groups_v2.py` reads (each side is one group; v2 re-segments it), so the new
batch goes through exactly the grouping and filters the existing DPO pool went through.

Teacher: one of the GLM reference runs, chosen the way the 2026-09-17 teacher refresh chose
(`select_teacher_refresh.candidate`): hard exclusions first, then fewer loop/detour cues, no broad
search without inspection, no unsupported completion claim, targeted inspection. When every valid
reference carries a score on the final checklist (artifacts from 2026-09-22), the score decides
first and those cues break ties. Otherwise the cues alone decide and the chosen teacher still has
to be judged.

King: the r1 rollout (replica order, as `export_paired_king_c3q.py` paired the existing pool), with
its frozen three-vote score on the same checklist. Scores never pick the King rollout.

Split: a sample that shares a task coordinate, task text or exact context with a pair already in
the pool inherits that pair's split, and an exact context match is dropped as a duplicate. Other
samples use the pool's deterministic task-group split (`split_c3q_train_dev.py`). Samples whose
task appears in a local evaluation batch are excluded.

No network or paid API call is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import build_sft_c_dataset as base  # noqa: E402
import select_teacher_refresh as refresh  # noqa: E402
from dataset_creator.extract import _references  # noqa: E402
from export_paired_king_c3q import canonical_messages, clean_messages  # noqa: E402

SCHEMA = "albedo-functional-turn-pairs-v1"


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha_messages(messages: list[dict[str, Any]]) -> str:
    raw = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def pool_split(group: str) -> str:
    """The deterministic split of `split_c3q_train_dev.py`."""
    value = int.from_bytes(hashlib.sha256(("c3q-dev-v1\0" + group).encode()).digest()[:8], "big")
    return "dev" if value / 2**64 < 0.1 else "train"


def identities(sample_id: str, prompt: list[dict[str, Any]]) -> list[str]:
    return refresh.identities(sample_id, prompt)


def load_pool(pool_root: Path) -> tuple[dict[str, str], set[str]]:
    """Identity -> split for every pair already in the pool, plus the pool's exact contexts."""
    split_of: dict[str, str] = {}
    contexts: set[str] = set()
    for split in ("train", "dev"):
        for row in rows(pool_root / f"{split}.jsonl"):
            keys = identities(str(row["sample_id"]), row["prompt"])
            for key in keys:
                split_of.setdefault(key, split)
            contexts.add(keys[2])
    return split_of, contexts


def load_eval_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        for row in rows(path):
            prompt = row.get("prompt")
            prompt = base.prompt_messages(prompt) if isinstance(prompt, str) else row.get("messages")
            if not prompt:
                raise ValueError(f"missing prefix in evaluation batch {path}")
            keys.update(identities(str(row["sample_id"]), prompt)[:2])
    return keys


def rendered(turns: list[dict[str, str]]) -> str:
    """The legacy `REFERENCE STEP n:` rendering the refresh scanner expects."""
    parts: list[str] = []
    step = 0
    for turn in turns:
        if turn["role"] == "assistant":
            step += 1
            parts.append(f"REFERENCE STEP {step}:\n{turn['content']}")
        else:
            parts.append(f"ENVIRONMENT OBSERVATION:\n{turn['content']}")
    return "\n\n".join(parts)


def teacher_candidates(
    prompt: list[dict[str, Any]], source: dict[str, Any], common: dict[str, Any],
    counts: Counter[str],
) -> list[dict[str, Any]]:
    scores = {s.get("run"): s.get("yes_rate") for s in source.get("reference_self_scores") or []}
    candidates: list[dict[str, Any]] = []
    for run, model, made_edit, turns in _references(source):
        turns = [
            {"role": t["role"], "content": str(t.get("content") or "").strip()}
            for t in turns if str(t.get("content") or "").strip()
        ]
        if not turns or turns[0]["role"] != "assistant":
            counts["reference_rejected:no_assistant_start"] += 1
            continue
        text = rendered(turns)
        entry = {"trajectory": text, "model": model, "reference_index": run, "self_score": None}
        try:
            row, _diag, rejects, rank = refresh.candidate(prompt, entry, common)
        except ValueError:
            counts["reference_rejected:parse"] += 1
            continue
        # The structured steps are authoritative; the rendering only feeds the scanner.
        if row["messages"][len(prompt):] != [{**t, "loss": t["role"] == "assistant"} for t in turns]:
            counts["reference_rejected:render_roundtrip"] += 1
            continue
        score = scores.get(run)
        candidates.append({
            "run": run,
            "model": model,
            "made_edit": made_edit,
            "completion": turns,
            "rejects": rejects,
            "rank": rank,
            "score": float(score) if isinstance(score, (int, float)) else None,
            "trajectory_id": row["trajectory_id"],
            "signals": row["selection_signals"],
        })
    return candidates


def choose_teacher(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
    valid = [c for c in candidates if not c["rejects"]]
    scored = bool(valid) and all(c["score"] is not None for c in valid)

    def key(c: dict[str, Any]) -> tuple[Any, ...]:
        return ((c["score"] if scored else -1.0), *c["rank"], c["trajectory_id"])

    return (max(valid, key=key) if valid else None), scored


def run_samples(run_dir: Path, counts: Counter[str]) -> Iterable[dict[str, Any]]:
    generated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows(run_dir / "generated-samples.jsonl"):
        generated[str(row["sample_id"]).split("#", 1)[0]].append(row)
    scoring: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line, row in enumerate(rows(run_dir / "scoring-results.jsonl"), 1):
        scoring[str(row["sample_id"]).split("#", 1)[0]].append({**row, "_line": line})

    for sid, scored_rows in scoring.items():
        counts["samples_seen"] += 1
        gens = sorted(
            generated.get(sid, []),
            key=lambda g: int(str(g["sample_id"]).rsplit("#r", 1)[-1])
            if "#r" in str(g["sample_id"]) else 0,
        )
        if not gens:
            counts["dropped:no_generated_rows"] += 1
            continue
        # The context is the turns before the first scored King turn. Re-parsing the rendered
        # prompt string misaligns whenever a message itself contains chat markers, which every
        # mini-coder-rs system prompt does.
        split_rollouts = []
        for replica, gen in enumerate(gens, 1):
            raw = gen.get("previous_king_turns")
            if gen.get("king_error") or not isinstance(raw, list):
                continue
            try:
                context, completion = base.split_turns(raw)
            except ValueError:
                counts["king_rollout_without_score_target"] += 1
                continue
            if any(isinstance(t, dict) and t.get("retry_feedback") for t in raw):
                counts["king_rollout_with_retry_feedback"] += 1
            split_rollouts.append((replica, gen, context, clean_messages(completion)))
        if not split_rollouts:
            counts["dropped:no_usable_king"] += 1
            continue
        if len({base.sha(canonical_messages(c)) for _, _, c, _ in split_rollouts}) != 1:
            counts["dropped:ambiguous_prefix"] += 1
            continue
        prompt = clean_messages(split_rollouts[0][2])
        try:
            parsed = base.prompt_messages(gens[0]["prompt"])
            counts[f"prompt_string_parse_agrees:{canonical_messages(parsed) == canonical_messages(prompt)}"] += 1
        except ValueError:
            counts["prompt_string_parse_agrees:False"] += 1

        king = None
        for replica, gen, _context, completion in split_rollouts:
            if replica > len(scored_rows) or scored_rows[replica - 1].get("king_score") is None:
                continue
            king = (replica, gen, completion, scored_rows[replica - 1])
            break
        if king is None:
            counts["dropped:no_usable_king"] += 1
            continue
        replica, gen, king_completion, king_scoring = king
        if not any(m["role"] == "assistant" for m in king_completion):
            counts["dropped:king_without_assistant_turn"] += 1
            continue

        source = king_scoring.get("question_source") or {}
        common = {
            "sample_id": sid,
            "eval_run_id": run_dir.name,
            "source": sid.split("/", 1)[0],
            "sample_phase": source.get("sample_phase"),
        }
        candidates = teacher_candidates(prompt, source, common, counts)
        teacher, score_decided = choose_teacher(candidates)
        if teacher is None:
            counts["dropped:no_valid_reference"] += 1
            continue
        questions = king_scoring.get("questions") or []
        if not questions:
            counts["dropped:no_frozen_checklist"] += 1
            continue
        yield {
            "sid": sid,
            "prompt": prompt,
            "teacher": teacher,
            "candidates": candidates,
            "score_decided": score_decided,
            "king_replica": replica,
            "king_sample_id": str(gen["sample_id"]),
            "king_completion": king_completion,
            "king_score": float(king_scoring["king_score"]),
            "king_amputated_thinking": king_scoring.get("king_amputated_thinking"),
            "scoring_line": king_scoring["_line"],
            "questions_sha256": base.sha(questions),
            "question_count": len(questions),
            "source_meta": common,
        }


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    pool_split_of, pool_contexts = load_pool(args.pool)
    eval_keys = load_eval_keys(args.eval_batch)
    finished: dict[str, str] = {}
    if args.dashboard:
        board = json.loads(args.dashboard.read_text(encoding="utf-8"))
        finished = {r["eval_run_id"]: r.get("finished_at") or "" for r in board.get("eval_runs", [])}

    # One pair per exact context: prefer an already-scored teacher (no judge call), then the
    # newest run.
    best: dict[str, tuple[tuple[Any, ...], int, int]] = {}
    spool_path = args.output / ".spool.jsonl"
    spool = spool_path.open("w+b")
    run_dirs = sorted(p for p in (args.root / "runs").glob("*/*") if p.is_dir())
    if args.max_runs:
        run_dirs = run_dirs[:args.max_runs]
    for run_dir in run_dirs:
        counts["runs"] += 1
        for sample in run_samples(run_dir, counts):
            sid, prompt = sample["sid"], sample["prompt"]
            keys = identities(sid, prompt)
            if keys[2] in pool_contexts:
                counts["dropped:context_already_in_pool"] += 1
                continue
            if eval_keys.intersection(keys[:2]):
                counts["dropped:task_in_eval_batch"] += 1
                continue
            inherited = next((pool_split_of[k] for k in keys if k in pool_split_of), None)
            task_group = keys[0]
            split = inherited or pool_split(task_group)
            counts[f"split_inherited:{bool(inherited)}"] += 1
            teacher = sample["teacher"]
            teacher_messages = [
                {**m, "loss": m["role"] == "assistant"} for m in teacher["completion"]
            ]
            king_messages = [
                {**m, "loss": m["role"] == "assistant"} for m in sample["king_completion"]
            ]
            record = {
                "schema": SCHEMA,
                "trajectory_id": teacher["trajectory_id"],
                "king_trajectory_id": base.sha([prompt, sample["king_completion"]]),
                "sample_id": sid,
                "instance_id": None,
                "task_group": task_group,
                "source": sample["source_meta"]["source"],
                "sample_phase": sample["source_meta"]["sample_phase"],
                "split": split,
                "prompt": prompt,
                "teacher_groups": [{"group_index": 1, "messages": teacher_messages}],
                "king_groups": [{"group_index": 1, "messages": king_messages}],
                "teacher_group_count": 1,
                "king_group_count": 1,
                "teacher_model_uri": teacher["model"],
                "king_model_uri": None,
                "selected_king_sample_id": sample["king_sample_id"],
                "selected_king_rollout_sample_id": sample["king_sample_id"],
                "prompt_sha256": sha_messages(prompt),
                "preference_status": (
                    "reference_scored" if sample["score_decided"] else "needs_teacher_judge"
                ),
            }
            meta = {
                "trajectory_id": teacher["trajectory_id"],
                "sample_id": sid,
                "eval_run_id": run_dir.name,
                "finished_at": finished.get(run_dir.name, ""),
                "split": split,
                "task_group": task_group,
                "context_sha256": keys[2],
                "teacher_reference_run": teacher["run"],
                "teacher_made_edit": teacher["made_edit"],
                "teacher_selection": "score_then_cues" if sample["score_decided"] else "cues_only",
                "teacher_reference_score": teacher["score"],
                "reference_candidates": [
                    {"run": c["run"], "score": c["score"], "rejects": c["rejects"], "rank": c["rank"]}
                    for c in sample["candidates"]
                ],
                "king_replica": sample["king_replica"],
                "king_sample_id": sample["king_sample_id"],
                "king_score": sample["king_score"],
                "king_amputated_thinking": sample["king_amputated_thinking"],
                "scoring_path": str(run_dir / "scoring-results.jsonl"),
                "scoring_line": sample["scoring_line"],
                "questions_sha256": sample["questions_sha256"],
                "question_count": sample["question_count"],
            }
            priority = (sample["score_decided"], meta["finished_at"], run_dir.name)
            context = keys[2]
            if context in best:
                counts["duplicate_context_within_batch"] += 1
                if priority <= best[context][0]:
                    continue
            payload = (compact({"record": record, "meta": meta}) + "\n").encode("utf-8")
            offset = spool.tell()
            spool.write(payload)
            best[context] = (priority, offset, len(payload))
        print(f"{run_dir.name[:8]} runs={counts['runs']} selected={len(best)}", flush=True)

    handles = {
        split: (args.output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
        for split in ("train", "dev")
    }
    try:
        with (args.output / "pairs-meta.jsonl").open("w", encoding="utf-8", newline="\n") as meta_out:
            for _context, (_priority, offset, size) in sorted(best.items()):
                spool.seek(offset)
                item = json.loads(spool.read(size).decode("utf-8"))
                record, meta = item["record"], item["meta"]
                handles[record["split"]].write(compact(record) + "\n")
                meta_out.write(compact(meta) + "\n")
                counts[f"selected:{record['split']}"] += 1
                counts[f"selected:{record['preference_status']}"] += 1
    finally:
        for handle in handles.values():
            handle.close()
        spool.close()
        spool_path.unlink(missing_ok=True)

    summary = {
        "schema": "albedo-new-batch-turn-pairs-summary-v1",
        "root": str(args.root),
        "pool": str(args.pool),
        "eval_batches": [str(p) for p in args.eval_batch],
        "counts": dict(sorted(counts.items())),
        "selected_pairs": len(best),
        "teacher_policy": "refresh cues (select_teacher_refresh.candidate); score first when every valid reference is scored",
        "king_policy": "first usable replica in order (r1), frozen three-vote score; scores never pick it",
        "api_calls": 0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="crawl root containing runs/")
    parser.add_argument("--pool", type=Path, required=True, help="joint behaviour groups v2 dir")
    parser.add_argument("--eval-batch", type=Path, action="append", default=[])
    parser.add_argument("--dashboard", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-runs", type=int, help="smoke test on the first N runs")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
