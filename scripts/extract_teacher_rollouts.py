#!/usr/bin/env python3
"""Extract unfiltered GLM reference trajectories from downloaded rollout artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield line_number, value


def _metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run-metadata.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _generated_by_sample(path: Path) -> dict[str, tuple[int, dict[str, Any]]]:
    if not path.is_file():
        return {}
    return {
        str(row.get("sample_id")): (line_number, row)
        for line_number, row in _jsonl(path)
        if row.get("sample_id")
    }


def _aligned_value(value: Any, index: int) -> Any:
    """Return an item from a per-reference vector without treating strings as vectors."""

    return value[index] if isinstance(value, list) and index < len(value) else None


def _reference_entries(source: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Normalize legacy singular and current multi-reference question-source schemas.

    The milestone-ladder protocol publishes ``reference_trajectories`` and
    ``reference_models`` arrays.  It also keeps ``reference_trajectory`` as a legacy alias for
    the first trajectory; reading both would duplicate reference one.
    """

    trajectories = source.get("reference_trajectories")
    if isinstance(trajectories, list):
        usable = [
            (index, trajectory)
            for index, trajectory in enumerate(trajectories)
            if isinstance(trajectory, str) and trajectory.strip()
        ]
        if usable:
            fallback_model = str(source.get("reference_model") or source.get("model") or "unknown")
            return (
                [
                    {
                        "reference_index": index + 1,
                        "trajectory": trajectory,
                        "model": str(
                            _aligned_value(source.get("reference_models"), index) or fallback_model
                        ),
                        "self_score": _aligned_value(source.get("reference_self_scores"), index),
                        "made_edit": _aligned_value(source.get("reference_made_edits"), index),
                    }
                    for index, trajectory in usable
                ],
                True,
            )

    trajectory = source.get("reference_trajectory")
    if not isinstance(trajectory, str) or not trajectory.strip():
        return [], False
    return (
        [
            {
                "reference_index": 1,
                "trajectory": trajectory,
                "model": str(source.get("reference_model") or source.get("model") or "unknown"),
                "self_score": source.get("reference_self_score"),
                "made_edit": source.get("reference_made_edit"),
            }
        ],
        False,
    )


def extract(corpus: Path, output: Path) -> dict[str, Any]:
    corpus = corpus.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "teacher-occurrences.jsonl"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")

    counters: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    sample_counts: Counter[str] = Counter()
    reference_hashes: set[str] = set()
    seen_multi_reference_sets: set[tuple[str, str, str]] = set()
    scoring_files = sorted(corpus.glob("runs/*/*/scoring-results.jsonl"))

    with temporary.open("w", encoding="utf-8", newline="\n") as writer:
        for scoring_path in scoring_files:
            run_dir = scoring_path.parent
            run_id = run_dir.name
            metadata = _metadata(run_dir)
            dashboard_run = metadata.get("dashboard_run")
            dashboard_run = dashboard_run if isinstance(dashboard_run, dict) else {}
            generated_path = run_dir / "generated-samples.jsonl"
            generated = _generated_by_sample(generated_path)
            scoring_occurrences: Counter[str] = Counter()
            counters["scoring_files"] += 1
            counters["generated_files"] += int(generated_path.is_file())

            for scoring_line, scoring in _jsonl(scoring_path):
                counters["scoring_rows"] += 1
                sample_id = str(scoring.get("sample_id") or "")
                scoring_occurrences[sample_id] += 1
                pass_index = scoring_occurrences[sample_id]
                source = scoring.get("question_source")
                source = source if isinstance(source, dict) else {}
                references, is_multi_reference = _reference_entries(source)
                if not references:
                    counters["missing_reference_trajectory"] += 1
                    continue

                reference_set_hash = hashlib.sha256(
                    json.dumps(
                        [
                            hashlib.sha256(reference["trajectory"].encode("utf-8")).hexdigest()
                            for reference in references
                        ],
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                reference_set_key = (run_id, sample_id, reference_set_hash)
                if is_multi_reference and reference_set_key in seen_multi_reference_sets:
                    # r1/r2 candidate rollouts share one frozen question source.  Its three
                    # references are repeated in both scoring rows but were generated only once.
                    counters["duplicate_multi_reference_sets_skipped"] += 1
                    continue
                if is_multi_reference:
                    seen_multi_reference_sets.add(reference_set_key)
                    counters["multi_reference_sets"] += 1
                else:
                    counters["legacy_reference_sets"] += 1
                counters["reference_sets"] += 1

                generated_record = generated.get(sample_id)
                rollout_sample_id = sample_id
                if generated_record is None:
                    rollout_sample_id = f"{sample_id}#r{pass_index}"
                    generated_record = generated.get(rollout_sample_id)
                generated_line = generated_record[0] if generated_record else None
                generated_row = generated_record[1] if generated_record else {}
                if generated_record is None:
                    counters["missing_generated_sample"] += 1
                else:
                    counters["joined_generated_sample"] += 1

                phase = str(source.get("sample_phase") or "unknown")
                for reference in references:
                    trajectory = reference["trajectory"]
                    reference_hash = hashlib.sha256(trajectory.encode("utf-8")).hexdigest()
                    reference_hashes.add(reference_hash)
                    sample_counts[sample_id] += 1
                    model = reference["model"]
                    model_counts[model] += 1
                    phase_counts[phase] += 1
                    counters["reference_occurrences"] += 1

                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "eval_run_id": run_id,
                        "sample_id": sample_id,
                        "rollout_sample_id": rollout_sample_id,
                        "pass_index": (
                            None
                            if is_multi_reference
                            else (pass_index if rollout_sample_id != sample_id else 1)
                        ),
                        "reference_scope": (
                            "base_sample" if is_multi_reference else "scoring_occurrence"
                        ),
                        "shared_across_rollouts": is_multi_reference,
                        "reference_index": reference["reference_index"],
                        "reference_count": len(references),
                        "reference_set_sha256": reference_set_hash,
                        "teacher_model": model,
                        "sample_phase": phase,
                        "reference_trajectory": trajectory,
                        "reference_sha256": reference_hash,
                        "reference_self_score": reference["self_score"],
                        "reference_made_edit": reference["made_edit"],
                        "scored": scoring.get("scored"),
                        "questions": scoring.get("questions"),
                        "prompt": generated_row.get("prompt"),
                        "rewrite_mode": generated_row.get("rewrite_mode"),
                        "submit_command": generated_row.get("submit_command"),
                        "submit_marker": generated_row.get("submit_marker"),
                        "source_model_uri": dashboard_run.get("model_uri"),
                        "finished_at": dashboard_run.get("finished_at"),
                        "scoring_source": scoring_path.relative_to(corpus).as_posix(),
                        "scoring_line": scoring_line,
                        "generated_source": (
                            generated_path.relative_to(corpus).as_posix()
                            if generated_path.is_file()
                            else None
                        ),
                        "generated_line": generated_line,
                    }
                    writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    os.replace(temporary, destination)
    duplicate_occurrences = sum(count - 1 for count in sample_counts.values() if count > 1)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "corpus": str(corpus),
        "output": str(destination),
        **dict(sorted(counters.items())),
        "unique_sample_ids": len(sample_counts),
        "repeated_sample_occurrences": duplicate_occurrences,
        "unique_reference_sha256": len(reference_hashes),
        "teacher_model_counts": dict(sorted(model_counts.items())),
        "sample_phase_counts": dict(sorted(phase_counts.items())),
        "filtering_applied": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.corpus, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
