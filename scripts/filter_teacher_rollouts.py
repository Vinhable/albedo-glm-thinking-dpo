#!/usr/bin/env python3
"""Audit, deduplicate, and materialize GLM teacher trajectories for agent SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import audit_rollouts as rollout_audit  # noqa: E402
import build_training_catalog as catalog  # noqa: E402
from dataset_creator.extract import parse_trajectory  # noqa: E402

from albedo_eval_service.shared.edit_detection import edited_in_turn  # noqa: E402
from albedo_eval_service.shared.observation_format import detect_format  # noqa: E402

SCHEMA_VERSION = 2
CHAT_BLOCK_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant|tool|environment)\n(.*?)<\|im_end\|>",
    re.DOTALL,
)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def parse_prompt(prompt: str) -> list[dict[str, str]]:
    messages = [
        {"role": role, "content": content.strip()}
        for role, content in CHAT_BLOCK_RE.findall(prompt or "")
    ]
    if len(messages) < 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError(f"unexpected prompt roles: {[m['role'] for m in messages[:4]]}")
    if any(not message["content"] for message in messages):
        raise ValueError("prompt contains an empty closed chat block")
    return messages


def _catalog_lookup(
    connection: sqlite3.Connection | None, sample_id: str, phase: str
) -> dict[str, Any]:
    if connection is None:
        return {}
    row = connection.execute(
        """
        SELECT instance_id, source, shard_path, row_index, cut_point, sampling_probability
          FROM sampling_cutpoints
         WHERE sample_id = ? AND sample_phase = ?
         ORDER BY sampling_probability DESC
         LIMIT 1
        """,
        (sample_id, phase),
    ).fetchone()
    if row is None:
        row = connection.execute(
            """
            SELECT instance_id, source, shard_path, row_index, cut_point, sampling_probability
              FROM sampling_cutpoints
             WHERE sample_id = ?
             ORDER BY sampling_probability DESC
             LIMIT 1
            """,
            (sample_id,),
        ).fetchone()
    if row is None:
        return {}
    keys = ("instance_id", "source", "shard", "row_index", "cut_point", "sampling_probability")
    return dict(zip(keys, row, strict=True))


def _has_targeted_inspect(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for command in rollout_audit.strict_action_blocks(str(message.get("content") or "")):
            if rollout_audit._is_targeted_inspect(command):
                return True
    return False


def _quality_weight(record: dict[str, Any]) -> float:
    weight = 1.0
    score = float(record.get("reference_self_score") or 0.0)
    if score < 0.8:
        weight *= 0.5
    elif score < 0.9:
        weight *= 0.75
    if record["loop"]:
        weight *= 0.3
    if record["broad_search_without_inspect"]:
        weight *= 0.5
    if record["unsupported_completion_claim"]:
        weight *= 0.3
    if not record["made_edit"]:
        weight *= 0.5
    elif not record["workflow_complete"]:
        weight *= 0.7
    return round(max(0.05, weight), 4)


def _rank(record: dict[str, Any]) -> tuple[Any, ...]:
    soft_count = sum(
        bool(record[name])
        for name in ("loop", "broad_search_without_inspect", "unsupported_completion_claim")
    )
    return (
        int(record["workflow_complete"]),
        int(record["made_edit"]),
        -soft_count,
        float(record.get("reference_self_score") or -1.0),
        -int(record["trajectory_chars"]),
        str(record.get("finished_at") or ""),
    )


def _dedup_key(record: dict[str, Any]) -> tuple[str, int]:
    """Keep separate routes from a multi-reference set while preserving legacy behavior."""

    sample_id = str(record["sample_id"])
    is_multi_reference = (
        bool(record.get("shared_across_rollouts")) or int(record.get("reference_count") or 1) > 1
    )
    reference_index = int(record.get("reference_index") or 1) if is_multi_reference else 0
    return sample_id, reference_index


def _split_key(record: dict[str, Any], dev_fraction: float, seed: str) -> str:
    identity = str(record.get("instance_id") or record["sample_id"])
    value = (
        int.from_bytes(hashlib.sha256(f"{seed}\0{identity}".encode()).digest()[:8], "big") / 2**64
    )
    return "dev" if value < dev_fraction else "train"


def _training_row(record: dict[str, Any], *, tier: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in record["prompt_messages"]:
        item = dict(message)
        if item.get("role") == "assistant":
            item["loss"] = False
        messages.append(item)
    for message in record["completion_messages"]:
        item = dict(message)
        if item.get("role") == "assistant":
            item["loss"] = True
        messages.append(item)
    return {
        "messages": messages,
        "sample_id": record["sample_id"],
        "instance_id": record.get("instance_id"),
        "source": record.get("source"),
        "sample_phase": record["sample_phase"],
        "cut_point": record.get("cut_point"),
        "observation_format": record["observation_format"],
        "teacher_model": record["teacher_model"],
        "reference_self_score": record["reference_self_score"],
        "quality_weight": record["quality_weight"],
        "sampling_probability": record.get("sampling_probability"),
        "split": record["split"],
        "tier": tier,
        "eval_run_id": record["eval_run_id"],
        "reference_sha256": record["reference_sha256"],
        "reference_index": record.get("reference_index", 1),
        "reference_count": record.get("reference_count", 1),
        "reference_scope": record.get("reference_scope", "scoring_occurrence"),
        "reference_set_sha256": record.get("reference_set_sha256"),
        "provenance": {
            "scoring_source": record["scoring_source"],
            "scoring_line": record["scoring_line"],
            "generated_source": record["generated_source"],
            "generated_line": record["generated_line"],
            "finished_at": record.get("finished_at"),
        },
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def build(
    raw_path: Path,
    catalog_path: Path | None,
    output: Path,
    *,
    max_chars: int = 120_000,
    min_strict_score: float = 0.9,
    dev_fraction: float = 0.1,
    split_seed: str = "teacher-v1",
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    connection = (
        sqlite3.connect(f"file:{catalog_path.resolve()}?mode=ro", uri=True)
        if catalog_path is not None
        else None
    )
    audit_rows: list[dict[str, Any]] = []
    best: dict[tuple[str, int], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()

    try:
        for raw in _jsonl(raw_path):
            counts["raw_occurrences"] += 1
            sample_id = str(raw.get("sample_id") or "")
            phase = str(raw.get("sample_phase") or "unknown")
            hard_reasons: list[str] = []
            try:
                prompt_messages = parse_prompt(str(raw.get("prompt") or ""))
                completion_messages = parse_trajectory(str(raw.get("reference_trajectory") or ""))
                if not completion_messages or not any(
                    message.get("role") == "assistant" for message in completion_messages
                ):
                    raise ValueError("reference has no assistant turn")
            except (TypeError, ValueError) as exc:
                prompt_messages = []
                completion_messages = []
                hard_reasons.append(f"parse_error:{exc}")

            if completion_messages:
                quality = catalog.audit_trajectory(
                    completion_messages,
                    supplemental_text=str(raw.get("reference_trajectory") or ""),
                    max_trajectory_chars=max_chars,
                )
            else:
                quality = {
                    "malformed_bash": True,
                    "loop": False,
                    "broad_search_without_inspect": False,
                    "unsupported_completion_claim": False,
                    "inspect_before_edit": False,
                    "verify_after_edit": False,
                    "workflow_complete": False,
                    "environment_error": False,
                    "output_truncated": False,
                    "too_long": False,
                    "trajectory_chars": len(str(raw.get("reference_trajectory") or "")),
                    "assistant_turns": 0,
                }
            for flag in ("malformed_bash", "environment_error", "output_truncated", "too_long"):
                if quality[flag]:
                    hard_reasons.append(flag)
            for flag in (
                "malformed_bash",
                "loop",
                "broad_search_without_inspect",
                "unsupported_completion_claim",
                "environment_error",
                "output_truncated",
                "too_long",
            ):
                flag_counts[flag] += int(bool(quality[flag]))

            coordinate = _catalog_lookup(connection, sample_id, phase)
            assistant_text = [
                str(message.get("content") or "")
                for message in completion_messages
                if message.get("role") == "assistant"
            ]
            made_edit = bool(raw.get("reference_made_edit")) or any(
                edited_in_turn(text) for text in assistant_text
            )
            record = {
                **raw,
                **coordinate,
                **{
                    key: bool(quality[key])
                    for key in (
                        "malformed_bash",
                        "loop",
                        "broad_search_without_inspect",
                        "unsupported_completion_claim",
                        "inspect_before_edit",
                        "verify_after_edit",
                        "workflow_complete",
                        "environment_error",
                        "output_truncated",
                        "too_long",
                    )
                },
                "trajectory_chars": int(quality["trajectory_chars"]),
                "assistant_turns": int(quality["assistant_turns"]),
                "made_edit": made_edit,
                "has_targeted_inspect": _has_targeted_inspect(completion_messages),
                "hard_reject": bool(hard_reasons),
                "hard_reasons": hard_reasons,
                "prompt_messages": prompt_messages,
                "completion_messages": completion_messages,
                "observation_format": (
                    detect_format(sample_id, prompt_messages) if prompt_messages else None
                ),
            }
            record["quality_weight"] = _quality_weight(record) if not hard_reasons else 0.0
            record["split"] = _split_key(record, dev_fraction, split_seed)
            audit_rows.append(record)
            if hard_reasons:
                counts["hard_rejected_occurrences"] += 1
                continue
            counts["hard_valid_occurrences"] += 1
            dedup_key = _dedup_key(record)
            current = best.get(dedup_key)
            if current is None or _rank(record) > _rank(current):
                best[dedup_key] = record
    finally:
        if connection is not None:
            connection.close()

    selected = sorted(
        best.values(),
        key=lambda row: (
            row["sample_id"],
            int(row.get("reference_index") or 1),
            row["eval_run_id"],
        ),
    )
    strict = [
        row
        for row in selected
        if float(row.get("reference_self_score") or 0.0) >= min_strict_score
        and not row["loop"]
        and not row["broad_search_without_inspect"]
        and not row["unsupported_completion_claim"]
    ]
    strong = [row for row in strict if row["made_edit"] and row["workflow_complete"]]
    navigation = [row for row in strict if not row["made_edit"] and row["has_targeted_inspect"]]

    audit_export = []
    for row in audit_rows:
        audit_export.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "prompt",
                    "reference_trajectory",
                    "questions",
                    "prompt_messages",
                    "completion_messages",
                }
            }
        )
    files = {
        "audit": ("audit.jsonl", audit_export),
        "weighted": (
            "teacher-sft-weighted.jsonl",
            [_training_row(row, tier="weighted") for row in selected],
        ),
        "strict": (
            "teacher-sft-strict.jsonl",
            [_training_row(row, tier="strict") for row in strict],
        ),
        "strong_edit": (
            "teacher-sft-strong-edit.jsonl",
            [_training_row(row, tier="strong_edit") for row in strong],
        ),
        "navigation": (
            "teacher-sft-navigation.jsonl",
            [_training_row(row, tier="navigation") for row in navigation],
        ),
    }
    file_counts: dict[str, int] = {}
    split_file_counts: dict[str, dict[str, int]] = {}
    for name, (filename, rows) in files.items():
        file_counts[name] = _write_jsonl(output / filename, rows)
        if name == "audit":
            continue
        split_file_counts[name] = {
            split: _write_jsonl(
                output / name.replace("_", "-") / f"{split}.jsonl",
                (row for row in rows if row["split"] == split),
            )
            for split in ("train", "dev")
        }
    selected_flag_counts = {
        flag: sum(bool(row[flag]) for row in selected)
        for flag in ("loop", "broad_search_without_inspect", "unsupported_completion_claim")
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "raw_path": str(raw_path.resolve()),
        "catalog_path": str(catalog_path.resolve()) if catalog_path is not None else None,
        "filter_policy": {
            "hard_flags": [
                "parse_error",
                "malformed_bash",
                "environment_error",
                "output_truncated",
                "too_long",
            ],
            "soft_flags": ["loop", "broad_search_without_inspect", "unsupported_completion_claim"],
            "dedup_key": "sample_id+reference_index_for_multi_reference",
            "min_strict_score": min_strict_score,
            "max_trajectory_chars": max_chars,
            "dev_fraction": dev_fraction,
            "split_identity": "instance_id_fallback_sample_id",
        },
        "counts": dict(sorted(counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "selected_soft_flag_counts": selected_flag_counts,
        "file_counts": file_counts,
        "split_file_counts": split_file_counts,
        "selected_train": sum(row["split"] == "train" for row in selected),
        "selected_dev": sum(row["split"] == "dev" for row in selected),
        "selected_unique_instances": len(
            {str(row.get("instance_id") or row["sample_id"]) for row in selected}
        ),
        "selected_weight_sum": round(sum(float(row["quality_weight"]) for row in selected), 4),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        help="Optional training catalog; without it, split identity falls back to sample_id",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=120_000)
    parser.add_argument("--min-strict-score", type=float, default=0.9)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", default="teacher-v1")
    args = parser.parse_args()
    summary = build(
        args.raw,
        args.catalog,
        args.output,
        max_chars=args.max_chars,
        min_strict_score=args.min_strict_score,
        dev_fraction=args.dev_fraction,
        split_seed=args.split_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
