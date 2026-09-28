#!/usr/bin/env python3
"""Build equal-count, behavior-aware trajectory groups for paired teacher/King data.

Every output pair receives the same K on both sides, where 4 <= K <= 6. Assistant turns are
atomic: the script never splits the content of one assistant response. Pairs for which either
side has fewer than four assistant turns are quarantined as structurally ineligible.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    sys.path.insert(0, str(location))

import audit_rollouts as rollout_audit  # noqa: E402
import build_training_catalog as catalog  # noqa: E402


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def flatten_side(row: dict[str, Any], side: str) -> list[dict[str, Any]]:
    return [
        message
        for group in row[f"{side}_groups"]
        for message in group["messages"]
    ]


def _behavior(content: str, *, edit_seen: bool, final_turn: bool) -> tuple[str, dict[str, bool]]:
    commands = catalog.action_blocks(content)
    edited = bool(catalog.edited_in_turn(content))
    tested = any(rollout_audit.TEST_RE.search(command) for command in commands)
    targeted = any(rollout_audit._is_targeted_inspect(command) for command in commands)
    diffed = any(rollout_audit.GIT_DIFF_RE.search(command) for command in commands)
    submitted = any(
        token in content.lower()
        for token in ("albedo submit", "albedo activate", "submit_marker")
    )
    flags = {
        "has_edit": edited,
        "has_test": tested,
        "has_targeted_inspect": targeted,
        "has_diff": diffed,
        "has_submit": submitted,
    }
    if submitted or (final_turn and not commands):
        return "finalize", flags
    if edited:
        return "edit", flags
    if tested:
        return ("verify_test" if edit_seen else "reproduce_test"), flags
    if diffed:
        return ("verify_inspect" if edit_seen else "inspect_diagnose"), flags
    if targeted or commands:
        return ("verify_inspect" if edit_seen else "inspect_diagnose"), flags
    return "reason_plan", flags


def atomic_units(completion: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positions = [
        index for index, message in enumerate(completion)
        if str(message.get("role") or "").lower() == "assistant"
    ]
    units: list[dict[str, Any]] = []
    edit_seen = False
    for ordinal, assistant_start in enumerate(positions, 1):
        start = 0 if ordinal == 1 else assistant_start
        end = positions[ordinal] if ordinal < len(positions) else len(completion)
        content = str(completion[assistant_start].get("content") or "")
        behavior, flags = _behavior(
            content, edit_seen=edit_seen, final_turn=ordinal == len(positions)
        )
        units.append({
            "ordinal": ordinal,
            "start": start,
            "end": end,
            "behavior": behavior,
            "flags": flags,
            "messages": completion[start:end],
        })
        edit_seen |= flags["has_edit"]
    return units


def _interval_cost(labels: list[str], start: int, end: int, target: float) -> float:
    histogram = Counter(labels[start:end])
    heterogeneous = (end - start) - max(histogram.values())
    balance = ((end - start) - target) ** 2
    split_same_behavior = 1 if start and labels[start - 1] == labels[start] else 0
    return heterogeneous * 100.0 + split_same_behavior * 8.0 + balance


def optimal_boundaries(units: list[dict[str, Any]], group_count: int) -> list[tuple[int, int]]:
    """Partition atomic turns into K contiguous, behavior-coherent non-empty groups."""

    count = len(units)
    if not 1 <= group_count <= count:
        raise ValueError(f"cannot partition {count} units into {group_count} groups")
    labels = [str(unit["behavior"]) for unit in units]
    target = count / group_count
    inf = float("inf")
    dp = [[inf] * (count + 1) for _ in range(group_count + 1)]
    previous = [[-1] * (count + 1) for _ in range(group_count + 1)]
    dp[0][0] = 0.0
    for groups in range(1, group_count + 1):
        for end in range(groups, count + 1):
            for start in range(groups - 1, end):
                candidate = dp[groups - 1][start] + _interval_cost(
                    labels, start, end, target
                )
                if candidate < dp[groups][end]:
                    dp[groups][end] = candidate
                    previous[groups][end] = start
    bounds: list[tuple[int, int]] = []
    end = count
    for groups in range(group_count, 0, -1):
        start = previous[groups][end]
        if start < 0:
            raise ValueError("partition dynamic program did not find a solution")
        bounds.append((start, end))
        end = start
    return list(reversed(bounds))


def grouped(units: list[dict[str, Any]], group_count: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for group_index, (start, end) in enumerate(
        optimal_boundaries(units, group_count), 1
    ):
        selected = units[start:end]
        histogram = Counter(str(unit["behavior"]) for unit in selected)
        dominant = max(histogram, key=lambda label: (histogram[label], -list(histogram).index(label)))
        messages = [message for unit in selected for message in unit["messages"]]
        result.append({
            "group_index": group_index,
            "dominant_behavior": dominant,
            "behavior_histogram": dict(histogram),
            "homogeneous": len(histogram) == 1,
            "assistant_turn_start": selected[0]["ordinal"],
            "assistant_turn_end": selected[-1]["ordinal"],
            "message_start": selected[0]["start"],
            "message_end": selected[-1]["end"],
            "has_edit": any(unit["flags"]["has_edit"] for unit in selected),
            "has_test": any(unit["flags"]["has_test"] for unit in selected),
            "has_targeted_inspect": any(
                unit["flags"]["has_targeted_inspect"] for unit in selected
            ),
            "has_diff": any(unit["flags"]["has_diff"] for unit in selected),
            "has_submit": any(unit["flags"]["has_submit"] for unit in selected),
            "messages": messages,
        })
    return result


def validate_groups(
    completion: list[dict[str, Any]], groups: list[dict[str, Any]], expected_count: int
) -> None:
    if len(groups) != expected_count or not 4 <= len(groups) <= 6:
        raise ValueError("group count violates the joint 4-6 contract")
    cursor = 0
    flattened: list[dict[str, Any]] = []
    for group in groups:
        if group["message_start"] != cursor:
            raise ValueError("non-contiguous group boundary")
        if group["message_end"] - group["message_start"] != len(group["messages"]):
            raise ValueError("group boundary length mismatch")
        cursor = group["message_end"]
        flattened.extend(group["messages"])
    if flattened != completion:
        raise ValueError("groups do not exactly reconstruct the continuation")


def build(source: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    handles = {
        split: (output / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
        for split in ("train", "dev")
    }
    exceptions = (output / "exceptions.jsonl").open("w", encoding="utf-8", newline="\n")
    try:
        for split in ("train", "dev"):
            for row in rows(source / f"{split}.jsonl"):
                teacher_completion = flatten_side(row, "teacher")
                king_completion = flatten_side(row, "king")
                teacher_units = atomic_units(teacher_completion)
                king_units = atomic_units(king_completion)
                minimum_turns = min(len(teacher_units), len(king_units))
                if minimum_turns < 4:
                    exceptions.write(json.dumps({
                        "schema": "albedo-joint-behavior-group-exception-v1",
                        "trajectory_id": row["trajectory_id"],
                        "king_trajectory_id": row["king_trajectory_id"],
                        "sample_id": row["sample_id"],
                        "split": split,
                        "teacher_assistant_turns": len(teacher_units),
                        "king_assistant_turns": len(king_units),
                        "reason": "fewer_than_four_atomic_assistant_turns",
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[f"exceptions:{split}"] += 1
                    continue
                group_count = min(6, minimum_turns)
                teacher_groups = grouped(teacher_units, group_count)
                king_groups = grouped(king_units, group_count)
                validate_groups(teacher_completion, teacher_groups, group_count)
                validate_groups(king_completion, king_groups, group_count)
                alignment = []
                for index, (teacher_group, king_group) in enumerate(
                    zip(teacher_groups, king_groups), 1
                ):
                    same = (
                        teacher_group["dominant_behavior"]
                        == king_group["dominant_behavior"]
                    )
                    alignment.append({
                        "group_index": index,
                        "teacher_behavior": teacher_group["dominant_behavior"],
                        "king_behavior": king_group["dominant_behavior"],
                        "same_behavior": same,
                    })
                    counts[f"aligned_behavior:{same}"] += 1
                    counts[f"teacher_behavior:{teacher_group['dominant_behavior']}"] += 1
                    counts[f"king_behavior:{king_group['dominant_behavior']}"] += 1
                    counts[f"teacher_homogeneous:{teacher_group['homogeneous']}"] += 1
                    counts[f"king_homogeneous:{king_group['homogeneous']}"] += 1
                record = {
                    **{
                        key: row.get(key)
                        for key in (
                            "trajectory_id", "king_trajectory_id", "sample_id", "instance_id",
                            "task_group", "source", "sample_phase", "split", "prompt",
                            "teacher_model_uri", "king_model_uri", "selected_king_sample_id",
                            "selected_king_rollout_sample_id", "prompt_sha256",
                            "preference_status",
                        )
                    },
                    "schema": "albedo-joint-behavior-groups-v1",
                    "group_count": group_count,
                    "teacher_assistant_turns": len(teacher_units),
                    "king_assistant_turns": len(king_units),
                    "teacher_groups": teacher_groups,
                    "king_groups": king_groups,
                    "group_alignment": alignment,
                }
                handles[split].write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                counts[f"selected:{split}"] += 1
                counts[f"group_count:{group_count}"] += 1
                counts["teacher_reconstructed_exact"] += 1
                counts["king_reconstructed_exact"] += 1
    finally:
        for handle in handles.values():
            handle.close()
        exceptions.close()
    total_alignment = counts["aligned_behavior:True"] + counts["aligned_behavior:False"]
    summary = {
        "schema": "albedo-joint-behavior-groups-summary-v1",
        "source": str(source.resolve()),
        "counts": dict(sorted(counts.items())),
        "same_behavior_alignment_rate": (
            counts["aligned_behavior:True"] / total_alignment if total_alignment else None
        ),
        "contract": {
            "minimum_groups": 4,
            "maximum_groups": 6,
            "equal_group_count_within_pair": True,
            "assistant_turns_are_atomic": True,
            "preference_assigned": False,
            "groups_are_independently_valid_dpo_pairs": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
