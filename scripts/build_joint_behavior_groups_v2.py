#!/usr/bin/env python3
"""Re-segment paired teacher/King trajectories with the corrected turn classifier.

Structural contract is unchanged from v1: every pair gets the same K on both sides with
4 <= K <= 6, assistant turns stay atomic, groups are contiguous, and the groups must reconstruct
the continuation exactly. Pairs where either side has fewer than four assistant turns remain
structurally ineligible.

What changes is the per-turn label, via `turn_behavior_v2`. v1 read shell commands for edit
detection with a fence regex that mispairs a ```python block against the following ```bash block,
counted reproduction scripts as repository edits, and started every continuation from a blank
state even when the prefix had already edited the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import turn_behavior_v2 as behavior_v2  # noqa: E402
from build_joint_behavior_groups import (  # noqa: E402
    flatten_side,
    optimal_boundaries,
    rows,
    validate_groups,
)

SCHEMA = "albedo-joint-behavior-groups-v2"
CARRIED_KEYS = (
    "trajectory_id", "king_trajectory_id", "sample_id", "instance_id", "task_group",
    "source", "sample_phase", "split", "prompt", "teacher_model_uri", "king_model_uri",
    "selected_king_sample_id", "selected_king_rollout_sample_id", "prompt_sha256",
    "preference_status",
)


def atomic_units(completion: list[dict[str, Any]], prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One unit per assistant turn, labelled with the trajectory state carried from the prefix."""
    classifier = behavior_v2.TrajectoryClassifier(prompt)
    positions = [
        index for index, message in enumerate(completion)
        if str(message.get("role") or "").lower() == "assistant"
    ]
    units: list[dict[str, Any]] = []
    for ordinal, assistant_start in enumerate(positions, 1):
        # The shared boundary can carry an environment observation before the first assistant
        # turn. It belongs to the first group rather than being dropped.
        start = 0 if ordinal == 1 else assistant_start
        end = positions[ordinal] if ordinal < len(positions) else len(completion)
        for message in completion[start:assistant_start]:
            if str(message.get("role") or "").lower() != "assistant":
                classifier.observe(str(message.get("content") or ""))
        content = str(completion[assistant_start].get("content") or "")
        label, flags = classifier.classify(content, final_turn=ordinal == len(positions))
        units.append({
            "ordinal": ordinal,
            "start": start,
            "end": end,
            "behavior": label,
            "flags": flags,
            "messages": completion[start:end],
        })
        for message in completion[assistant_start + 1:end]:
            if str(message.get("role") or "").lower() != "assistant":
                classifier.observe(str(message.get("content") or ""))
    return units


def grouped(units: list[dict[str, Any]], group_count: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for group_index, (start, end) in enumerate(optimal_boundaries(units, group_count), 1):
        selected = units[start:end]
        histogram = Counter(str(unit["behavior"]) for unit in selected)
        dominant = max(
            histogram, key=lambda label: (histogram[label], -list(histogram).index(label))
        )
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
            "has_scratch_write": any(unit["flags"]["has_scratch_write"] for unit in selected),
            "has_test": any(unit["flags"]["has_test"] for unit in selected),
            "has_targeted_inspect": any(unit["flags"]["has_targeted_inspect"] for unit in selected),
            "has_diff": any(unit["flags"]["has_diff"] for unit in selected),
            "has_submit": any(unit["flags"]["has_submit"] for unit in selected),
            "messages": [message for unit in selected for message in unit["messages"]],
        })
    return result


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
            for index, row in enumerate(rows(source / f"{split}.jsonl"), 1):
                prompt = row.get("prompt") or []
                teacher_completion = flatten_side(row, "teacher")
                king_completion = flatten_side(row, "king")
                teacher_units = atomic_units(teacher_completion, prompt)
                king_units = atomic_units(king_completion, prompt)
                minimum_turns = min(len(teacher_units), len(king_units))
                if minimum_turns < 4:
                    exceptions.write(json.dumps({
                        "schema": "albedo-joint-behavior-group-exception-v2",
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
                for position, (teacher_group, king_group) in enumerate(
                    zip(teacher_groups, king_groups), 1
                ):
                    same = teacher_group["dominant_behavior"] == king_group["dominant_behavior"]
                    alignment.append({
                        "group_index": position,
                        "teacher_behavior": teacher_group["dominant_behavior"],
                        "king_behavior": king_group["dominant_behavior"],
                        "same_behavior": same,
                    })
                    counts[f"aligned_behavior:{same}"] += 1
                    counts[f"teacher_behavior:{teacher_group['dominant_behavior']}"] += 1
                    counts[f"king_behavior:{king_group['dominant_behavior']}"] += 1
                    counts[f"teacher_homogeneous:{teacher_group['homogeneous']}"] += 1
                    counts[f"king_homogeneous:{king_group['homogeneous']}"] += 1

                aligned = sum(1 for entry in alignment if entry["same_behavior"])
                record = {
                    **{key: row.get(key) for key in CARRIED_KEYS},
                    "schema": SCHEMA,
                    "group_count": group_count,
                    "teacher_assistant_turns": len(teacher_units),
                    "king_assistant_turns": len(king_units),
                    "teacher_behavior_sequence": [g["dominant_behavior"] for g in teacher_groups],
                    "king_behavior_sequence": [g["dominant_behavior"] for g in king_groups],
                    "aligned_groups": aligned,
                    "aligned_fraction": round(aligned / group_count, 6),
                    "fully_aligned": aligned == group_count,
                    "teacher_has_edit": any(g["has_edit"] for g in teacher_groups),
                    "king_has_edit": any(g["has_edit"] for g in king_groups),
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
                counts[f"fully_aligned:{aligned == group_count}"] += 1
                if index % 500 == 0:
                    print(f"{split}: {index} pairs", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
        exceptions.close()

    total = counts["aligned_behavior:True"] + counts["aligned_behavior:False"]
    summary = {
        "schema": "albedo-joint-behavior-groups-summary-v2",
        "source": str(source.resolve()),
        "counts": dict(sorted(counts.items())),
        "same_behavior_alignment_rate": (
            counts["aligned_behavior:True"] / total if total else None
        ),
        "contract": {
            "minimum_groups": 4,
            "maximum_groups": 6,
            "equal_group_count_within_pair": True,
            "assistant_turns_are_atomic": True,
            "preference_assigned": False,
            "groups_are_independently_valid_dpo_pairs": False,
        },
        "classifier": {
            "module": "turn_behavior_v2",
            "fence_pairing": "markers paired in document order, mid-line openers supported",
            "heredoc_bodies_excluded_from_edit_detection": True,
            "source_edit_requires_preexisting_path": True,
            "prompt_prefix_replayed": True,
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
    print(json.dumps(build(args.source, args.output)["counts"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
