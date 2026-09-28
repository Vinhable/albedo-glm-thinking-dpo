#!/usr/bin/env python3
"""Export a King-only message-loss corpus paired row-for-row with C3-Q teacher data.

The source rollout archive remains authoritative.  This exporter never ranks King
replicas by score: it deterministically prefers the exact sample id, then r1, r2,
... after requiring an exact role/content prefix match at C3-Q's completion_start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPLICA_RE = re.compile(r"#r(\d+)$")


def json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def sha256_text(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_sample_id(sample_id: str) -> str:
    return REPLICA_RE.sub("", sample_id)


def replica_sort_key(sample_id: str, requested: str) -> tuple[int, int, str]:
    if sample_id == requested:
        return (0, 0, sample_id)
    match = REPLICA_RE.search(sample_id)
    if match:
        return (1, int(match.group(1)), sample_id)
    return (2, 0, sample_id)


def clean_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return []
        role, content = item.get("role"), item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return []
        messages.append({"role": role, "content": content})
    return messages


def canonical_messages(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Normalize only transport-level line endings/trailing whitespace."""
    return [
        (message["role"], message["content"].replace("\r\n", "\n").rstrip())
        for message in messages
    ]


def prefix_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = clean_messages(row.get("messages"))
    boundary = row.get("completion_start")
    if type(boundary) is not int or not 0 <= boundary < len(messages):
        raise ValueError("invalid completion_start")
    return messages[:boundary]


def valid_king_candidate(sample: dict[str, Any]) -> bool:
    error = sample.get("king_error")
    return not error and bool(clean_messages(sample.get("previous_king_turns")))


def select_candidate(
    teacher: dict[str, Any], candidates: Iterable[dict[str, Any]]
) -> tuple[dict[str, Any] | None, list[str], Counter[str]]:
    requested = str(teacher["sample_id"])
    requested_run = str(teacher["eval_run_id"])
    prefix = prefix_messages(teacher)
    diagnostics: Counter[str] = Counter()
    compatible: list[dict[str, Any]] = []
    candidate_ids: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("sample_id") or "")
        candidate_run = str(candidate.get("_archive_eval_run_id") or requested_run)
        candidate_ids.append(f"{candidate_run}:{candidate_id}")
        if not valid_king_candidate(candidate):
            diagnostics["invalid_king"] += 1
            continue
        turns = clean_messages(candidate["previous_king_turns"])
        if len(turns) <= len(prefix):
            diagnostics["no_continuation"] += 1
            continue
        raw_prefix = turns[: len(prefix)]
        if raw_prefix == prefix:
            candidate = {**candidate, "_prefix_match_kind": "exact"}
        elif canonical_messages(raw_prefix) == canonical_messages(prefix):
            candidate = {**candidate, "_prefix_match_kind": "canonical_rstrip"}
        else:
            diagnostics["prefix_mismatch"] += 1
            continue
        compatible.append(candidate)
    compatible.sort(
        key=lambda item: (
            str(item.get("_archive_eval_run_id") or requested_run) != requested_run,
            *replica_sort_key(str(item["sample_id"]), requested),
            str(item.get("_archive_eval_run_id") or requested_run),
        )
    )
    return (compatible[0] if compatible else None), sorted(candidate_ids), diagnostics


def load_archive_index(
    rollout_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    connection = sqlite3.connect(rollout_root / "index" / "rollouts.sqlite3")
    try:
        paths = {
            str(run_id): rollout_root / str(relative_path)
            for run_id, relative_path in connection.execute(
                "SELECT eval_run_id, relative_path FROM artifacts "
                "WHERE artifact_type='GENERATED_SAMPLES' AND status IN ('downloaded','validated') "
                "AND relative_path IS NOT NULL"
            )
        }
        runs: dict[str, dict[str, Any]] = {}
        for run_id, raw in connection.execute(
            "SELECT eval_run_id, dashboard_run_json FROM runs"
        ):
            document = json.loads(raw)
            king = document.get("king") if isinstance(document.get("king"), dict) else {}
            runs[str(run_id)] = {
                "king_model_uri": king.get("model_uri"),
                "king_version": king.get("king_version"),
                "king_hotkey": king.get("hotkey"),
                "eval_finished_at": document.get("finished_at"),
            }
        return paths, runs
    finally:
        connection.close()


def load_needed_rollouts(
    rollout_root: Path,
    wanted: dict[str, set[str]],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    paths, _ = load_archive_index(rollout_root)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_runs: list[str] = []
    wanted_bases = set().union(*wanted.values()) if wanted else set()
    for run_id in wanted:
        if run_id not in paths or not paths[run_id].is_file():
            missing_runs.append(run_id)
    # A source sample can occur in multiple eval runs. Search the whole public
    # archive so a version-shifted source run can be rescued by a different run
    # with an exactly matching canonical prefix.
    for run_id, path in sorted(paths.items()):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                sample_id = str(sample.get("sample_id") or "")
                base = base_sample_id(sample_id)
                if base in wanted_bases:
                    sample["_archive_eval_run_id"] = run_id
                    candidates[base].append(sample)
    return candidates, missing_runs


def read_teacher_rows(c3_root: Path) -> tuple[list[tuple[str, int, dict[str, Any]]], dict[str, set[str]]]:
    rows: list[tuple[str, int, dict[str, Any]]] = []
    wanted: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "dev"):
        path = c3_root / f"{split}.message-loss.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"blank line at {path}:{line_number}")
                row = json.loads(line)
                # The frozen C3-Q files are the authoritative task-group split.
                # Some teacher-refresh rows intentionally retain their upstream
                # provenance label ("candidate") in the embedded split field.
                run_id, sample_id = str(row["eval_run_id"]), str(row["sample_id"])
                prefix_messages(row)
                rows.append((split, line_number, row))
                wanted[run_id].add(base_sample_id(sample_id))
    return rows, wanted


def read_teacher_requirements(
    c3_root: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[int, dict[str, list[tuple[str, int]]]]]]:
    """Read only lightweight prefix fingerprints, not the multi-GB message payloads."""
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    requirements: dict[str, dict[int, dict[str, list[tuple[str, int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for split in ("train", "dev"):
        path = c3_root / f"{split}.message-loss.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"blank line at {path}:{line_number}")
                row = json.loads(line)
                prefix = prefix_messages(row)
                key = (split, line_number)
                base = base_sample_id(str(row["sample_id"]))
                canonical_hash = sha256_text(canonical_messages(prefix))
                entries[key] = {
                    "eval_run_id": str(row["eval_run_id"]),
                    "sample_id": str(row["sample_id"]),
                    "base_sample_id": base,
                    "boundary": int(row["completion_start"]),
                    "canonical_prefix_sha256": canonical_hash,
                    "exact_prefix_sha256": sha256_text(prefix),
                }
                requirements[base][len(prefix)][canonical_hash].append(key)
    return entries, requirements


def candidate_rank(candidate_run: str, candidate_id: str, entry: dict[str, Any]) -> tuple[Any, ...]:
    requested = str(entry["sample_id"])
    return (
        candidate_run != str(entry["eval_run_id"]),
        *replica_sort_key(candidate_id, requested),
        candidate_run,
    )


def scan_candidate_locations(
    paths: dict[str, Path],
    entries: dict[tuple[str, int], dict[str, Any]],
    requirements: dict[str, dict[int, dict[str, list[tuple[str, int]]]]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], Counter[str]]:
    """Find the best compatible King while retaining only file byte offsets."""
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for run_id, path in sorted(paths.items()):
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                if not raw_line.strip():
                    continue
                sample = json.loads(raw_line)
                sample_id = str(sample.get("sample_id") or "")
                base = base_sample_id(sample_id)
                by_boundary = requirements.get(base)
                if not by_boundary:
                    continue
                if not valid_king_candidate(sample):
                    counts["invalid_king"] += 1
                    continue
                turns = clean_messages(sample["previous_king_turns"])
                for boundary, by_hash in by_boundary.items():
                    if len(turns) <= boundary:
                        counts["no_continuation"] += 1
                        continue
                    raw_prefix = turns[:boundary]
                    canonical_hash = sha256_text(canonical_messages(raw_prefix))
                    teacher_keys = by_hash.get(canonical_hash)
                    if not teacher_keys:
                        continue
                    exact_hash = sha256_text(raw_prefix)
                    for key in teacher_keys:
                        entry = entries[key]
                        rank = candidate_rank(run_id, sample_id, entry)
                        current = selected.get(key)
                        if current is not None and current["rank"] <= rank:
                            continue
                        selected[key] = {
                            "rank": rank,
                            "path": str(path),
                            "offset": offset,
                            "length": len(raw_line),
                            "eval_run_id": run_id,
                            "sample_id": sample_id,
                            "prefix_match_kind": (
                                "exact"
                                if exact_hash == entry["exact_prefix_sha256"]
                                else "canonical_rstrip"
                            ),
                        }
                        counts["compatible_candidates"] += 1
    return selected, counts


def read_candidate_at(locator: dict[str, Any], handles: dict[str, Any]) -> dict[str, Any]:
    path = str(locator["path"])
    handle = handles.get(path)
    if handle is None:
        handle = Path(path).open("rb")
        handles[path] = handle
    handle.seek(int(locator["offset"]))
    raw = handle.read(int(locator["length"]))
    sample = json.loads(raw)
    sample["_archive_eval_run_id"] = locator["eval_run_id"]
    sample["_prefix_match_kind"] = locator["prefix_match_kind"]
    return sample


def king_row(
    teacher: dict[str, Any],
    candidate: dict[str, Any],
    run_meta: dict[str, Any],
    candidate_ids: list[str],
) -> dict[str, Any]:
    boundary = int(teacher["completion_start"])
    raw = clean_messages(candidate["previous_king_turns"])
    # Preserve the frozen C3-Q prompt byte-for-byte. Candidate eligibility has
    # already established an exact or transport-only canonical prefix match.
    clean = prefix_messages(teacher) + raw[boundary:]
    messages = [
        {**message, "loss": index >= boundary and message["role"] == "assistant"}
        for index, message in enumerate(clean)
    ]
    if not any(message["loss"] for message in messages):
        raise ValueError("selected King trajectory has no supervised assistant message")
    selected_id = str(candidate["sample_id"])
    teacher_tid = str(teacher["trajectory_id"])
    trajectory_id = hashlib.sha256(
        f"king-paired-v1\0{teacher_tid}\0{selected_id}\0{sha256_text(clean)}".encode("utf-8")
    ).hexdigest()
    return {
        **{key: value for key, value in teacher.items() if key not in {"messages", "trajectory_id", "model_uri", "origin", "c3_provenance"}},
        "trajectory_id": trajectory_id,
        "messages": messages,
        "completion_start": boundary,
        "origin": "king",
        "model_uri": run_meta.get("king_model_uri"),
        "split": teacher["split"],
        "dataset_version": "c3q-paired-king-v1",
        "paired_teacher_trajectory_id": teacher_tid,
        "paired_teacher_model_uri": teacher.get("model_uri"),
        "rollout_sample_id": selected_id,
        "king_eval_run_id": candidate.get("_archive_eval_run_id"),
        "pairing_method": "global_base_sample_canonical_prefix;prefer_source_run_then_exact_then_replica_order",
        "prefix_match_kind": candidate.get("_prefix_match_kind"),
        "king_version": run_meta.get("king_version"),
        "king_hotkey": run_meta.get("king_hotkey"),
        "eval_finished_at": run_meta.get("eval_finished_at"),
        "available_king_rollout_ids": candidate_ids,
        "selected_king_content_sha256": sha256_text(clean),
    }


def export(c3_root: Path, rollout_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    entries, requirements = read_teacher_requirements(c3_root)
    paths, run_meta = load_archive_index(rollout_root)
    source_runs = {str(entry["eval_run_id"]) for entry in entries.values()}
    missing_runs = sorted(source_runs - set(paths))
    selected, scan_counts = scan_candidate_locations(paths, entries, requirements)
    output.mkdir(parents=True)
    handles = {
        split: (output / f"{split}.message-loss.jsonl").open("w", encoding="utf-8", newline="\n")
        for split in ("train", "dev")
    }
    manifest = (output / "pairing-manifest.jsonl").open("w", encoding="utf-8", newline="\n")
    rejected = (output / "unpaired.jsonl").open("w", encoding="utf-8", newline="\n")
    counts: Counter[str] = Counter(scan_counts)
    unique_teacher: set[str] = set()
    unique_king_content: set[str] = set()
    candidate_handles: dict[str, Any] = {}
    try:
        for split in ("train", "dev"):
            teacher_path = c3_root / f"{split}.message-loss.jsonl"
            with teacher_path.open("r", encoding="utf-8") as teacher_handle:
                for line_number, line in enumerate(teacher_handle, 1):
                    teacher = json.loads(line)
                    teacher["split"] = split
                    key = (split, line_number)
                    run_id, sample_id = str(teacher["eval_run_id"]), str(teacher["sample_id"])
                    locator = selected.get(key)
                    if locator is None:
                        reason = "missing_run" if run_id in missing_runs else "no_compatible_king"
                        rejected.write(json_line({
                            "split": split,
                            "teacher_line": line_number,
                            "teacher_trajectory_id": teacher["trajectory_id"],
                            "eval_run_id": run_id,
                            "sample_id": sample_id,
                            "reason": reason,
                        }))
                        counts[f"unpaired:{reason}"] += 1
                        continue
                    candidate = read_candidate_at(locator, candidate_handles)
                    candidate_ids = [f"{locator['eval_run_id']}:{locator['sample_id']}"]
                    king_run = str(locator["eval_run_id"])
                    exported = king_row(teacher, candidate, run_meta.get(king_run, {}), candidate_ids)
                    handles[split].write(json_line(exported))
                    unique_teacher.add(str(teacher["trajectory_id"]))
                    unique_king_content.add(str(exported["selected_king_content_sha256"]))
                    counts[f"paired:{split}"] += 1
                    same_run = king_run == run_id
                    exact_id = candidate["sample_id"] == sample_id
                    method = (
                        ("source_run" if same_run else "alternate_run")
                        + ("_exact_id" if exact_id else "_replica")
                        + "_"
                        + str(candidate.get("_prefix_match_kind"))
                    )
                    counts[f"selection:{method}"] += 1
                    manifest.write(json_line({
                        "split": split,
                        "teacher_line": line_number,
                        "teacher_trajectory_id": teacher["trajectory_id"],
                        "king_trajectory_id": exported["trajectory_id"],
                        "eval_run_id": run_id,
                        "teacher_sample_id": sample_id,
                        "selected_king_sample_id": candidate["sample_id"],
                        "selected_king_eval_run_id": king_run,
                        "selection": method,
                        "prefix_sha256": sha256_text(prefix_messages(teacher)),
                        "king_content_sha256": exported["selected_king_content_sha256"],
                    }))
    finally:
        for handle in candidate_handles.values():
            handle.close()
        for handle in handles.values():
            handle.close()
        manifest.close()
        rejected.close()

    paired = counts["paired:train"] + counts["paired:dev"]
    summary = {
        "schema": "c3q-paired-king-v1",
        "source_c3q_root": str(c3_root),
        "source_rollout_root": str(rollout_root),
        "teacher_rows": len(entries),
        "paired_rows": paired,
        "unpaired_rows": len(entries) - paired,
        "unique_teacher_trajectories": len(unique_teacher),
        "unique_king_contents": len(unique_king_content),
        "missing_run_count": len(missing_runs),
        "missing_runs": sorted(missing_runs),
        "counts": dict(sorted(counts.items())),
        "selection_policy": "canonical prefix match; prefer source eval run, exact sample id, then numeric replica order; scores are never consulted",
        "loss_contract": "loss=true only for assistant messages at/after the paired C3-Q completion_start",
        "merged_into_c3q": False,
        "files": {},
    }
    for name in ("train.message-loss.jsonl", "dev.message-loss.jsonl", "pairing-manifest.jsonl", "unpaired.jsonl"):
        path = output / name
        summary["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--c3-root", type=Path, required=True)
    result.add_argument("--rollout-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    summary = export(args.c3_root.resolve(), args.rollout_root.resolve(), args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["unpaired_rows"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
