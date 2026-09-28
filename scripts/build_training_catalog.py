#!/usr/bin/env python3
"""Build a deduplicated training-data catalog from original corpora and public rollouts.

The catalog stores references and audit features only. It never copies or mutates raw parquet or
rollout artifacts, and every output is written below ``--output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import audit_rollouts as rollout_audit  # noqa: E402
import render_trajectories as renderer  # noqa: E402
from prepare_datasets import SOURCES  # noqa: E402

from albedo_eval_service.shared.edit_detection import edited_in_turn  # noqa: E402
from albedo_eval_service.shared.observation_format import (  # noqa: E402
    OPENHANDS_TRUNCATION_NOTICE,
    action_blocks,
    detect_format,
    is_truncated,
)
from generated_sample_fields import side_output  # noqa: E402

SCHEMA_VERSION = 2
SOURCE_LICENSES = {
    "mini-coder": "MIT",
    "mini-coder-rs": "MIT",
    "open-swe-traces": "CC-BY-4.0",
    "swe-hero": "CC-BY-4.0",
}

PATCH_PATH_RE = re.compile(
    r"^diff --git a/(.+?) b/(.+?)$|^\+\+\+\s+(?:b/)?([^\t\n]+)", re.MULTILINE
)
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".ex",
    ".exs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
}
TEST_PATH_RE = re.compile(
    r"(?:^|/)(?:tests?|testing|specs?|__tests__)(?:/|$)|"
    r"(?:^|/)(?:test_|spec_)[^/]+$|(?:_test|_spec)\.[^/]+$",
    re.IGNORECASE,
)
INFRA_ERROR_RE = re.compile(
    r"MODEL_RESPONSE_TOKEN_LIMIT_EXCEEDED|ABANDONED_AFTER_[A-Z_]+|"
    r"repo[- ]context (?:service )?(?:unavailable|failed)|"
    r"observation simulation (?:unavailable|failed)|"
    r"internal evaluator error|generation backend error",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any, *, pretty: bool = False) -> str:
    kwargs = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        return json.dumps(value, indent=2, **kwargs) + "\n"
    return json.dumps(value, separators=(",", ":"), **kwargs)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "resolved", "verified", "1"}:
            return True
        if lowered in {"false", "no", "unresolved", "-1", "0"}:
            return False
        return None
    if isinstance(value, (int, float)):
        return value > 0
    return bool(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_sample_id(sample_id: str) -> tuple[str, int, int] | None:
    """Split ``source/data/shard.parquet:row:turn`` from the right."""

    try:
        shard, row, turn = sample_id.rsplit(":", 2)
        return shard, int(row), int(turn)
    except (ValueError, AttributeError):
        return None


def _source_from_shard(shard: str) -> str:
    return shard.split("/", 1)[0]


def _message_content(message: Any) -> str:
    return str(message.get("content") or "") if isinstance(message, dict) else ""


def _messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [message for message in value if isinstance(message, dict)]


def _task_text(messages: list[dict[str, Any]]) -> str:
    seen_assistant = False
    user_parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").lower()
        if role == "assistant":
            seen_assistant = True
        elif role == "user" and not seen_assistant:
            user_parts.append(_message_content(message))
    return "\n".join(user_parts).strip()


def task_key(repo: str, messages: list[dict[str, Any]]) -> str:
    normalized_repo = repo.strip().lower().replace("__", "/")
    normalized_task = " ".join(_task_text(messages).split()).lower()
    return hashlib.sha256(f"{normalized_repo}\0{normalized_task}".encode()).hexdigest()


def patch_paths(text: str) -> list[str]:
    paths: set[str] = set()
    for match in PATCH_PATH_RE.finditer(text or ""):
        path = match.group(2) or match.group(3) or match.group(1) or ""
        path = path.strip().removeprefix("b/")
        if path and path != "/dev/null":
            paths.add(path)
    return sorted(paths)


def _is_source_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if TEST_PATH_RE.search(normalized):
        return False
    return Path(normalized).suffix.lower() in SOURCE_SUFFIXES


def _candidate_document(turns: Iterable[str]) -> str:
    blocks: list[str] = ["FULL CANDIDATE TRAJECTORY"]
    for index, turn in enumerate(turns, start=1):
        blocks.append(f"CANDIDATE OUTPUT {index}:\n------\n{turn}\n------")
        blocks.append(
            "ENVIRONMENT OBSERVATION:\n------\n<returncode>0</returncode>\n"
            "<output>\n</output>\n------"
        )
    return "\n\n".join(blocks)


def _sequence_features(messages: list[dict[str, Any]]) -> dict[str, Any]:
    assistant_turns = [
        _message_content(message)
        for message in messages
        if str(message.get("role") or "").lower() == "assistant"
    ]
    edit_indices = [
        index for index, turn in enumerate(assistant_turns, start=1) if edited_in_turn(turn)
    ]
    inspect_indices: list[int] = []
    verify_indices: list[int] = []
    for index, turn in enumerate(assistant_turns, start=1):
        commands = action_blocks(turn)
        targeted = any(rollout_audit._is_targeted_inspect(command) for command in commands)
        if targeted:
            inspect_indices.append(index)
        if targeted or any(
            rollout_audit.TEST_RE.search(command) or rollout_audit.GIT_DIFF_RE.search(command)
            for command in commands
        ):
            verify_indices.append(index)
    first_edit = min(edit_indices, default=None)
    inspect_before = bool(
        first_edit is not None and any(index < first_edit for index in inspect_indices)
    )
    verify_after = bool(
        first_edit is not None and any(index > first_edit for index in verify_indices)
    )
    return {
        "inspect_before_edit": inspect_before,
        "verify_after_edit": verify_after,
        "workflow_complete": inspect_before and verify_after,
        "first_edit_turn": first_edit,
        "inspect_turns": inspect_indices[:16],
        "verify_turns": verify_indices[:16],
    }


def audit_trajectory(
    messages: list[dict[str, Any]],
    *,
    patch: str = "",
    supplemental_text: str = "",
    explicit_error: str = "",
    max_trajectory_chars: int = 120_000,
) -> dict[str, Any]:
    """Audit a full trajectory with the four rollout flags plus training-quality gates."""

    assistant_turns = [
        _message_content(message)
        for message in messages
        if str(message.get("role") or "").lower() == "assistant"
    ]
    document = _candidate_document(assistant_turns)
    base = rollout_audit.audit_candidate(document)
    sequence = _sequence_features(messages)
    all_text = "\n".join(_message_content(message) for message in messages)
    patch_text = patch or supplemental_text
    modified_paths = patch_paths(patch_text)
    patch_present = bool(patch.strip()) or bool(modified_paths)
    source_patch = any(_is_source_path(path) for path in modified_paths)
    trajectory_chars = len(all_text)
    output_truncated = bool(
        OPENHANDS_TRUNCATION_NOTICE in all_text
        or any(is_truncated(turn) for turn in assistant_turns)
    )
    environment_error = bool(explicit_error.strip() or INFRA_ERROR_RE.search(all_text))
    too_long = trajectory_chars > max_trajectory_chars
    extended_flags = list(base["flags"])
    if not patch_present:
        extended_flags.append("missing_patch")
    elif not source_patch:
        extended_flags.append("no_source_file_patch")
    if not sequence["workflow_complete"]:
        extended_flags.append("incomplete_inspect_edit_verify")
    if too_long:
        extended_flags.append("trajectory_too_long")
    if environment_error:
        extended_flags.append("environment_error")
    if output_truncated:
        extended_flags.append("output_truncated")
    return {
        **{flag: bool(base[flag]) for flag in rollout_audit.MAJOR_FLAGS},
        "four_filter_pass": bool(base["quality_pass"]),
        "patch_present": patch_present,
        "patch_source_file": source_patch,
        "modified_paths": modified_paths,
        **{
            key: sequence[key]
            for key in ("inspect_before_edit", "verify_after_edit", "workflow_complete")
        },
        "too_long": too_long,
        "environment_error": environment_error,
        "output_truncated": output_truncated,
        "trajectory_chars": trajectory_chars,
        "estimated_tokens": (trajectory_chars + 3) // 4,
        "assistant_turns": len(assistant_turns),
        "extended_flags": extended_flags,
        "evidence": {
            "four_filters": base["evidence"],
            "modified_paths": modified_paths[:32],
            "sequence": {
                "first_edit_turn": sequence["first_edit_turn"],
                "inspect_turns": sequence["inspect_turns"],
                "verify_turns": sequence["verify_turns"],
            },
            "explicit_error": explicit_error[:500],
        },
    }


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE trajectories (
            trajectory_id TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            source TEXT NOT NULL,
            instance_id TEXT,
            task_key TEXT NOT NULL,
            repo TEXT,
            language TEXT,
            family TEXT,
            observation_format TEXT,
            source_resolved INTEGER,
            source_verified INTEGER,
            success_signal INTEGER,
            generator_model TEXT,
            dataset_license TEXT,
            repo_license TEXT,
            provenance_uri TEXT,
            upstream_dataset TEXT,
            eval_run_id TEXT,
            sample_id TEXT,
            side TEXT,
            snapshot_hash TEXT,
            shard_path TEXT,
            shard_sha256 TEXT,
            row_index INTEGER,
            sample_phase TEXT,
            cut_point INTEGER,
            sampling_probability REAL,
            score REAL,
            scored INTEGER NOT NULL,
            public_holdout INTEGER NOT NULL,
            four_filter_pass INTEGER NOT NULL,
            malformed_bash INTEGER NOT NULL,
            loop INTEGER NOT NULL,
            broad_search_without_inspect INTEGER NOT NULL,
            unsupported_completion_claim INTEGER NOT NULL,
            patch_present INTEGER NOT NULL,
            patch_source_file INTEGER NOT NULL,
            modified_paths_json TEXT NOT NULL,
            inspect_before_edit INTEGER NOT NULL,
            verify_after_edit INTEGER NOT NULL,
            workflow_complete INTEGER NOT NULL,
            too_long INTEGER NOT NULL,
            environment_error INTEGER NOT NULL,
            output_truncated INTEGER NOT NULL,
            trajectory_chars INTEGER NOT NULL,
            estimated_tokens INTEGER NOT NULL,
            assistant_turns INTEGER NOT NULL,
            base_eligible_sft INTEGER NOT NULL,
            instance_dup_count INTEGER NOT NULL DEFAULT 1,
            task_dup_count INTEGER NOT NULL DEFAULT 1,
            instance_dedup_rank INTEGER,
            task_dedup_rank INTEGER,
            eligible_sft INTEGER NOT NULL DEFAULT 0,
            flags_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL
        );
        CREATE TABLE quarantine_instances (
            instance_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            sample_occurrences INTEGER NOT NULL,
            rollout_candidate_count INTEGER NOT NULL,
            sample_ids_json TEXT NOT NULL
        );
        CREATE TABLE unresolved_rollout_coordinates (
            sample_id TEXT PRIMARY KEY,
            reason TEXT NOT NULL
        );
        CREATE TABLE sampling_cutpoints (
            snapshot_hash TEXT NOT NULL,
            source TEXT NOT NULL,
            shard_path TEXT NOT NULL,
            shard_sha256 TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            instance_id TEXT NOT NULL,
            sample_phase TEXT NOT NULL,
            cut_point INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            sampling_probability REAL NOT NULL,
            analytic_probability REAL NOT NULL,
            selected_count INTEGER NOT NULL,
            trials INTEGER NOT NULL,
            probability_method TEXT NOT NULL,
            PRIMARY KEY (shard_path, row_index, sample_phase, cut_point)
        );
        CREATE INDEX trajectories_origin_idx ON trajectories(origin, source);
        CREATE INDEX trajectories_instance_idx ON trajectories(instance_id);
        CREATE INDEX trajectories_task_idx ON trajectories(task_key);
        CREATE INDEX trajectories_quality_idx ON trajectories(eligible_sft, public_holdout);
        CREATE INDEX trajectories_model_idx ON trajectories(generator_model);
        CREATE INDEX sampling_cutpoints_sample_idx ON sampling_cutpoints(sample_id, sample_phase);
        CREATE INDEX sampling_cutpoints_instance_idx ON sampling_cutpoints(instance_id);
        """
    )


def _manifest_coordinate_maps(
    manifest: dict[str, Any], rollout_sample_counts: dict[str, int]
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, int]], list[str]]:
    wanted: dict[tuple[str, int], list[str]] = defaultdict(list)
    unresolved: list[str] = []
    for sample_id in rollout_sample_counts:
        parsed = parse_sample_id(sample_id)
        if parsed is None:
            unresolved.append(sample_id)
            continue
        shard, row, _turn = parsed
        wanted[(shard, row)].append(sample_id)

    found: dict[tuple[str, int], dict[str, Any]] = {}
    quarantine: dict[str, dict[str, int]] = defaultdict(dict)
    for source in manifest.get("sources", []):
        for shard in source.get("shards", []):
            shard_path = str(shard.get("path") or shard.get("name") or "")
            rows_meta = shard.get("rows_meta") or []
            for row_index, meta in enumerate(rows_meta):
                key = (shard_path, row_index)
                if key not in wanted:
                    continue
                value = dict(meta) if isinstance(meta, dict) else {}
                value["source"] = str(source.get("name") or _source_from_shard(shard_path))
                value["shard_sha256"] = str(shard.get("sha256") or "")
                found[key] = value
                instance_id = str(value.get("iid") or value.get("instance_id") or "")
                if instance_id:
                    for sample_id in wanted[key]:
                        quarantine[instance_id][sample_id] = rollout_sample_counts[sample_id]
    for key, sample_ids in wanted.items():
        if key not in found:
            unresolved.extend(sample_ids)
    return found, quarantine, sorted(set(unresolved))


def _rollout_sample_counts(rollout_db: Path) -> dict[str, int]:
    connection = sqlite3.connect(rollout_db)
    try:
        return {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT sample_id, COUNT(*) FROM samples GROUP BY sample_id"
            )
        }
    finally:
        connection.close()


def _insert_sampling_cutpoints(
    connection: sqlite3.Connection,
    probabilities_path: Path,
    manifest: dict[str, Any],
    snapshot_hash: str,
) -> int:
    shard_hashes = {
        str(shard.get("path") or shard.get("name") or ""): str(shard.get("sha256") or "")
        for source in manifest.get("sources", [])
        for shard in source.get("shards", [])
    }
    parquet = pq.ParquetFile(probabilities_path)
    required = {
        "snapshot_hash",
        "source",
        "shard_path",
        "row_index",
        "instance_id",
        "sample_phase",
        "cut_point",
        "sample_id",
        "sampling_probability",
        "analytic_probability",
        "selected_count",
        "trials",
        "probability_method",
    }
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"sampling-probability parquet is missing columns: {sorted(missing)}")
    count = 0
    columns = sorted(required)
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        values = []
        for row in batch.to_pylist():
            if str(row["snapshot_hash"]) != snapshot_hash:
                raise ValueError(
                    "sampling probability snapshot hash does not match dataset manifest: "
                    f"{row['snapshot_hash']} != {snapshot_hash}"
                )
            shard_path = str(row["shard_path"])
            if shard_path not in shard_hashes:
                raise ValueError(f"sampling probability references unknown shard: {shard_path}")
            probability = float(row["sampling_probability"])
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"invalid sampling probability for {row['sample_id']}: {probability}")
            values.append(
                (
                    snapshot_hash,
                    str(row["source"]),
                    shard_path,
                    shard_hashes[shard_path],
                    int(row["row_index"]),
                    str(row["instance_id"]),
                    str(row["sample_phase"]),
                    int(row["cut_point"]),
                    str(row["sample_id"]),
                    probability,
                    float(row["analytic_probability"]),
                    int(row["selected_count"]),
                    int(row["trials"]),
                    str(row["probability_method"]),
                )
            )
        connection.executemany(
            "INSERT INTO sampling_cutpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        count += len(values)
    connection.commit()
    return count


def _attach_sampling_probabilities(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        UPDATE trajectories
           SET sampling_probability = (
               SELECT SUM(c.sampling_probability)
                 FROM sampling_cutpoints c
                WHERE c.shard_path = trajectories.shard_path
                  AND c.row_index = trajectories.row_index
           )
         WHERE origin = 'original';

        UPDATE trajectories
           SET sampling_probability = (
               SELECT SUM(c.sampling_probability)
                 FROM sampling_cutpoints c
                WHERE c.shard_path = trajectories.shard_path
                  AND c.row_index = trajectories.row_index
                  AND c.cut_point = trajectories.cut_point
           ),
               sample_phase = (
               SELECT GROUP_CONCAT(c.sample_phase, '|')
                 FROM sampling_cutpoints c
                WHERE c.shard_path = trajectories.shard_path
                  AND c.row_index = trajectories.row_index
                  AND c.cut_point = trajectories.cut_point
           )
         WHERE origin = 'rollout';
        """
    )
    connection.commit()


def _row_model(row: dict[str, Any]) -> str:
    direct = row.get("generator_model") or row.get("model")
    if direct:
        return str(direct)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("model") or metadata.get("model_name") or "")
    return ""


def _row_patch(row: dict[str, Any]) -> str:
    return str(row.get("patch") or row.get("model_patch") or "")


def _success_signal(resolved: bool | None, verified: bool | None) -> bool | None:
    if resolved is True or verified is True:
        return True
    if resolved is False or verified is False:
        return False
    return None


INSERT_COLUMNS = (
    "trajectory_id",
    "origin",
    "source",
    "instance_id",
    "task_key",
    "repo",
    "language",
    "family",
    "observation_format",
    "source_resolved",
    "source_verified",
    "success_signal",
    "generator_model",
    "dataset_license",
    "repo_license",
    "provenance_uri",
    "upstream_dataset",
    "eval_run_id",
    "sample_id",
    "side",
    "snapshot_hash",
    "shard_path",
    "shard_sha256",
    "row_index",
    "sample_phase",
    "cut_point",
    "sampling_probability",
    "score",
    "scored",
    "public_holdout",
    "four_filter_pass",
    *rollout_audit.MAJOR_FLAGS,
    "patch_present",
    "patch_source_file",
    "modified_paths_json",
    "inspect_before_edit",
    "verify_after_edit",
    "workflow_complete",
    "too_long",
    "environment_error",
    "output_truncated",
    "trajectory_chars",
    "estimated_tokens",
    "assistant_turns",
    "base_eligible_sft",
    "flags_json",
    "evidence_json",
)


def _insert(connection: sqlite3.Connection, row: dict[str, Any]) -> None:
    booleans = {
        "source_resolved",
        "source_verified",
        "success_signal",
        "scored",
        "public_holdout",
        "four_filter_pass",
        *rollout_audit.MAJOR_FLAGS,
        "patch_present",
        "patch_source_file",
        "inspect_before_edit",
        "verify_after_edit",
        "workflow_complete",
        "too_long",
        "environment_error",
        "output_truncated",
        "base_eligible_sft",
    }
    values = [
        None
        if row.get(column) is None
        else int(bool(row[column]))
        if column in booleans
        else row.get(column)
        for column in INSERT_COLUMNS
    ]
    placeholders = ",".join("?" for _ in INSERT_COLUMNS)
    connection.execute(
        f"INSERT INTO trajectories ({','.join(INSERT_COLUMNS)}) VALUES ({placeholders})",
        values,
    )


def _original_rows(
    dataset_root: Path,
    manifest: dict[str, Any],
    snapshot_hash: str,
    skip_shards: set[str] | frozenset[str] = frozenset(),
) -> Iterable[tuple[str, str, str, str, int, dict[str, Any], dict[str, Any]]]:
    for source in manifest.get("sources", []):
        source_name = str(source["name"])
        for shard in source.get("shards", []):
            shard_path = str(shard.get("path") or shard.get("name") or "")
            if shard_path in skip_shards:
                continue
            shard_sha256 = str(shard.get("sha256") or "")
            path = dataset_root / shard_path
            parquet = pq.ParquetFile(path)
            available = set(parquet.schema_arrow.names)
            columns = [
                column
                for column in (
                    "instance_id",
                    "messages",
                    "first_edit",
                    "family",
                    "repo",
                    "language",
                    "resolved",
                    "verified",
                    "model",
                    "generator_model",
                    "patch",
                    "model_patch",
                    "repo_license",
                    "license",
                    "upstream_dataset",
                    "hf_dataset_name",
                    "dataset",
                    "metadata",
                )
                if column in available
            ]
            metadata_rows = shard.get("rows_meta") or []
            row_index = 0
            for batch in parquet.iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    meta = metadata_rows[row_index] if row_index < len(metadata_rows) else {}
                    yield (
                        snapshot_hash,
                        source_name,
                        shard_path,
                        shard_sha256,
                        row_index,
                        row,
                        meta,
                    )
                    row_index += 1


def _insert_originals(
    connection: sqlite3.Connection,
    dataset_root: Path,
    manifest: dict[str, Any],
    snapshot_hash: str,
    quarantine_ids: set[str],
    *,
    max_trajectory_chars: int,
    workers: int = 1,
    commit_every: int = 2_000,
    skip_shards: set[str] | frozenset[str] = frozenset(),
    raw_annotations: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> int:
    payloads = _original_rows(dataset_root, manifest, snapshot_hash, skip_shards)
    if workers <= 1:
        prepared_rows: Iterable[dict[str, Any]] = (
            _prepare_original_row(
                item, quarantine_ids, max_trajectory_chars, raw_annotations or {}
            )
            for item in payloads
        )
        count = _insert_prepared_rows(connection, prepared_rows, commit_every=commit_every)
    else:
        count = 0
        batch_size = max(64, workers * 16)
        start_method = (
            "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context(start_method),
            initializer=_init_original_worker,
            initargs=(quarantine_ids, max_trajectory_chars, raw_annotations or {}),
        ) as executor:
            while batch := list(islice(payloads, batch_size)):
                prepared_rows = executor.map(_prepare_original_worker, batch, chunksize=4)
                count = _insert_prepared_rows(
                    connection,
                    prepared_rows,
                    count=count,
                    commit_every=commit_every,
                )
    connection.commit()
    return count


def _reusable_shards(
    reuse_catalog: Path, manifest: dict[str, Any]
) -> tuple[set[str], dict[str, str]]:
    """Return shards whose old and new manifest hashes are identical."""

    old = sqlite3.connect(reuse_catalog)
    try:
        metadata = dict(old.execute("SELECT key, value FROM metadata"))
    finally:
        old.close()
    old_root = Path(str(metadata.get("dataset_root") or "")).expanduser()
    old_manifest_path = old_root / "manifest.json"
    if not old_manifest_path.is_file():
        return set(), {}
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    old_hashes = {
        str(shard.get("path") or shard.get("name") or ""): str(shard.get("sha256") or "")
        for source in old_manifest.get("sources", [])
        for shard in source.get("shards", [])
    }
    new_hashes = {
        str(shard.get("path") or shard.get("name") or ""): str(shard.get("sha256") or "")
        for source in manifest.get("sources", [])
        for shard in source.get("shards", [])
    }
    reusable = {
        path for path, digest in new_hashes.items() if digest and old_hashes.get(path) == digest
    }
    return reusable, new_hashes


def _raw_annotations(
    raw_root: Path, manifest: dict[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Recover fields intentionally omitted by the production-v1 rendered schema.

    Selection follows the renderer's repository order, filters, deduplication, and minimum-turn
    checks.  Only instance ids present in the pinned manifest are retained.
    """

    wanted_by_source = {
        str(source["name"]): {
            str(meta.get("iid") or meta.get("instance_id") or "")
            for shard in source.get("shards", [])
            for meta in shard.get("rows_meta", [])
        }
        for source in manifest.get("sources", [])
    }
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    for source, spec in SOURCES.items():
        if not spec.get("render"):
            continue
        wanted = wanted_by_source.get(source, set())
        seen_repos: Counter[str] = Counter()
        seen_ids: set[str] = set()
        for shard in renderer._raw_shards(raw_root, spec):
            parquet = pq.ParquetFile(shard)
            available = set(parquet.schema_arrow.names)
            columns = [
                column
                for column in (
                    "instance_id",
                    "repo",
                    "hf_dataset_name",
                    "dataset",
                    "language",
                    "license",
                    "resolved",
                    "verified",
                    "model",
                    "generator_model",
                    "patch",
                    "model_patch",
                    "metadata",
                )
                if column in available
            ]
            turns_col = next(
                column
                for column in ("messages", "trajectory", "conversation")
                if column in available
            )
            for batch in parquet.iter_batches(batch_size=64, columns=columns + [turns_col]):
                for row in batch.to_pylist():
                    instance_id = str(row.get("instance_id") or "")
                    if not instance_id or instance_id in seen_ids:
                        continue
                    reason = renderer._keep(row, instance_id, spec, seen_repos)
                    if reason:
                        continue
                    if instance_id not in wanted:
                        # It was not retained in the pinned render; do not let an upstream-only row
                        # influence repo caps or annotation provenance.
                        continue
                    messages, _first_edit = renderer.render_turns(row.get(turns_col) or [])
                    assistant = sum(1 for message in messages if message["role"] == "assistant")
                    if assistant < 2 or any(not message["content"] for message in messages):
                        continue
                    repo = renderer._repo_of(row, instance_id)
                    seen_repos[repo] += 1
                    seen_ids.add(instance_id)
                    annotations[(source, instance_id)] = {
                        "resolved": renderer._optional_bool(row.get("resolved")),
                        "verified": renderer._optional_bool(row.get("verified")),
                        "generator_model": renderer._generator_model(row),
                        "patch": renderer._patch(row),
                        "repo_license": str(row.get("license") or ""),
                        "upstream_dataset": str(
                            row.get("hf_dataset_name")
                            or row.get("dataset")
                            or renderer._upstream_repo(shard, raw_root, spec)
                        ),
                    }
        missing = wanted - seen_ids
        if missing:
            raise ValueError(
                f"raw annotation recovery could not reproduce {len(missing)} {source} rows; "
                f"examples: {sorted(missing)[:3]}"
            )
    return annotations


def _insert_reused_originals(
    connection: sqlite3.Connection,
    reuse_catalog: Path,
    reusable_shards: set[str],
    shard_hashes: dict[str, str],
    snapshot_hash: str,
) -> int:
    if not reusable_shards:
        return 0
    connection.execute("CREATE TEMP TABLE reusable_shards(path TEXT PRIMARY KEY, sha256 TEXT)")
    connection.executemany(
        "INSERT INTO reusable_shards VALUES (?, ?)",
        ((path, shard_hashes[path]) for path in sorted(reusable_shards)),
    )
    connection.execute("ATTACH DATABASE ? AS reuse", (str(reuse_catalog),))
    try:
        old_columns = {
            str(row[1]) for row in connection.execute("PRAGMA reuse.table_info(trajectories)")
        }
        selections: list[str] = []
        for column in INSERT_COLUMNS:
            if column == "snapshot_hash":
                selections.append("?")
            elif column == "shard_sha256":
                selections.append("s.sha256")
            elif column in {"sample_phase", "cut_point", "sampling_probability"}:
                selections.append("NULL")
            elif column == "public_holdout":
                selections.append(
                    "EXISTS (SELECT 1 FROM main.quarantine_instances q "
                    "WHERE q.instance_id = r.instance_id)"
                )
            elif column == "base_eligible_sft":
                selections.append(
                    "(r.success_signal = 1 AND r.four_filter_pass = 1 "
                    "AND r.patch_present = 1 AND r.patch_source_file = 1 "
                    "AND r.workflow_complete = 1 AND r.too_long = 0 "
                    "AND r.environment_error = 0 AND r.output_truncated = 0)"
                )
            elif column not in old_columns:
                selections.append("NULL")
            else:
                selections.append(f"r.{column}")
        placeholders = ",".join(selections)
        connection.execute(
            f"INSERT INTO trajectories ({','.join(INSERT_COLUMNS)}) "
            f"SELECT {placeholders} FROM reuse.trajectories r "
            "JOIN reusable_shards s ON s.path = r.shard_path "
            "WHERE r.origin = 'original'",
            (snapshot_hash,),
        )
        count = int(connection.execute("SELECT changes()").fetchone()[0])
        connection.commit()
    finally:
        connection.execute("DETACH DATABASE reuse")
    return count


_WORKER_QUARANTINE_IDS: frozenset[str] = frozenset()
_WORKER_MAX_TRAJECTORY_CHARS = 120_000
_WORKER_RAW_ANNOTATIONS: dict[tuple[str, str], dict[str, Any]] = {}


def _init_original_worker(
    quarantine_ids: set[str],
    max_trajectory_chars: int,
    raw_annotations: dict[tuple[str, str], dict[str, Any]],
) -> None:
    global _WORKER_QUARANTINE_IDS, _WORKER_MAX_TRAJECTORY_CHARS, _WORKER_RAW_ANNOTATIONS
    _WORKER_QUARANTINE_IDS = frozenset(quarantine_ids)
    _WORKER_MAX_TRAJECTORY_CHARS = max_trajectory_chars
    _WORKER_RAW_ANNOTATIONS = raw_annotations


def _prepare_original_worker(
    item: tuple[str, str, str, str, int, dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    return _prepare_original_row(
        item,
        _WORKER_QUARANTINE_IDS,
        _WORKER_MAX_TRAJECTORY_CHARS,
        _WORKER_RAW_ANNOTATIONS,
    )


def _prepare_original_row(
    item: tuple[str, str, str, str, int, dict[str, Any], dict[str, Any]],
    quarantine_ids: set[str] | frozenset[str],
    max_trajectory_chars: int,
    raw_annotations: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    snapshot_hash, source, shard_path, shard_sha256, row_index, raw, meta_value = item
    meta = meta_value if isinstance(meta_value, dict) else {}
    messages = _messages(raw.get("messages"))
    instance_id = str(raw.get("instance_id") or meta.get("iid") or "")
    annotation = (raw_annotations or {}).get((source, instance_id), {})
    repo = str(raw.get("repo") or meta.get("repo") or instance_id.split(".", 1)[0])
    language = str(raw.get("language") or meta.get("language") or "unknown")
    family = str(raw.get("family") or meta.get("family") or "unknown")
    resolved_value = raw.get("resolved")
    if resolved_value is None:
        resolved_value = annotation.get("resolved")
    resolved = _optional_bool(resolved_value)
    verified_value = raw.get("verified")
    if verified_value is None:
        verified_value = annotation.get("verified", meta.get("verified"))
    verified = _optional_bool(
        verified_value
    )
    success = _success_signal(resolved, verified)
    patch = _row_patch(raw) or str(annotation.get("patch") or "")
    audit = audit_trajectory(
        messages,
        patch=patch,
        max_trajectory_chars=max_trajectory_chars,
    )
    public_holdout = instance_id in quarantine_ids
    base_eligible = bool(
        success is True
        and audit["four_filter_pass"]
        and audit["patch_present"]
        and audit["patch_source_file"]
        and audit["workflow_complete"]
        and not audit["too_long"]
        and not audit["environment_error"]
        and not audit["output_truncated"]
    )
    upstream_dataset = str(
        raw.get("upstream_dataset")
        or raw.get("hf_dataset_name")
        or raw.get("dataset")
        or annotation.get("upstream_dataset")
        or SOURCES[source]["repos"][0]
    )
    provenance_dataset = (
        upstream_dataset
        if upstream_dataset in SOURCES[source]["repos"]
        else SOURCES[source]["repos"][0]
    )
    return {
        "trajectory_id": f"original:{shard_path}:{row_index}",
        "origin": "original",
        "source": source,
        "instance_id": instance_id,
        "task_key": task_key(repo, messages),
        "repo": repo,
        "language": language,
        "family": family,
        "observation_format": detect_format(f"{shard_path}:{row_index}:0", messages),
        "source_resolved": resolved,
        "source_verified": verified,
        "success_signal": success,
        "generator_model": _row_model(raw) or str(annotation.get("generator_model") or ""),
        "dataset_license": SOURCE_LICENSES.get(source, "UNKNOWN"),
        "repo_license": str(
            raw.get("repo_license")
            or raw.get("license")
            or annotation.get("repo_license")
            or ""
        ),
        "provenance_uri": f"hf://{provenance_dataset}",
        "upstream_dataset": upstream_dataset,
        "eval_run_id": None,
        "sample_id": f"{shard_path}:{row_index}",
        "side": None,
        "snapshot_hash": snapshot_hash,
        "shard_path": shard_path,
        "shard_sha256": shard_sha256,
        "row_index": row_index,
        "sample_phase": None,
        "cut_point": None,
        "sampling_probability": None,
        "score": None,
        "scored": False,
        "public_holdout": public_holdout,
        **{
            key: audit[key]
            for key in (
                "four_filter_pass",
                *rollout_audit.MAJOR_FLAGS,
                "patch_present",
                "patch_source_file",
                "inspect_before_edit",
                "verify_after_edit",
                "workflow_complete",
                "too_long",
                "environment_error",
                "output_truncated",
                "trajectory_chars",
                "estimated_tokens",
                "assistant_turns",
            )
        },
        "modified_paths_json": _json(audit["modified_paths"]),
        "base_eligible_sft": base_eligible,
        "flags_json": _json(audit["extended_flags"]),
        "evidence_json": _json(audit["evidence"]),
    }


def _insert_prepared_rows(
    connection: sqlite3.Connection,
    rows: Iterable[dict[str, Any]],
    *,
    count: int = 0,
    commit_every: int,
) -> int:
    for row in rows:
        _insert(connection, row)
        count += 1
        if count % commit_every == 0:
            connection.commit()
            print(f"original trajectories indexed: {count}", flush=True)
    return count


def _audit_rows(audit_db: Path) -> dict[tuple[str, str, str], sqlite3.Row]:
    connection = sqlite3.connect(audit_db)
    connection.row_factory = sqlite3.Row
    try:
        return {
            (str(row["eval_run_id"]), str(row["sample_id"]), str(row["side"])): row
            for row in connection.execute("SELECT * FROM candidates")
        }
    finally:
        connection.close()


def _insert_rollouts(
    connection: sqlite3.Connection,
    rollout_root: Path,
    coordinate_meta: dict[tuple[str, int], dict[str, Any]],
    snapshot_hash: str,
    *,
    max_trajectory_chars: int,
) -> int:
    rollout_db = rollout_root / "index" / "rollouts.sqlite3"
    audit = _audit_rows(rollout_root / "audit" / "audit.sqlite3")
    source_connection = sqlite3.connect(rollout_db)
    source_connection.row_factory = sqlite3.Row
    count = 0
    try:
        query = """
            SELECT r.eval_run_id, a.relative_path
            FROM runs r JOIN artifacts a ON a.eval_run_id = r.eval_run_id
            WHERE a.artifact_type = 'GENERATED_SAMPLES'
            ORDER BY r.finished_at, r.eval_run_id
        """
        for run in source_connection.execute(query):
            generated_path = rollout_root / str(run["relative_path"])
            with generated_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    sample_id = str(sample.get("sample_id") or "")
                    parsed = parse_sample_id(sample_id)
                    shard_path, row_index, cut_point = parsed if parsed else ("", -1, -1)
                    meta = coordinate_meta.get((shard_path, row_index), {})
                    source = str(meta.get("source") or _source_from_shard(shard_path))
                    instance_id = str(meta.get("iid") or meta.get("instance_id") or "")
                    repo = str(meta.get("repo") or instance_id.split(".", 1)[0])
                    for side in ("previous_king", "challenger"):
                        is_challenger = side == "challenger"
                        turns_key = "challenger_turns" if is_challenger else "previous_king_turns"
                        messages = _messages(sample.get(turns_key))
                        document = side_output(sample, side)
                        explicit_error = str(
                            sample.get("chal_error" if is_challenger else "king_error") or ""
                        )
                        full_audit = audit_trajectory(
                            messages,
                            supplemental_text=document,
                            explicit_error=explicit_error,
                            max_trajectory_chars=max_trajectory_chars,
                        )
                        indexed = audit.get((str(run["eval_run_id"]), sample_id, side))
                        if indexed is not None:
                            for flag in rollout_audit.MAJOR_FLAGS:
                                full_audit[flag] = bool(indexed[flag])
                            full_audit["four_filter_pass"] = bool(indexed["quality_pass"])
                        score = _as_float(indexed["score"] if indexed is not None else None)
                        model = str(indexed["model_uri"] if indexed is not None else "")
                        extended_flags = [
                            flag for flag in rollout_audit.MAJOR_FLAGS if full_audit[flag]
                        ]
                        extended_flags.extend(
                            flag
                            for flag in full_audit["extended_flags"]
                            if flag not in rollout_audit.MAJOR_FLAGS
                        )
                        row = {
                            "trajectory_id": f"rollout:{run['eval_run_id']}:{sample_id}:{side}",
                            "origin": "rollout",
                            "source": source,
                            "instance_id": instance_id,
                            "task_key": task_key(repo, messages),
                            "repo": repo,
                            "language": str(meta.get("language") or "unknown"),
                            "family": str(meta.get("family") or "unknown"),
                            "observation_format": detect_format(sample_id, messages),
                            "source_resolved": _optional_bool(meta.get("resolved")),
                            "source_verified": _optional_bool(meta.get("verified")),
                            "success_signal": bool(score is not None and score >= 0.65),
                            "generator_model": model,
                            "dataset_license": SOURCE_LICENSES.get(source, "UNKNOWN"),
                            "repo_license": "",
                            "provenance_uri": "https://albedo.tech/data/dashboard.json",
                            "upstream_dataset": SOURCES.get(source, {}).get("repos", [""])[0],
                            "eval_run_id": str(run["eval_run_id"]),
                            "sample_id": sample_id,
                            "side": side,
                            "snapshot_hash": snapshot_hash,
                            "shard_path": shard_path,
                            "shard_sha256": str(meta.get("shard_sha256") or ""),
                            "row_index": row_index,
                            "sample_phase": (
                                str(indexed["sample_phase"] or "")
                                if indexed is not None and "sample_phase" in indexed.keys()
                                else None
                            ),
                            "cut_point": cut_point if cut_point >= 0 else None,
                            "sampling_probability": None,
                            "score": score,
                            "scored": bool(indexed is not None and indexed["scored"]),
                            "public_holdout": True,
                            **{
                                key: full_audit[key]
                                for key in (
                                    "four_filter_pass",
                                    *rollout_audit.MAJOR_FLAGS,
                                    "patch_present",
                                    "patch_source_file",
                                    "inspect_before_edit",
                                    "verify_after_edit",
                                    "workflow_complete",
                                    "too_long",
                                    "environment_error",
                                    "output_truncated",
                                    "trajectory_chars",
                                    "estimated_tokens",
                                    "assistant_turns",
                                )
                            },
                            "modified_paths_json": _json(full_audit["modified_paths"]),
                            "base_eligible_sft": False,
                            "flags_json": _json(extended_flags),
                            "evidence_json": _json(full_audit["evidence"]),
                        }
                        _insert(connection, row)
                        count += 1
        connection.commit()
    finally:
        source_connection.close()
    return count


def _deduplicate(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TEMP TABLE instance_counts AS
          SELECT instance_id, COUNT(*) AS n
          FROM trajectories WHERE instance_id IS NOT NULL AND instance_id != ''
          GROUP BY instance_id;
        CREATE UNIQUE INDEX instance_counts_id_idx ON instance_counts(instance_id);
        CREATE TEMP TABLE task_counts AS
          SELECT task_key, COUNT(*) AS n FROM trajectories GROUP BY task_key;
        CREATE UNIQUE INDEX task_counts_key_idx ON task_counts(task_key);
        UPDATE trajectories
          SET instance_dup_count = COALESCE(
                (SELECT n FROM instance_counts c WHERE c.instance_id = trajectories.instance_id), 1
              ),
              task_dup_count = COALESCE(
                (SELECT n FROM task_counts c WHERE c.task_key = trajectories.task_key), 1
              );
        CREATE TEMP TABLE instance_ranks AS
          SELECT trajectory_id,
                 ROW_NUMBER() OVER (
                   PARTITION BY CASE WHEN instance_id IS NULL OR instance_id = ''
                                     THEN trajectory_id ELSE instance_id END
                   ORDER BY base_eligible_sft DESC, public_holdout ASC,
                            success_signal DESC, four_filter_pass DESC,
                            workflow_complete DESC, trajectory_chars ASC, trajectory_id
                 ) AS rank
          FROM trajectories;
        CREATE UNIQUE INDEX instance_ranks_id_idx ON instance_ranks(trajectory_id);
        UPDATE trajectories
          SET instance_dedup_rank = (
            SELECT rank FROM instance_ranks r WHERE r.trajectory_id = trajectories.trajectory_id
          );
        CREATE TEMP TABLE task_ranks AS
          SELECT trajectory_id,
                 ROW_NUMBER() OVER (
                   PARTITION BY task_key
                   ORDER BY base_eligible_sft DESC, public_holdout ASC,
                            instance_dedup_rank ASC, success_signal DESC,
                            four_filter_pass DESC, workflow_complete DESC,
                            trajectory_chars ASC, trajectory_id
                 ) AS rank
          FROM trajectories;
        CREATE UNIQUE INDEX task_ranks_id_idx ON task_ranks(trajectory_id);
        UPDATE trajectories
          SET task_dedup_rank = (
            SELECT rank FROM task_ranks r WHERE r.trajectory_id = trajectories.trajectory_id
          );
        UPDATE trajectories
          SET eligible_sft = CASE
            WHEN base_eligible_sft = 1
             AND instance_dedup_rank = 1
             AND task_dedup_rank = 1 THEN 1 ELSE 0 END;
        DROP TABLE instance_counts;
        DROP TABLE task_counts;
        DROP TABLE instance_ranks;
        DROP TABLE task_ranks;
        CREATE VIEW original_trajectories AS
          SELECT * FROM trajectories WHERE origin = 'original';
        CREATE VIEW rollout_trajectories AS
          SELECT * FROM trajectories WHERE origin = 'rollout';
        CREATE VIEW held_out_trajectories AS
          SELECT * FROM trajectories WHERE public_holdout = 1;
        CREATE VIEW clean_four_filter AS
          SELECT * FROM trajectories WHERE four_filter_pass = 1;
        CREATE VIEW eligible_sft_trajectories AS
          SELECT * FROM trajectories WHERE eligible_sft = 1;
        CREATE VIEW eligible_original_cutpoints AS
          SELECT t.trajectory_id, t.source, t.instance_id, t.task_key, t.repo, t.language,
                 t.family, t.observation_format, t.generator_model, t.dataset_license,
                 t.repo_license, t.provenance_uri, t.upstream_dataset, t.public_holdout,
                 t.shard_path, t.shard_sha256, t.row_index,
                 c.snapshot_hash, c.sample_phase, c.cut_point, c.sample_id,
                 c.sampling_probability, c.analytic_probability, c.selected_count,
                 c.trials, c.probability_method
            FROM eligible_sft_trajectories t
            JOIN sampling_cutpoints c
              ON c.shard_path = t.shard_path AND c.row_index = t.row_index
           WHERE t.origin = 'original';
        """
    )
    connection.commit()


def _summary(
    connection: sqlite3.Connection,
    *,
    errors: list[str],
    paths: dict[str, str],
) -> dict[str, Any]:
    connection.row_factory = sqlite3.Row

    def scalar(query: str) -> int:
        return int(connection.execute(query).fetchone()[0])

    by_origin = {
        str(row["origin"]): {
            "count": int(row["n"]),
            "four_filter_pass": int(row["clean"]),
            "held_out": int(row["held"]),
            "base_eligible_sft": int(row["base_sft"]),
            "eligible_sft_after_dedup": int(row["sft"]),
        }
        for row in connection.execute(
            """
            SELECT origin, COUNT(*) n, SUM(four_filter_pass) clean,
                   SUM(public_holdout) held, SUM(base_eligible_sft) base_sft,
                   SUM(eligible_sft) sft
            FROM trajectories GROUP BY origin ORDER BY origin
            """
        )
    }
    by_source = {
        str(row["source"]): {
            "count": int(row["n"]),
            "successful": int(row["successful"] or 0),
            "four_filter_pass": int(row["clean"]),
            "patch_present": int(row["patch"]),
            "workflow_complete": int(row["workflow"]),
            "held_out": int(row["held"]),
            "base_eligible_sft": int(row["base_sft"]),
            "eligible_sft_after_dedup": int(row["sft"]),
        }
        for row in connection.execute(
            """
            SELECT source, COUNT(*) n,
                   SUM(CASE WHEN success_signal = 1 THEN 1 ELSE 0 END) successful,
                   SUM(four_filter_pass) clean, SUM(patch_present) patch,
                   SUM(workflow_complete) workflow, SUM(public_holdout) held,
                   SUM(base_eligible_sft) base_sft, SUM(eligible_sft) sft
            FROM trajectories GROUP BY source ORDER BY source
            """
        )
    }
    flag_counts = {
        flag: scalar(f"SELECT SUM({flag}) FROM trajectories")
        for flag in (
            *rollout_audit.MAJOR_FLAGS,
            "patch_present",
            "patch_source_file",
            "workflow_complete",
            "too_long",
            "environment_error",
            "output_truncated",
        )
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        **paths,
        "trajectory_count": scalar("SELECT COUNT(*) FROM trajectories"),
        "original_count": scalar("SELECT COUNT(*) FROM original_trajectories"),
        "rollout_candidate_count": scalar("SELECT COUNT(*) FROM rollout_trajectories"),
        "unique_instance_count": scalar(
            "SELECT COUNT(DISTINCT instance_id) FROM trajectories WHERE instance_id != ''"
        ),
        "quarantine_instance_count": scalar("SELECT COUNT(*) FROM quarantine_instances"),
        "quarantine_sample_occurrence_count": scalar(
            "SELECT COALESCE(SUM(sample_occurrences), 0) FROM quarantine_instances"
        ),
        "quarantine_rollout_candidate_count": scalar(
            "SELECT COALESCE(SUM(rollout_candidate_count), 0) FROM quarantine_instances"
        ),
        "unresolved_rollout_coordinate_count": scalar(
            "SELECT COUNT(*) FROM unresolved_rollout_coordinates"
        ),
        "base_eligible_sft_count": scalar("SELECT SUM(base_eligible_sft) FROM trajectories"),
        "eligible_sft_count": scalar("SELECT COUNT(*) FROM eligible_sft_trajectories"),
        "instance_duplicate_row_count": scalar(
            "SELECT SUM(instance_dup_count > 1) FROM trajectories"
        ),
        "task_duplicate_row_count": scalar("SELECT SUM(task_dup_count > 1) FROM trajectories"),
        "by_origin": by_origin,
        "by_source": by_source,
        "feature_counts": flag_counts,
        "errors": errors,
    }


def build_catalog(
    dataset_root: Path,
    rollout_root: Path,
    output: Path,
    *,
    max_trajectory_chars: int = 120_000,
    workers: int = 1,
    sampling_probabilities: Path | None = None,
    reuse_catalog: Path | None = None,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    rollout_db = rollout_root / "index" / "rollouts.sqlite3"
    for required in (manifest_path, rollout_db, rollout_root / "audit" / "audit.sqlite3"):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest_payload = manifest_path.read_bytes()
    snapshot_hash = hashlib.sha256(manifest_payload).hexdigest()
    manifest = json.loads(manifest_payload)
    provenance_path = dataset_root / "snapshot-provenance.json"
    snapshot_provenance: dict[str, Any] = {}
    if provenance_path.is_file():
        snapshot_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    verification_level = str(snapshot_provenance.get("verification_level") or "unreported")
    if sampling_probabilities is not None and not sampling_probabilities.is_file():
        raise FileNotFoundError(sampling_probabilities)
    if reuse_catalog is not None and not reuse_catalog.is_file():
        raise FileNotFoundError(reuse_catalog)
    if raw_root is not None and not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    rollout_sample_counts = _rollout_sample_counts(rollout_db)
    coordinate_meta, quarantine, unresolved = _manifest_coordinate_maps(
        manifest, rollout_sample_counts
    )

    output.mkdir(parents=True, exist_ok=True)
    destination = output / "catalog.sqlite3"
    temporary = output / f".catalog.{os.getpid()}.tmp.sqlite3"
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    errors = [f"unresolved rollout coordinate: {sample_id}" for sample_id in unresolved]
    try:
        _create_schema(connection)
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            (
                ("schema_version", str(SCHEMA_VERSION)),
                ("generated_at", _utc_now()),
                ("dataset_root", str(dataset_root)),
                ("snapshot_hash", snapshot_hash),
                ("snapshot_verification_level", verification_level),
                ("snapshot_provenance", _json(snapshot_provenance)),
                ("rollout_root", str(rollout_root)),
                ("max_trajectory_chars", str(max_trajectory_chars)),
                (
                    "sampling_probabilities",
                    str(sampling_probabilities) if sampling_probabilities is not None else "",
                ),
                ("reuse_catalog", str(reuse_catalog) if reuse_catalog is not None else ""),
                ("raw_root", str(raw_root) if raw_root is not None else ""),
            ),
        )
        sampling_cutpoint_count = (
            _insert_sampling_cutpoints(
                connection, sampling_probabilities, manifest, snapshot_hash
            )
            if sampling_probabilities is not None
            else 0
        )
        connection.executemany(
            "INSERT INTO unresolved_rollout_coordinates VALUES (?, ?)",
            ((sample_id, "not found in local manifest") for sample_id in unresolved),
        )
        for instance_id, sample_counts in sorted(quarantine.items()):
            sample_ids = sorted(sample_counts)
            sources = sorted(
                {_source_from_shard(parse_sample_id(sample_id)[0]) for sample_id in sample_ids}
            )
            occurrences = sum(sample_counts.values())
            connection.execute(
                "INSERT INTO quarantine_instances VALUES (?, ?, ?, ?, ?)",
                (
                    instance_id,
                    ",".join(sources),
                    occurrences,
                    occurrences * 2,
                    _json(sample_ids),
                ),
            )
        connection.commit()
        reusable_shards: set[str] = set()
        reused_original_count = 0
        if reuse_catalog is not None:
            reusable_shards, new_shard_hashes = _reusable_shards(reuse_catalog, manifest)
            reused_original_count = _insert_reused_originals(
                connection,
                reuse_catalog,
                reusable_shards,
                new_shard_hashes,
                snapshot_hash,
            )
            print(
                f"reused original audit rows: {reused_original_count} "
                f"across {len(reusable_shards)} byte-identical shards",
                flush=True,
            )
        annotations = _raw_annotations(raw_root, manifest) if raw_root is not None else {}
        if annotations:
            print(f"recovered raw annotations: {len(annotations)}", flush=True)
        indexed_original_count = _insert_originals(
            connection,
            dataset_root,
            manifest,
            snapshot_hash,
            set(quarantine),
            max_trajectory_chars=max_trajectory_chars,
            workers=workers,
            skip_shards=reusable_shards,
            raw_annotations=annotations,
        )
        original_count = reused_original_count + indexed_original_count
        print(f"original indexing complete: {original_count}", flush=True)
        rollout_count = _insert_rollouts(
            connection,
            rollout_root,
            coordinate_meta,
            snapshot_hash,
            max_trajectory_chars=max_trajectory_chars,
        )
        print(f"rollout indexing complete: {rollout_count}", flush=True)
        _attach_sampling_probabilities(connection)
        _deduplicate(connection)
        summary = _summary(
            connection,
            errors=errors,
            paths={
                "dataset_root": str(dataset_root),
                "rollout_root": str(rollout_root),
                "database": str(destination),
            },
        )
        summary["snapshot_hash"] = snapshot_hash
        summary["snapshot_verification_level"] = verification_level
        summary["sampling_cutpoint_count"] = sampling_cutpoint_count
        summary["reused_original_count"] = reused_original_count
        summary["reused_shard_count"] = len(reusable_shards)
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        summary["sqlite_integrity"] = integrity
    finally:
        connection.close()
    os.replace(temporary, destination)

    export = sqlite3.connect(destination)
    export.row_factory = sqlite3.Row
    try:
        quarantine_lines = [
            _json(dict(row))
            for row in export.execute("SELECT * FROM quarantine_instances ORDER BY instance_id")
        ]
        sft_lines = [
            _json(dict(row))
            for row in export.execute(
                """
                SELECT trajectory_id, source, instance_id, repo, language, family,
                       observation_format, generator_model, dataset_license,
                       shard_path, row_index, estimated_tokens
                FROM eligible_sft_trajectories
                ORDER BY source, shard_path, row_index
                """
            )
        ]
    finally:
        export.close()
    _write_atomic(output / "quarantine-instances.jsonl", "\n".join(quarantine_lines) + "\n")
    _write_atomic(output / "sft-candidates.jsonl", "\n".join(sft_lines) + "\n")
    _write_atomic(output / "summary.json", _json(summary, pretty=True))
    _write_atomic(
        output / "provenance.json",
        _json(
            {
                source: {
                    "license": SOURCE_LICENSES[source],
                    "repositories": SOURCES[source]["repos"],
                }
                for source in SOURCES
            },
            pretty=True,
        ),
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-trajectory-chars", type=int, default=120_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--sampling-probabilities",
        type=Path,
        default=None,
        help="Parquet produced by estimate_sampling_probabilities.py for this manifest.",
    )
    parser.add_argument(
        "--reuse-catalog",
        type=Path,
        help="Reuse deterministic audit rows from byte-identical shards in an older catalog.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Pinned raw rendered-source root used to recover omitted labels/patch provenance.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_catalog(
        args.dataset_root.expanduser().resolve(),
        args.rollout_root.expanduser().resolve(),
        args.output.expanduser().resolve(),
        max_trajectory_chars=args.max_trajectory_chars,
        workers=max(1, args.workers),
        sampling_probabilities=(
            args.sampling_probabilities.expanduser().resolve()
            if args.sampling_probabilities is not None
            else None
        ),
        reuse_catalog=(
            args.reuse_catalog.expanduser().resolve() if args.reuse_catalog is not None else None
        ),
        raw_root=(args.raw_root.expanduser().resolve() if args.raw_root is not None else None),
    )
    print(_json(summary, pretty=True), end="")
    return 1 if args.strict and summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
