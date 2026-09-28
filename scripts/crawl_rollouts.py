#!/usr/bin/env python3
"""Download and index public Albedo rollout artifacts.

The crawler is intentionally standard-library-only so it can run in a fresh checkout on a
training server before the project's large ML dependencies are installed.  It keeps immutable
dashboard snapshots, resumes existing downloads, validates artifacts against hashes published in
``verdict.json``, and rebuilds a compact SQLite sample index without copying trajectory text into
the database.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from generated_sample_fields import side_output  # noqa: E402

DEFAULT_DASHBOARD_URL = "https://albedo.tech/data/dashboard.json"
DEFAULT_KING_VERSIONS = (121, 122)
DEFAULT_TOP_CHALLENGERS = 20
SCHEMA_VERSION = 1

ARTIFACT_FILENAMES = {
    "GENERATED_SAMPLES": "generated-samples.jsonl",
    "REMOTE_PROGRESS": "progress.jsonl",
    "REMOTE_LOGS": "remote-logs.txt",
    "SCORING_RESULTS": "scoring-results.jsonl",
    "EVAL_VERDICT": "verdict.json",
    "REQUEST": "request.json",
}

_PRINT_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_text(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(path, _json_text(value, pretty=True))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _roman(number: int) -> str:
    values = (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )
    if number <= 0:
        return str(number)
    result: list[str] = []
    remaining = number
    for value, symbol in values:
        while remaining >= value:
            result.append(symbol)
            remaining -= value
    return "".join(result)


def _safe_slug(text: str, *, fallback: str = "unknown", limit: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").casefold()).strip("-")
    return (slug or fallback)[:limit].rstrip("-")


def _revision_from_uri(model_uri: str) -> str:
    return model_uri.rsplit("@", 1)[1] if "@" in model_uri else ""


def _model_slug(model_uri: str, uid: Any, target_versions: Iterable[int] = ()) -> str:
    versions = sorted(set(target_versions))
    if len(versions) == 1:
        return f"albedo-{_roman(versions[0]).casefold()}"
    repository = model_uri.split("@", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    revision = _revision_from_uri(model_uri)[:10]
    identity = f"uid-{uid}" if uid is not None else "challenger"
    suffix = f"-{revision}" if revision else ""
    return _safe_slug(f"{identity}-{repository}{suffix}")


def _dashboard_runs(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    runs = dashboard.get("eval_runs")
    if not isinstance(runs, list):
        raise ValueError("dashboard does not contain an eval_runs list")
    return [run for run in runs if isinstance(run, dict) and run.get("eval_run_id")]


def target_models_by_version(
    dashboard: dict[str, Any], versions: Iterable[int]
) -> dict[int, str]:
    wanted = set(versions)
    found: dict[int, str] = {}
    reign = dashboard.get("reign")
    members = reign.get("members", []) if isinstance(reign, dict) else []
    for member in members if isinstance(members, list) else []:
        if not isinstance(member, dict):
            continue
        version = _as_int(member.get("king_version"))
        uri = str(member.get("model_uri") or "")
        if version in wanted and uri:
            found[version] = uri
    for run in _dashboard_runs(dashboard):
        version = _as_int(run.get("king_version"))
        uri = str(run.get("model_uri") or "")
        if version in wanted and uri and bool(run.get("coronated")):
            found[version] = uri
    return found


def select_runs(
    dashboard: dict[str, Any],
    *,
    king_versions: Iterable[int] = DEFAULT_KING_VERSIONS,
    top_challengers: int = DEFAULT_TOP_CHALLENGERS,
    min_challenger_score: float | None = None,
    min_win_margin: float | None = None,
    all_runs: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Select target-king runs plus strong challenger runs from one dashboard snapshot."""

    versions = tuple(sorted(set(king_versions)))
    targets = target_models_by_version(dashboard, versions)
    target_versions_by_uri: dict[str, list[int]] = defaultdict(list)
    for version, uri in targets.items():
        target_versions_by_uri[uri].append(version)

    runs = _dashboard_runs(dashboard)
    reasons: dict[str, set[str]] = defaultdict(set)
    records: dict[str, dict[str, Any]] = {}
    for run in runs:
        run_id = str(run["eval_run_id"])
        records[run_id] = run
        if all_runs and isinstance(run.get("artifacts"), dict) and run.get("artifacts"):
            reasons[run_id].add("all_visible_runs")
        uri = str(run.get("model_uri") or "")
        for version in target_versions_by_uri.get(uri, []):
            reasons[run_id].add(f"king_version:{version}")

    ranked = [
        run
        for run in runs
        if str(run.get("model_uri") or "") not in target_versions_by_uri
        and _as_float(run.get("score_challenger")) is not None
        and isinstance(run.get("artifacts"), dict)
        and bool(run.get("artifacts"))
    ]
    ranked.sort(
        key=lambda run: (
            -float(run["score_challenger"]),
            str(run.get("finished_at") or ""),
            str(run["eval_run_id"]),
        )
    )
    for rank, run in enumerate(ranked[: max(0, top_challengers)], start=1):
        reasons[str(run["eval_run_id"])].add(f"top_challenger:{rank}")

    for run in ranked:
        run_id = str(run["eval_run_id"])
        score = _as_float(run.get("score_challenger"))
        margin = _as_float(run.get("win_margin"))
        if min_challenger_score is not None and score is not None:
            if score >= min_challenger_score:
                reasons[run_id].add(f"min_challenger_score:{min_challenger_score:g}")
        if min_win_margin is not None and margin is not None and margin >= min_win_margin:
            reasons[run_id].add(f"min_win_margin:{min_win_margin:g}")

    selected: list[dict[str, Any]] = []
    for run_id, run_reasons in reasons.items():
        run = dict(records[run_id])
        uri = str(run.get("model_uri") or "")
        target_versions = target_versions_by_uri.get(uri, [])
        run["_selection_reasons"] = sorted(run_reasons)
        run["_target_versions"] = sorted(target_versions)
        run["_model_slug"] = _model_slug(uri, run.get("uid"), target_versions)
        selected.append(run)
    selected.sort(
        key=lambda run: (
            0 if run["_target_versions"] else 1,
            run["_target_versions"] or [999999],
            -(_as_float(run.get("score_challenger")) or -1.0),
            str(run["eval_run_id"]),
        )
    )
    return selected, targets


def _request(
    url: str, *, timeout: float, headers: dict[str, str] | None = None
) -> urllib.response.addinfourl:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme for {url!r}")
    request_headers = {"User-Agent": "albedo-rollout-crawler/1", **(headers or {})}
    request = urllib.request.Request(url, headers=request_headers)
    return urllib.request.urlopen(request, timeout=timeout)


def _retry(operation: Any, *, retries: int, label: str) -> Any:
    for attempt in range(retries + 1):
        try:
            return operation()
        except urllib.error.HTTPError as exc:
            if exc.code in {400, 401, 403, 404} or attempt >= retries:
                raise
            delay = min(2**attempt, 8)
        except (OSError, TimeoutError, urllib.error.URLError):
            if attempt >= retries:
                raise
            delay = min(2**attempt, 8)
        with _PRINT_LOCK:
            print(f"retrying {label} in {delay}s (attempt {attempt + 2}/{retries + 1})")
        time.sleep(delay)
    raise AssertionError("retry loop exhausted")


def fetch_dashboard(url: str, *, timeout: float, retries: int) -> dict[str, Any]:
    separator = "&" if "?" in url else "?"
    cache_busted = f"{url}{separator}t={int(time.time())}"

    def fetch() -> bytes:
        with _request(cache_busted, timeout=timeout) as response:
            return response.read()

    raw = _retry(fetch, retries=retries, label="dashboard")
    dashboard = json.loads(raw.decode("utf-8"))
    if not isinstance(dashboard, dict):
        raise ValueError("dashboard response is not a JSON object")
    _dashboard_runs(dashboard)
    return dashboard


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal_sha(value: Any) -> str:
    text = str(value or "").casefold().strip()
    return text.removeprefix("sha256:")


def _download(
    url: str,
    destination: Path,
    *,
    timeout: float,
    retries: int,
    force: bool,
    expected_sha256: str = "",
    expected_size: int | None = None,
) -> dict[str, Any]:
    expected_sha256 = _normal_sha(expected_sha256)
    if destination.is_file() and not force:
        size = destination.stat().st_size
        digest = _sha256(destination)
        size_ok = expected_size is None or size == expected_size
        hash_ok = not expected_sha256 or digest == expected_sha256
        if size_ok and hash_ok:
            return {
                "status": (
                    "validated" if expected_sha256 or expected_size is not None else "existing"
                ),
                "size_bytes": size,
                "sha256": digest,
            }

    destination.parent.mkdir(parents=True, exist_ok=True)
    # A stable partial name lets interrupted large public artifacts resume with HTTP Range.
    # The final checksum/size validation still guards against stale or malformed partials.
    temporary = destination.with_name(f".{destination.name}.part")
    if not temporary.exists():
        legacy_partials = list(destination.parent.glob(f".{destination.name}.*.part"))
        if legacy_partials:
            os.replace(max(legacy_partials, key=lambda path: path.stat().st_size), temporary)

    def fetch() -> None:
        offset = temporary.stat().st_size if temporary.is_file() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else None
        with _request(url, timeout=timeout, headers=headers) as response:
            resumed = offset > 0 and getattr(response, "status", None) == 206
            mode = "ab" if resumed else "wb"
            with temporary.open(mode) as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(temporary, destination)

    _retry(fetch, retries=retries, label=url)
    size = destination.stat().st_size
    digest = _sha256(destination)
    if expected_size is not None and size != expected_size:
        status = "size_mismatch"
    elif expected_sha256 and digest != expected_sha256:
        status = "checksum_mismatch"
    else:
        status = "validated" if expected_sha256 or expected_size is not None else "downloaded"
    return {"status": status, "size_bytes": size, "sha256": digest}


def _artifact_filename(artifact_type: str, url: str) -> str:
    known = ARTIFACT_FILENAMES.get(artifact_type.upper())
    if known:
        return known
    basename = Path(urllib.parse.urlparse(url).path).name
    return basename if basename and basename not in {".", ".."} else _safe_slug(artifact_type)


def _request_url(artifacts: dict[str, Any]) -> str:
    preferred = artifacts.get("EVAL_VERDICT")
    candidates = [preferred] + list(artifacts.values())
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        parsed = urllib.parse.urlparse(candidate)
        path = parsed.path.rsplit("/", 1)[0] + "/request.json"
        return urllib.parse.urlunparse(parsed._replace(path=path, query="", fragment=""))
    return ""


def _expected_artifacts(verdict_path: Path) -> dict[str, dict[str, Any]]:
    if not verdict_path.is_file():
        return {}
    try:
        verdict = _read_json(verdict_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    metadata = verdict.get("artifact_metadata")
    expected: dict[str, dict[str, Any]] = {}
    if not isinstance(metadata, dict):
        return expected
    for value in metadata.values():
        if not isinstance(value, dict):
            continue
        object_key = str(value.get("object_key") or "")
        uri = str(value.get("uri") or "")
        basename = Path(object_key or urllib.parse.urlparse(uri).path).name
        if not basename:
            continue
        expected[basename] = {
            "sha256": _normal_sha(value.get("sha256")),
            "size_bytes": _as_int(value.get("size_bytes")),
        }
    return expected


def _merge_run_metadata(path: Path, run: dict[str, Any], observed_at: str) -> dict[str, Any]:
    old: dict[str, Any] = {}
    if path.is_file():
        try:
            old = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            old = {}
    old_reasons = old.get("selection_reasons")
    reasons = set(old_reasons if isinstance(old_reasons, list) else [])
    reasons.update(run.get("_selection_reasons", []))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "first_seen_at": old.get("first_seen_at") or observed_at,
        "last_seen_at": observed_at,
        "model_slug": run["_model_slug"],
        "selection_reasons": sorted(reasons),
        "target_versions": run.get("_target_versions", []),
        "dashboard_run": {key: value for key, value in run.items() if not key.startswith("_")},
    }
    _write_json_atomic(path, metadata)
    return metadata


def crawl_run(
    root: Path,
    run: dict[str, Any],
    *,
    observed_at: str,
    timeout: float,
    retries: int,
    force: bool,
    include_request: bool,
    artifact_types: set[str] | None = None,
) -> dict[str, Any]:
    run_id = str(run["eval_run_id"])
    run_dir = root / "runs" / str(run["_model_slug"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _merge_run_metadata(run_dir / "run-metadata.json", run, observed_at)

    dashboard_artifacts = run.get("artifacts")
    artifacts = dict(dashboard_artifacts) if isinstance(dashboard_artifacts, dict) else {}
    if include_request and "REQUEST" not in artifacts:
        request_url = _request_url(artifacts)
        if request_url:
            artifacts["REQUEST"] = request_url

    records: list[dict[str, Any]] = []
    verdict_url = artifacts.get("EVAL_VERDICT")
    if isinstance(verdict_url, str) and verdict_url:
        verdict_path = run_dir / ARTIFACT_FILENAMES["EVAL_VERDICT"]
        try:
            result = _download(
                verdict_url,
                verdict_path,
                timeout=timeout,
                retries=retries,
                force=force,
            )
        except Exception as exc:  # Continue so other public artifacts can still be collected.
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        records.append(
            {
                "artifact_type": "EVAL_VERDICT",
                "url": verdict_url,
                "relative_path": verdict_path.relative_to(root).as_posix(),
                **result,
            }
        )

    expected = _expected_artifacts(run_dir / ARTIFACT_FILENAMES["EVAL_VERDICT"])
    for artifact_type, url_value in sorted(artifacts.items()):
        artifact_type = str(artifact_type).upper()
        if artifact_type == "EVAL_VERDICT" or not isinstance(url_value, str) or not url_value:
            continue
        if artifact_types is not None and artifact_type not in artifact_types:
            continue
        filename = _artifact_filename(artifact_type, url_value)
        destination = run_dir / filename
        expectation = expected.get(filename, {})
        try:
            result = _download(
                url_value,
                destination,
                timeout=timeout,
                retries=retries,
                force=force,
                expected_sha256=str(expectation.get("sha256") or ""),
                expected_size=_as_int(expectation.get("size_bytes")),
            )
        except urllib.error.HTTPError as exc:
            if artifact_type == "REQUEST" and exc.code == 404:
                result = {"status": "not_public", "error": f"HTTP {exc.code}"}
            else:
                result = {"status": "error", "error": f"HTTPError: {exc}"}
        except Exception as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        records.append(
            {
                "artifact_type": artifact_type,
                "url": url_value,
                "relative_path": destination.relative_to(root).as_posix(),
                "expected_sha256": expectation.get("sha256"),
                "expected_size_bytes": expectation.get("size_bytes"),
                **result,
            }
        )

    state = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "eval_run_id": run_id,
        "artifacts": sorted(records, key=lambda item: str(item["artifact_type"])),
    }
    _write_json_atomic(run_dir / "artifact-state.json", state)
    failed = [
        item
        for item in records
        if item.get("status") in {"error", "size_mismatch", "checksum_mismatch"}
    ]
    with _PRINT_LOCK:
        print(
            f"[{run_id}] {run['_model_slug']}: "
            f"{len(records) - len(failed)}/{len(records)} artifacts ready"
        )
    return {"eval_run_id": run_id, "artifact_count": len(records), "failures": failed}


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield line_number, value


def _source_from_sample_id(sample_id: str) -> str:
    if "/data/" in sample_id:
        return sample_id.split("/data/", 1)[0]
    return sample_id.split(":", 1)[0]


def _turn_count(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    return _as_int(value)


def _discover_metadata(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    discovered: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / "runs").glob("*/*/run-metadata.json")):
        try:
            discovered.append((path.parent, _read_json(path)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
    return discovered


def _create_index_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE models (
            model_uri TEXT PRIMARY KEY,
            model_slug TEXT NOT NULL,
            target_versions_json TEXT NOT NULL,
            selection_reasons_json TEXT NOT NULL,
            run_count INTEGER NOT NULL,
            best_challenger_score REAL,
            mean_challenger_score REAL,
            best_win_margin REAL
        );
        CREATE TABLE runs (
            eval_run_id TEXT PRIMARY KEY,
            submission_id TEXT,
            model_uri TEXT NOT NULL,
            model_slug TEXT NOT NULL,
            uid INTEGER,
            hotkey TEXT,
            finished_at TEXT,
            challenger_won INTEGER,
            coronated INTEGER,
            assigned_king_version INTEGER,
            score_challenger REAL,
            score_king REAL,
            win_margin REAL,
            required_win_margin REAL,
            scored_sample_count INTEGER,
            scoring_mode TEXT,
            selection_reasons_json TEXT NOT NULL,
            target_versions_json TEXT NOT NULL,
            relative_directory TEXT NOT NULL,
            dashboard_run_json TEXT NOT NULL
        );
        CREATE TABLE artifacts (
            eval_run_id TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            url TEXT,
            relative_path TEXT,
            status TEXT NOT NULL,
            size_bytes INTEGER,
            sha256 TEXT,
            expected_size_bytes INTEGER,
            expected_sha256 TEXT,
            error TEXT,
            PRIMARY KEY (eval_run_id, artifact_type),
            FOREIGN KEY (eval_run_id) REFERENCES runs(eval_run_id)
        );
        CREATE TABLE samples (
            eval_run_id TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            source TEXT,
            sample_phase TEXT,
            rewrite_mode TEXT,
            generated_line INTEGER,
            scoring_line INTEGER,
            scored INTEGER,
            king_score REAL,
            challenger_score REAL,
            score_delta REAL,
            question_count INTEGER,
            king_error TEXT,
            challenger_error TEXT,
            king_turns INTEGER,
            challenger_turns INTEGER,
            king_output_chars INTEGER,
            challenger_output_chars INTEGER,
            PRIMARY KEY (eval_run_id, sample_id),
            FOREIGN KEY (eval_run_id) REFERENCES runs(eval_run_id)
        );
        CREATE INDEX samples_source_idx ON samples(source);
        CREATE INDEX samples_phase_idx ON samples(sample_phase);
        CREATE INDEX samples_delta_idx ON samples(score_delta);
        CREATE INDEX runs_score_idx ON runs(score_challenger);
        """
    )


def build_index(root: Path) -> dict[str, Any]:
    """Rebuild the cumulative SQLite and JSONL indexes from downloaded run directories."""

    index_dir = root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    database = index_dir / "rollouts.sqlite3"
    temporary = index_dir / f".rollouts.{os.getpid()}.tmp.sqlite3"
    if temporary.exists():
        temporary.unlink()

    discovered = _discover_metadata(root)
    errors: list[str] = []
    run_rows: list[dict[str, Any]] = []
    model_accumulator: dict[str, dict[str, Any]] = {}
    sample_count = 0
    artifact_count = 0
    indexed_directories: dict[str, Path] = {}
    duplicate_directories: list[str] = []

    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        _create_index_schema(connection)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (("schema_version", str(SCHEMA_VERSION)), ("generated_at", _utc_now())),
        )
        for run_dir, metadata in discovered:
            run = metadata.get("dashboard_run")
            if not isinstance(run, dict) or not run.get("eval_run_id"):
                errors.append(f"invalid run metadata: {run_dir}")
                continue
            run_id = str(run["eval_run_id"])
            if run_id in indexed_directories:
                original_dir = indexed_directories[run_id]
                filenames = ("generated-samples.jsonl", "scoring-results.jsonl", "verdict.json")
                identical = all(
                    (original_dir / name).is_file() and (run_dir / name).is_file()
                    and _sha256(original_dir / name) == _sha256(run_dir / name)
                    for name in filenames
                )
                if identical:
                    duplicate_directories.append(run_dir.relative_to(root).as_posix())
                else:
                    errors.append(f"conflicting duplicate run directory: {run_dir}")
                continue
            indexed_directories[run_id] = run_dir
            model_uri = str(run.get("model_uri") or "")
            model_slug = str(metadata.get("model_slug") or run_dir.parent.name)
            reasons = metadata.get("selection_reasons") or []
            versions = metadata.get("target_versions") or []
            relative_directory = run_dir.relative_to(root).as_posix()
            row = {
                "eval_run_id": run_id,
                "submission_id": run.get("submission_id"),
                "model_uri": model_uri,
                "model_slug": model_slug,
                "uid": _as_int(run.get("uid")),
                "hotkey": run.get("hotkey"),
                "finished_at": run.get("finished_at"),
                "challenger_won": bool(run.get("challenger_won")),
                "coronated": bool(run.get("coronated")),
                "assigned_king_version": _as_int(run.get("king_version")),
                "score_challenger": _as_float(run.get("score_challenger")),
                "score_king": _as_float(run.get("score_king")),
                "win_margin": _as_float(run.get("win_margin")),
                "required_win_margin": _as_float(run.get("required_win_margin")),
                "scored_sample_count": _as_int(run.get("scored_sample_count")),
                "scoring_mode": run.get("scoring_mode"),
                "selection_reasons": reasons,
                "target_versions": versions,
                "relative_directory": relative_directory,
            }
            run_rows.append(row)
            connection.execute(
                """
                INSERT INTO runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    row["submission_id"],
                    model_uri,
                    model_slug,
                    row["uid"],
                    row["hotkey"],
                    row["finished_at"],
                    int(row["challenger_won"]),
                    int(row["coronated"]),
                    row["assigned_king_version"],
                    row["score_challenger"],
                    row["score_king"],
                    row["win_margin"],
                    row["required_win_margin"],
                    row["scored_sample_count"],
                    row["scoring_mode"],
                    _json_text(reasons),
                    _json_text(versions),
                    relative_directory,
                    _json_text(run),
                ),
            )

            accumulator = model_accumulator.setdefault(
                model_uri,
                {
                    "model_uri": model_uri,
                    "model_slug": model_slug,
                    "target_versions": set(),
                    "selection_reasons": set(),
                    "scores": [],
                    "margins": [],
                    "run_count": 0,
                },
            )
            accumulator["target_versions"].update(versions)
            accumulator["selection_reasons"].update(reasons)
            accumulator["run_count"] += 1
            if row["score_challenger"] is not None:
                accumulator["scores"].append(row["score_challenger"])
            if row["win_margin"] is not None:
                accumulator["margins"].append(row["win_margin"])

            state_path = run_dir / "artifact-state.json"
            if state_path.is_file():
                try:
                    state = _read_json(state_path)
                    artifact_rows = state.get("artifacts") or []
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{run_id}: invalid artifact state: {exc}")
                    artifact_rows = []
                for artifact in artifact_rows if isinstance(artifact_rows, list) else []:
                    if not isinstance(artifact, dict) or not artifact.get("artifact_type"):
                        continue
                    artifact_count += 1
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            str(artifact["artifact_type"]),
                            artifact.get("url"),
                            artifact.get("relative_path"),
                            str(artifact.get("status") or "unknown"),
                            _as_int(artifact.get("size_bytes")),
                            artifact.get("sha256"),
                            _as_int(artifact.get("expected_size_bytes")),
                            artifact.get("expected_sha256"),
                            artifact.get("error"),
                        ),
                    )

            generated_path = run_dir / ARTIFACT_FILENAMES["GENERATED_SAMPLES"]
            if generated_path.is_file():
                try:
                    for line_number, sample in _iter_jsonl(generated_path):
                        sample_id = str(sample.get("sample_id") or "")
                        if not sample_id:
                            continue
                        connection.execute(
                            """
                            INSERT INTO samples (
                                eval_run_id, sample_id, source, rewrite_mode, generated_line,
                                king_error, challenger_error, king_turns, challenger_turns,
                                king_output_chars, challenger_output_chars
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(eval_run_id, sample_id) DO UPDATE SET
                                source=excluded.source,
                                rewrite_mode=excluded.rewrite_mode,
                                generated_line=excluded.generated_line,
                                king_error=excluded.king_error,
                                challenger_error=excluded.challenger_error,
                                king_turns=excluded.king_turns,
                                challenger_turns=excluded.challenger_turns,
                                king_output_chars=excluded.king_output_chars,
                                challenger_output_chars=excluded.challenger_output_chars
                            """,
                            (
                                run_id,
                                sample_id,
                                _source_from_sample_id(sample_id),
                                sample.get("rewrite_mode"),
                                line_number,
                                sample.get("king_error"),
                                sample.get("chal_error"),
                                _turn_count(sample.get("previous_king_turns")),
                                _turn_count(sample.get("challenger_turns")),
                                len(side_output(sample, "previous_king")),
                                len(side_output(sample, "challenger")),
                            ),
                        )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{run_id}: generated samples: {exc}")

            scoring_path = run_dir / ARTIFACT_FILENAMES["SCORING_RESULTS"]
            if scoring_path.is_file():
                try:
                    for line_number, sample in _iter_jsonl(scoring_path):
                        sample_id = str(sample.get("sample_id") or "")
                        if not sample_id:
                            continue
                        question_source = sample.get("question_source")
                        phase = (
                            question_source.get("sample_phase")
                            if isinstance(question_source, dict)
                            else None
                        )
                        king_score = _as_float(sample.get("king_score"))
                        challenger_score = _as_float(sample.get("challenger_score"))
                        delta = (
                            challenger_score - king_score
                            if king_score is not None and challenger_score is not None
                            else None
                        )
                        questions = sample.get("questions")
                        connection.execute(
                            """
                            INSERT INTO samples (
                                eval_run_id, sample_id, source, sample_phase, scoring_line,
                                scored, king_score, challenger_score, score_delta, question_count
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(eval_run_id, sample_id) DO UPDATE SET
                                sample_phase=excluded.sample_phase,
                                scoring_line=excluded.scoring_line,
                                scored=excluded.scored,
                                king_score=excluded.king_score,
                                challenger_score=excluded.challenger_score,
                                score_delta=excluded.score_delta,
                                question_count=excluded.question_count
                            """,
                            (
                                run_id,
                                sample_id,
                                _source_from_sample_id(sample_id),
                                phase,
                                line_number,
                                int(bool(sample.get("scored"))),
                                king_score,
                                challenger_score,
                                delta,
                                len(questions) if isinstance(questions, list) else None,
                            ),
                        )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{run_id}: scoring results: {exc}")

        model_rows: list[dict[str, Any]] = []
        for accumulator in model_accumulator.values():
            scores = accumulator.pop("scores")
            margins = accumulator.pop("margins")
            accumulator["target_versions"] = sorted(accumulator["target_versions"])
            accumulator["selection_reasons"] = sorted(accumulator["selection_reasons"])
            accumulator["best_challenger_score"] = max(scores) if scores else None
            accumulator["mean_challenger_score"] = sum(scores) / len(scores) if scores else None
            accumulator["best_win_margin"] = max(margins) if margins else None
            model_rows.append(accumulator)
            connection.execute(
                "INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    accumulator["model_uri"],
                    accumulator["model_slug"],
                    _json_text(accumulator["target_versions"]),
                    _json_text(accumulator["selection_reasons"]),
                    accumulator["run_count"],
                    accumulator["best_challenger_score"],
                    accumulator["mean_challenger_score"],
                    accumulator["best_win_margin"],
                ),
            )
        sample_count = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, database)

    run_rows.sort(key=lambda item: (str(item.get("finished_at") or ""), item["eval_run_id"]))
    model_rows.sort(
        key=lambda item: (-(item.get("best_challenger_score") or -1.0), item["model_uri"])
    )
    _write_text_atomic(
        index_dir / "runs.jsonl", "".join(_json_text(row) + "\n" for row in run_rows)
    )
    _write_text_atomic(
        index_dir / "models.jsonl", "".join(_json_text(row) + "\n" for row in model_rows)
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "database": database.relative_to(root).as_posix(),
        "run_count": len(run_rows),
        "model_count": len(model_rows),
        "artifact_count": artifact_count,
        "sample_count": sample_count,
        "index_errors": errors,
        "identical_duplicate_directories": duplicate_directories,
    }
    _write_json_atomic(index_dir / "summary.json", summary)
    return summary


def _snapshot_dashboard(root: Path, dashboard: dict[str, Any], observed_at: str) -> None:
    compact_time = observed_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    snapshots = root / "dashboard-snapshots"
    _write_json_atomic(snapshots / f"dashboard-{compact_time}.json", dashboard)
    _write_json_atomic(snapshots / "latest.json", dashboard)


def _selection_document(
    selected: list[dict[str, Any]], targets: dict[int, str], observed_at: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "selected_at": observed_at,
        "target_models": {str(version): uri for version, uri in sorted(targets.items())},
        "runs": [
            {
                "eval_run_id": run["eval_run_id"],
                "model_uri": run.get("model_uri"),
                "model_slug": run["_model_slug"],
                "score_challenger": run.get("score_challenger"),
                "score_king": run.get("score_king"),
                "win_margin": run.get("win_margin"),
                "selection_reasons": run["_selection_reasons"],
                "target_versions": run["_target_versions"],
            }
            for run in selected
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-url", default=DEFAULT_DASHBOARD_URL)
    parser.add_argument("--dashboard-file", type=Path, help="Use a local dashboard snapshot")
    parser.add_argument("--output", type=Path, default=Path("rollouts/corpus"))
    parser.add_argument(
        "--king-version",
        action="append",
        type=int,
        dest="king_versions",
        help="Always include every run of this crowned model; repeatable (default: 121, 122)",
    )
    parser.add_argument("--top-challengers", type=int, default=DEFAULT_TOP_CHALLENGERS)
    parser.add_argument("--min-challenger-score", type=float)
    parser.add_argument("--min-win-margin", type=float)
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Include every eval run with public artifacts in the current dashboard snapshot",
    )
    parser.add_argument(
        "--run-id-file",
        type=Path,
        help="Restrict selected dashboard runs to eval run IDs listed one per line",
    )
    parser.add_argument(
        "--artifact-type",
        action="append",
        help="Download only this artifact type (repeatable); EVAL_VERDICT is always retained",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Redownload existing artifacts")
    parser.add_argument("--no-request", action="store_true", help="Do not probe request.json")
    parser.add_argument("--dry-run", action="store_true", help="Show selection without downloading")
    parser.add_argument(
        "--index-only", action="store_true", help="Only rebuild indexes from the existing corpus"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit nonzero on required artifact/checksum failures"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.output.expanduser().resolve()
    if args.index_only:
        summary = build_index(root)
        print(_json_text(summary, pretty=True), end="")
        return 1 if summary["index_errors"] and args.strict else 0

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.top_challengers < 0:
        raise SystemExit("--top-challengers cannot be negative")
    versions = args.king_versions or list(DEFAULT_KING_VERSIONS)
    if args.dashboard_file:
        dashboard = _read_json(args.dashboard_file.expanduser().resolve())
        _dashboard_runs(dashboard)
    else:
        dashboard = fetch_dashboard(
            args.dashboard_url, timeout=args.timeout, retries=max(0, args.retries)
        )
    selected, targets = select_runs(
        dashboard,
        king_versions=versions,
        top_challengers=args.top_challengers,
        min_challenger_score=args.min_challenger_score,
        min_win_margin=args.min_win_margin,
        all_runs=args.all_runs,
    )
    if args.run_id_file:
        requested_run_ids = {
            line.strip()
            for line in args.run_id_file.expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        selected = [run for run in selected if str(run.get("eval_run_id")) in requested_run_ids]
    missing_targets = sorted(set(versions) - set(targets))
    if missing_targets:
        print(
            "warning: target king versions absent from this dashboard snapshot: "
            + ", ".join(map(str, missing_targets)),
            file=sys.stderr,
        )
    selection = _selection_document(selected, targets, _utc_now())
    print(_json_text(selection, pretty=True), end="")
    if args.dry_run:
        return 0

    observed_at = str(selection["selected_at"])
    artifact_types = (
        {str(value).upper() for value in args.artifact_type} if args.artifact_type else None
    )
    root.mkdir(parents=True, exist_ok=True)
    _snapshot_dashboard(root, dashboard, observed_at)
    _write_json_atomic(root / "latest-selection.json", selection)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                crawl_run,
                root,
                run,
                observed_at=observed_at,
                timeout=args.timeout,
                retries=max(0, args.retries),
                force=args.force,
                include_request=(
                    not args.no_request
                    and (artifact_types is None or "REQUEST" in artifact_types)
                ),
                artifact_types=artifact_types,
            )
            for run in selected
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    summary = build_index(root)
    failures = [failure for result in results for failure in result["failures"]]
    catalog = {
        **summary,
        "dashboard_url": args.dashboard_url,
        "selected_run_count": len(selected),
        "target_king_versions": versions,
        "top_challengers": args.top_challengers,
        "min_challenger_score": args.min_challenger_score,
        "min_win_margin": args.min_win_margin,
        "all_runs": args.all_runs,
        "artifact_types": sorted(artifact_types) if artifact_types is not None else None,
        "crawl_failure_count": len(failures),
        "crawl_failures": failures,
    }
    _write_json_atomic(root / "catalog.json", catalog)
    print(_json_text(catalog, pretty=True), end="")
    return 2 if args.strict and (failures or summary["index_errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
