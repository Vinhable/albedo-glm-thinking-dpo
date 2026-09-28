#!/usr/bin/env python3
"""Build sample-side audit tables and deterministic quality filters for rollout corpora."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from albedo_eval_service.shared.edit_detection import edited_in_turn  # noqa: E402
from albedo_eval_service.shared.loop_check import (  # noqa: E402
    candidate_turns,
    commands_of,
    loop_verdict,
)
from albedo_eval_service.shared.observation_format import (  # noqa: E402
    strip_leaked_reasoning,
    unusable_turn,
)
from albedo_eval_service.shared.submit_protocol import is_exact_submission  # noqa: E402
from generated_sample_fields import side_output  # noqa: E402

SCHEMA_VERSION = 1
SIDES = ("previous_king", "challenger")
MAJOR_FLAGS = (
    "malformed_bash",
    "loop",
    "broad_search_without_inspect",
    "unsupported_completion_claim",
)

TEST_RE = re.compile(
    r"\b(?:pytest|cargo\s+test|go\s+test|npm\s+(?:test|run\s+test)|"
    r"pnpm\s+(?:test|run\s+test)|yarn\s+test|tox|nox|rspec|mvn\s+test|"
    r"gradle\s+test|ctest|make\s+test)\b",
    re.IGNORECASE,
)
GIT_DIFF_RE = re.compile(r"\bgit\s+diff\b", re.IGNORECASE)
SEARCH_RE = re.compile(r"\b(?:rg|grep|git\s+grep|find|fd|tree)\b", re.IGNORECASE)
RECURSIVE_SEARCH_RE = re.compile(
    r"\bfind\s+(?:\.|/workspace|\$PWD)\b|\b(?:grep|rg)\b[^\n]*(?:\s-[^\s]*[Rr]|\s\.\s*$)"
    r"|\brg\s+--files\b|\bls\s+-[^\s]*R|\btree\b",
    re.IGNORECASE,
)
FILE_TOKEN_RE = re.compile(r"(?:/workspace/)?[\w./~-]*[\w-]+\.[A-Za-z][A-Za-z0-9_]*")
TARGETED_VIEW_RE = re.compile(
    r"\b(?:sed\s+-n|cat\s+-n|nl\s+-ba|head|tail|less|bat)\b"
    r"|\b(?:rg|grep|git\s+diff)\b[^\n]*[\w./~-]+\.[A-Za-z][A-Za-z0-9_]*"
    r"|\b(?:read_text|read_bytes)\s*\(|\bopen\s*\([^)]*['\"]r",
    re.IGNORECASE,
)
ACTION_BLOCK_RE = re.compile(
    r"```(?:bash|sh|shell)[ \t]*\n(.*?)```|<([a-z_]*bash[a-z_]*)>(.*?)</\2>",
    re.IGNORECASE | re.DOTALL,
)

EDIT_CLAIM_RE = re.compile(
    r"\b(?:the\s+)?(?:bug|issue|problem|fix|implementation|change|patch)\s+"
    r"(?:is|has\s+been|was)\s+(?:fixed|resolved|implemented|applied|completed|done)\b"
    r"|\b(?:i(?:'ve|\s+have)?|we(?:'ve|\s+have)?)\s+"
    r"(?:fixed|resolved|implemented|applied|corrected)\b"
    r"|\bsuccessfully\s+(?:fixed|resolved|implemented|applied|corrected)\b",
    re.IGNORECASE,
)
TEST_CLAIM_RE = re.compile(
    r"\b(?:all\s+)?(?:tests?|checks?|test\s+suite)\s+"
    r"(?:now\s+)?(?:pass|passes|passed|are\s+passing|succeeded|is\s+green)\b"
    r"|\bverified\s+(?:by|with|using)\s+(?:the\s+)?(?:tests?|test\s+suite|pytest)\b",
    re.IGNORECASE,
)
COMPLETION_CLAIM_RE = re.compile(
    r"\b(?:the\s+)?task\s+(?:is|has\s+been)\s+(?:complete|completed|done|finished)\b"
    r"|\b(?:work|implementation)\s+is\s+(?:complete|done|finished)\b"
    r"|\bready\s+(?:to|for)\s+(?:submit|submission)\b",
    re.IGNORECASE,
)

PROSE_LIKE_HEADS = {
    "Actually,",
    "And",
    "But",
    "File",
    "For",
    "If",
    "Let",
    "On",
    "Since",
    "So",
    "The",
    "This",
    "Wait,",
    "When",
    "think>",
}


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


def _source_from_sample_id(sample_id: str) -> str:
    if "/data/" in sample_id:
        return sample_id.split("/data/", 1)[0]
    return sample_id.split(":", 1)[0]


def _clip(text: str, limit: int = 240) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _command_head(command: str) -> str:
    text = command.strip()
    while text.startswith("cd ") and "&&" in text:
        text = text.split("&&", 1)[1].strip()
    token = text.split(maxsplit=1)[0] if text else ""
    return token.rsplit("/", 1)[-1]


def strict_action_blocks(text: str) -> list[str]:
    """Extract explicit shell actions that satisfy the training prompt contract."""

    return [
        " ".join((match.group(1) if match.group(1) is not None else match.group(3)).split())
        for match in ACTION_BLOCK_RE.finditer(text or "")
    ]


def _is_search(command: str) -> bool:
    return bool(SEARCH_RE.search(command))


def _is_targeted_inspect(command: str) -> bool:
    if not TARGETED_VIEW_RE.search(command):
        return False
    # ``cat > file`` and ``cat >> file`` are edits, not inspections.
    if re.search(r"\bcat\s*(?:>|>>)", command):
        return False
    return bool(FILE_TOKEN_RE.search(command) or re.search(r"\bsed\s+-n\b", command))


def _is_broad_search(command: str) -> bool:
    if RECURSIVE_SEARCH_RE.search(command):
        return True
    if not _is_search(command):
        return False
    # A search constrained to an explicit source file is already a targeted inspection.
    return not _is_targeted_inspect(command)


def _visible_prose(turn: str) -> str:
    visible = strip_leaked_reasoning(turn or "")
    return ACTION_BLOCK_RE.sub(" ", visible)


def _claim_snippets(text: str, pattern: re.Pattern[str]) -> list[str]:
    snippets: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        if pattern.search(part):
            snippets.append(_clip(part))
    return snippets


def prefix_messages(document: str, messages: Any) -> tuple[list[dict[str, Any]], bool]:
    """Return messages before the first scored candidate turn."""

    if not isinstance(messages, list):
        return [], False
    turns = candidate_turns(document)
    if not turns:
        return [], False
    first = turns[0].strip()
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        if str(message.get("content") or "").strip() == first:
            return messages[:index], True
    return [], False


def _prefix_features(messages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    assistant_turns = [
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    commands = [command for turn in assistant_turns for command in strict_action_blocks(turn)]
    return {
        "turns": len(assistant_turns),
        "edited": any(edited_in_turn(turn) for turn in assistant_turns),
        "tested": any(TEST_RE.search(command) for command in commands),
        "git_diff": any(GIT_DIFF_RE.search(command) for command in commands),
        "targeted_inspect": any(_is_targeted_inspect(command) for command in commands),
    }


def audit_candidate(
    document: str,
    *,
    submit_command: str = "",
    prefix: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Compute four deterministic quality flags and supporting diagnostics."""

    turns = candidate_turns(document)
    prefix_info = _prefix_features(prefix)
    official_commands = commands_of(turns)
    all_commands: list[tuple[int, str]] = []
    malformed_evidence: list[dict[str, Any]] = []
    edit_turns: list[int] = []
    test_turns: list[int] = []
    diff_turns: list[int] = []
    submit_turns: list[int] = []
    visible_chars = 0

    for turn_index, turn in enumerate(turns, start=1):
        blocks = strict_action_blocks(turn)
        all_commands.extend((turn_index, command) for command in blocks)
        bad_reason = unusable_turn(turn)
        if bad_reason:
            malformed_evidence.append(
                {"turn": turn_index, "reason": bad_reason, "excerpt": _clip(turn)}
            )
        if len(blocks) > 1:
            malformed_evidence.append(
                {
                    "turn": turn_index,
                    "reason": f"multiple action blocks ({len(blocks)})",
                    "excerpt": _clip(turn),
                }
            )
        for command in blocks:
            head = _command_head(command)
            if not command.strip():
                malformed_evidence.append(
                    {"turn": turn_index, "reason": "empty action block", "excerpt": ""}
                )
            elif head in PROSE_LIKE_HEADS:
                malformed_evidence.append(
                    {
                        "turn": turn_index,
                        "reason": f"prose-like command head {head!r}",
                        "excerpt": _clip(command),
                    }
                )
        if edited_in_turn(turn):
            edit_turns.append(turn_index)
        if any(TEST_RE.search(command) for command in blocks):
            test_turns.append(turn_index)
        if any(GIT_DIFF_RE.search(command) for command in blocks):
            diff_turns.append(turn_index)
        if is_exact_submission(turn, submit_command):
            submit_turns.append(turn_index)
        visible_chars += len(_visible_prose(turn))

    loop = loop_verdict(turns)
    broad_commands = [
        (index, command) for index, command in all_commands if _is_broad_search(command)
    ]
    inspect_commands = [
        (index, command) for index, command in all_commands if _is_targeted_inspect(command)
    ]
    first_broad = broad_commands[0][0] if broad_commands else None
    inspect_after_broad = bool(
        first_broad is not None and any(index >= first_broad for index, _ in inspect_commands)
    )
    broad_without_inspect = len(broad_commands) >= 2 and not inspect_after_broad

    unsupported_evidence: list[dict[str, Any]] = []
    edit_seen = bool(prefix_info["edited"])
    test_seen = bool(prefix_info["tested"])
    diff_seen = bool(prefix_info["git_diff"])
    for turn_index, turn in enumerate(turns, start=1):
        if turn_index in edit_turns:
            edit_seen = True
        if turn_index in test_turns:
            test_seen = True
        if turn_index in diff_turns:
            diff_seen = True
        prose = _visible_prose(turn)
        for snippet in _claim_snippets(prose, EDIT_CLAIM_RE):
            if not edit_seen:
                unsupported_evidence.append(
                    {"turn": turn_index, "reason": "fix claim without prior edit", "text": snippet}
                )
        for snippet in _claim_snippets(prose, TEST_CLAIM_RE):
            if not test_seen:
                unsupported_evidence.append(
                    {
                        "turn": turn_index,
                        "reason": "test-pass claim without prior test command",
                        "text": snippet,
                    }
                )
        for snippet in _claim_snippets(prose, COMPLETION_CLAIM_RE):
            if not edit_seen or not (test_seen or diff_seen):
                unsupported_evidence.append(
                    {
                        "turn": turn_index,
                        "reason": "completion claim without edit and verification evidence",
                        "text": snippet,
                    }
                )

    flags = {
        "malformed_bash": bool(malformed_evidence),
        "loop": bool(loop.looped),
        "broad_search_without_inspect": broad_without_inspect,
        "unsupported_completion_claim": bool(unsupported_evidence),
    }
    evidence = {
        "malformed_bash": malformed_evidence[:8],
        "loop": {
            "reasons": list(loop.reasons),
            "commands": [
                {
                    "command": _clip(entry.command),
                    "count": entry.count,
                    "longest_run": entry.longest_run,
                }
                for entry in loop.commands[:5]
            ],
        },
        "broad_search_without_inspect": {
            "broad_commands": [
                {"turn": index, "command": _clip(command)}
                for index, command in broad_commands[:8]
            ],
            "targeted_inspect_commands": [
                {"turn": index, "command": _clip(command)}
                for index, command in inspect_commands[:8]
            ],
        },
        "unsupported_completion_claim": unsupported_evidence[:8],
    }
    first_edit = min(edit_turns) if edit_turns else None
    first_inspect = min((index for index, _ in inspect_commands), default=None)
    return {
        **flags,
        "quality_pass": not any(flags.values()),
        "flags": [flag for flag in MAJOR_FLAGS if flags[flag]],
        "evidence": evidence,
        "turn_count": len(turns),
        "command_count": len(official_commands),
        "unique_command_count": len(set(official_commands)),
        "duplicate_command_ratio": loop.dup_cmd_ratio,
        "max_consecutive_command_run": loop.max_cmd_run,
        "malformed_turn_count": len({item["turn"] for item in malformed_evidence}),
        "prose_like_command_count": sum(
            1 for _, command in all_commands if _command_head(command) in PROSE_LIKE_HEADS
        ),
        "search_count": sum(1 for _, command in all_commands if _is_search(command)),
        "broad_search_count": len(broad_commands),
        "targeted_inspect_count": len(inspect_commands),
        "edited": bool(edit_turns),
        "tested": bool(test_turns),
        "git_diff": bool(diff_turns),
        "exact_submit": bool(submit_turns),
        "edit_before_inspect": bool(
            first_edit is not None
            and not prefix_info["targeted_inspect"]
            and (first_inspect is None or first_edit < first_inspect)
        ),
        "unsupported_claim_count": len(unsupported_evidence),
        "visible_prose_chars": visible_chars,
        "output_chars": len(document or ""),
        "prefix_turn_count": prefix_info["turns"],
        "prefix_edited": prefix_info["edited"],
        "prefix_tested": prefix_info["tested"],
        "prefix_git_diff": prefix_info["git_diff"],
        "prefix_targeted_inspect": prefix_info["targeted_inspect"],
    }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield line_number, value


def _load_run_inputs(
    corpus: Path, connection: sqlite3.Connection
) -> Iterable[tuple[sqlite3.Row, Path, Path]]:
    connection.row_factory = sqlite3.Row
    query = """
        SELECT r.*,
               generated.relative_path AS generated_path,
               scoring.relative_path AS scoring_path
        FROM runs r
        JOIN artifacts generated
          ON generated.eval_run_id = r.eval_run_id
         AND generated.artifact_type = 'GENERATED_SAMPLES'
        LEFT JOIN artifacts scoring
          ON scoring.eval_run_id = r.eval_run_id
         AND scoring.artifact_type = 'SCORING_RESULTS'
        ORDER BY r.finished_at, r.eval_run_id
    """
    for row in connection.execute(query):
        generated = corpus / str(row["generated_path"])
        scoring = corpus / str(row["scoring_path"]) if row["scoring_path"] else Path()
        if generated.is_file():
            yield row, generated, scoring


def _side_identity(run: sqlite3.Row, dashboard_run: dict[str, Any], side: str) -> tuple[str, str]:
    if side == "challenger":
        return str(run["model_uri"]), str(run["model_slug"])
    king = dashboard_run.get("king")
    king = king if isinstance(king, dict) else {}
    uri = str(king.get("model_uri") or "unknown-king")
    version = _as_int(king.get("king_version"))
    revision = uri.rsplit("@", 1)[-1][:10] if "@" in uri else "unknown"
    slug = f"king-{version}" if version is not None else f"king-{revision}"
    return uri, slug


CANDIDATE_COLUMNS = (
    "eval_run_id",
    "sample_id",
    "side",
    "model_uri",
    "model_slug",
    "source",
    "sample_phase",
    "rewrite_mode",
    "generated_line",
    "scoring_line",
    "score",
    "peer_score",
    "score_delta",
    "scored",
    "quality_pass",
    "sft_eligible",
    *MAJOR_FLAGS,
    "turn_count",
    "command_count",
    "unique_command_count",
    "duplicate_command_ratio",
    "max_consecutive_command_run",
    "malformed_turn_count",
    "prose_like_command_count",
    "search_count",
    "broad_search_count",
    "targeted_inspect_count",
    "edited",
    "tested",
    "git_diff",
    "exact_submit",
    "edit_before_inspect",
    "unsupported_claim_count",
    "visible_prose_chars",
    "output_chars",
    "prefix_match",
    "prefix_turn_count",
    "prefix_edited",
    "prefix_tested",
    "prefix_git_diff",
    "prefix_targeted_inspect",
    "flags_json",
    "evidence_json",
)


def _create_schema(connection: sqlite3.Connection) -> None:
    real_columns = {"score", "peer_score", "score_delta", "duplicate_command_ratio"}
    text_columns = {
        "eval_run_id",
        "sample_id",
        "side",
        "model_uri",
        "model_slug",
        "source",
        "sample_phase",
        "rewrite_mode",
        "flags_json",
        "evidence_json",
    }
    definitions: list[str] = []
    for column in CANDIDATE_COLUMNS:
        if column in text_columns:
            sql_type = "TEXT"
        elif column in real_columns:
            sql_type = "REAL"
        else:
            sql_type = "INTEGER"
        required = " NOT NULL" if column in {"eval_run_id", "sample_id", "side"} else ""
        definitions.append(f"{column} {sql_type}{required}")
    connection.execute(
        f"""
        CREATE TABLE candidates (
            {', '.join(definitions)},
            PRIMARY KEY (eval_run_id, sample_id, side)
        )
        """
    )
    connection.executescript(
        """
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE preference_pairs (
            eval_run_id TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            source TEXT,
            sample_phase TEXT,
            king_score REAL,
            challenger_score REAL,
            challenger_delta REAL,
            preferred_side TEXT,
            preferred_score REAL,
            rejected_score REAL,
            preferred_quality_pass INTEGER,
            rejected_quality_pass INTEGER,
            eligible INTEGER NOT NULL,
            rejection_reasons_json TEXT NOT NULL,
            PRIMARY KEY (eval_run_id, sample_id)
        );
        CREATE INDEX candidates_flags_idx ON candidates(quality_pass, sft_eligible);
        CREATE INDEX candidates_source_idx ON candidates(source, sample_phase);
        CREATE INDEX candidates_score_idx ON candidates(score);
        CREATE INDEX candidates_model_idx ON candidates(model_uri);
        CREATE INDEX pairs_eligible_idx ON preference_pairs(eligible);
        CREATE VIEW clean_candidates AS SELECT * FROM candidates WHERE quality_pass = 1;
        CREATE VIEW flagged_candidates AS SELECT * FROM candidates WHERE quality_pass = 0;
        CREATE VIEW sft_candidates AS SELECT * FROM candidates WHERE sft_eligible = 1;
        CREATE VIEW eligible_preference_pairs AS
            SELECT * FROM preference_pairs WHERE eligible = 1;
        """
    )


def _candidate_for_database(row: dict[str, Any]) -> tuple[Any, ...]:
    boolean_columns = {
        "scored",
        "quality_pass",
        "sft_eligible",
        *MAJOR_FLAGS,
        "edited",
        "tested",
        "git_diff",
        "exact_submit",
        "edit_before_inspect",
        "prefix_match",
        "prefix_edited",
        "prefix_tested",
        "prefix_git_diff",
        "prefix_targeted_inspect",
    }
    return tuple(
        int(bool(row[column])) if column in boolean_columns else row[column]
        for column in CANDIDATE_COLUMNS
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _preference_pair(
    king: dict[str, Any],
    challenger: dict[str, Any],
    *,
    min_delta: float,
    min_preferred_score: float,
) -> dict[str, Any]:
    king_score = _as_float(king.get("score"))
    challenger_score = _as_float(challenger.get("score"))
    delta = (
        challenger_score - king_score
        if king_score is not None and challenger_score is not None
        else None
    )
    if delta is None or delta == 0:
        preferred, rejected = None, None
    elif delta > 0:
        preferred, rejected = challenger, king
    else:
        preferred, rejected = king, challenger
    reasons: list[str] = []
    if preferred is None:
        reasons.append("missing_or_tied_scores")
    else:
        if abs(delta) < min_delta:
            reasons.append("delta_below_threshold")
        if (_as_float(preferred.get("score")) or 0.0) < min_preferred_score:
            reasons.append("preferred_score_below_threshold")
        if not preferred.get("quality_pass"):
            reasons.append("preferred_failed_quality_filter")
    return {
        "eval_run_id": challenger["eval_run_id"],
        "sample_id": challenger["sample_id"],
        "source": challenger["source"],
        "sample_phase": challenger["sample_phase"],
        "king_score": king_score,
        "challenger_score": challenger_score,
        "challenger_delta": delta,
        "preferred_side": preferred["side"] if preferred else None,
        "preferred_score": preferred["score"] if preferred else None,
        "rejected_score": rejected["score"] if rejected else None,
        "preferred_quality_pass": bool(preferred and preferred["quality_pass"]),
        "rejected_quality_pass": bool(rejected and rejected["quality_pass"]),
        "eligible": not reasons,
        "rejection_reasons": reasons,
    }


def build_audit(
    corpus: Path,
    output: Path,
    *,
    min_sft_score: float = 0.65,
    min_pair_delta: float = 0.15,
    min_preferred_score: float = 0.65,
) -> dict[str, Any]:
    corpus_db = corpus / "index" / "rollouts.sqlite3"
    if not corpus_db.is_file():
        raise FileNotFoundError(f"corpus index not found: {corpus_db}")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "audit.sqlite3"
    temporary = output / f".audit.{os.getpid()}.tmp.sqlite3"
    if temporary.exists():
        temporary.unlink()

    source_connection = sqlite3.connect(corpus_db)
    audit_connection = sqlite3.connect(temporary)
    candidates: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        _create_schema(audit_connection)
        audit_connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            (
                ("schema_version", str(SCHEMA_VERSION)),
                ("generated_at", _utc_now()),
                ("corpus", str(corpus)),
                ("min_sft_score", str(min_sft_score)),
                ("min_pair_delta", str(min_pair_delta)),
                ("min_preferred_score", str(min_preferred_score)),
            ),
        )
        placeholders = ",".join("?" for _ in CANDIDATE_COLUMNS)
        insert_candidate = (
            f"INSERT INTO candidates ({','.join(CANDIDATE_COLUMNS)}) VALUES ({placeholders})"
        )
        for run, generated_path, scoring_path in _load_run_inputs(corpus, source_connection):
            dashboard_run = json.loads(str(run["dashboard_run_json"]))
            scores: dict[str, tuple[int, dict[str, Any]]] = {}
            if scoring_path.is_file():
                try:
                    scores = {
                        str(item.get("sample_id")): (line_number, item)
                        for line_number, item in _iter_jsonl(scoring_path)
                        if item.get("sample_id")
                    }
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{run['eval_run_id']}: scoring: {exc}")
            try:
                generated_rows = list(_iter_jsonl(generated_path))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{run['eval_run_id']}: generated: {exc}")
                continue
            for generated_line, sample in generated_rows:
                sample_id = str(sample.get("sample_id") or "")
                if not sample_id:
                    continue
                scoring_line, scoring = scores.get(sample_id, (None, {}))
                question_source = scoring.get("question_source")
                phase = (
                    question_source.get("sample_phase")
                    if isinstance(question_source, dict)
                    else None
                )
                side_rows: dict[str, dict[str, Any]] = {}
                for side in SIDES:
                    is_challenger = side == "challenger"
                    messages_key = "challenger_turns" if is_challenger else "previous_king_turns"
                    score_key = "challenger_score" if is_challenger else "king_score"
                    peer_key = "king_score" if is_challenger else "challenger_score"
                    document = side_output(sample, side)
                    prefix, prefix_match = prefix_messages(document, sample.get(messages_key))
                    features = audit_candidate(
                        document,
                        submit_command=str(sample.get("submit_command") or ""),
                        prefix=prefix,
                    )
                    score = _as_float(scoring.get(score_key))
                    peer_score = _as_float(scoring.get(peer_key))
                    model_uri, model_slug = _side_identity(run, dashboard_run, side)
                    row = {
                        "eval_run_id": str(run["eval_run_id"]),
                        "sample_id": sample_id,
                        "side": side,
                        "model_uri": model_uri,
                        "model_slug": model_slug,
                        "source": _source_from_sample_id(sample_id),
                        "sample_phase": phase,
                        "rewrite_mode": sample.get("rewrite_mode"),
                        "generated_line": generated_line,
                        "scoring_line": scoring_line,
                        "score": score,
                        "peer_score": peer_score,
                        "score_delta": (
                            score - peer_score
                            if score is not None and peer_score is not None
                            else None
                        ),
                        "scored": bool(scoring.get("scored")),
                        "quality_pass": features["quality_pass"],
                        "sft_eligible": bool(
                            features["quality_pass"]
                            and scoring.get("scored")
                            and score is not None
                            and score >= min_sft_score
                        ),
                        "prefix_match": prefix_match,
                        **{
                            key: value
                            for key, value in features.items()
                            if key not in {"flags", "evidence", "quality_pass"}
                        },
                        "flags_json": _json_text(features["flags"]),
                        "evidence_json": _json_text(features["evidence"]),
                    }
                    audit_connection.execute(insert_candidate, _candidate_for_database(row))
                    candidates.append(row)
                    side_rows[side] = row
                pair = _preference_pair(
                    side_rows["previous_king"],
                    side_rows["challenger"],
                    min_delta=min_pair_delta,
                    min_preferred_score=min_preferred_score,
                )
                pairs.append(pair)
                audit_connection.execute(
                    """
                    INSERT INTO preference_pairs
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pair["eval_run_id"],
                        pair["sample_id"],
                        pair["source"],
                        pair["sample_phase"],
                        pair["king_score"],
                        pair["challenger_score"],
                        pair["challenger_delta"],
                        pair["preferred_side"],
                        pair["preferred_score"],
                        pair["rejected_score"],
                        int(pair["preferred_quality_pass"]),
                        int(pair["rejected_quality_pass"]),
                        int(pair["eligible"]),
                        _json_text(pair["rejection_reasons"]),
                    ),
                )
        audit_connection.commit()
    finally:
        source_connection.close()
        audit_connection.close()
    os.replace(temporary, destination)

    _write_csv(output / "candidates.csv", candidates)
    flagged = [row for row in candidates if not row["quality_pass"]]
    sft = [row for row in candidates if row["sft_eligible"]]
    eligible_pairs = [pair for pair in pairs if pair["eligible"]]
    _write_text_atomic(
        output / "flagged-candidates.jsonl",
        "".join(_json_text(row) + "\n" for row in flagged),
    )
    _write_text_atomic(
        output / "sft-candidates.jsonl",
        "".join(
            _json_text(
                {
                    key: row[key]
                    for key in (
                        "eval_run_id",
                        "sample_id",
                        "side",
                        "model_uri",
                        "source",
                        "sample_phase",
                        "score",
                        "generated_line",
                    )
                }
            )
            + "\n"
            for row in sft
        ),
    )
    _write_text_atomic(
        output / "preference-pairs.jsonl",
        "".join(_json_text(pair) + "\n" for pair in eligible_pairs),
    )

    by_side: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        side_rows = [row for row in candidates if row["side"] == side]
        by_side[side] = {
            "candidate_count": len(side_rows),
            "quality_pass_count": sum(bool(row["quality_pass"]) for row in side_rows),
            "quality_pass_rate": (
                sum(bool(row["quality_pass"]) for row in side_rows) / len(side_rows)
                if side_rows
                else None
            ),
            "sft_eligible_count": sum(bool(row["sft_eligible"]) for row in side_rows),
            "flag_counts": {
                flag: sum(bool(row[flag]) for row in side_rows) for flag in MAJOR_FLAGS
            },
        }
    flag_combinations = Counter(
        "+".join(json.loads(row["flags_json"])) or "clean" for row in candidates
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "corpus": str(corpus),
        "database": str(destination),
        "sample_count": len(pairs),
        "candidate_count": len(candidates),
        "quality_pass_count": len(candidates) - len(flagged),
        "quality_pass_rate": (
            (len(candidates) - len(flagged)) / len(candidates) if candidates else None
        ),
        "sft_eligible_count": len(sft),
        "preference_eligible_count": len(eligible_pairs),
        "thresholds": {
            "min_sft_score": min_sft_score,
            "min_pair_delta": min_pair_delta,
            "min_preferred_score": min_preferred_score,
            "min_broad_searches_without_later_inspect": 2,
        },
        "by_side": by_side,
        "flag_combinations": dict(flag_combinations.most_common()),
        "errors": errors,
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("rollouts/corpus"))
    parser.add_argument(
        "--output",
        type=Path,
        help="Audit output directory (default: <corpus>/audit)",
    )
    parser.add_argument("--min-sft-score", type=float, default=0.65)
    parser.add_argument("--min-pair-delta", type=float, default=0.15)
    parser.add_argument("--min-preferred-score", type=float, default=0.65)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on JSONL errors")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = args.corpus.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else corpus / "audit"
    summary = build_audit(
        corpus,
        output,
        min_sft_score=args.min_sft_score,
        min_pair_delta=args.min_pair_delta,
        min_preferred_score=args.min_preferred_score,
    )
    print(_json_text(summary, pretty=True), end="")
    return 1 if args.strict and summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
