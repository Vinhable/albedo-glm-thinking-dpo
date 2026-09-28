#!/usr/bin/env python3
"""Read-only, deterministic Stage-A review scanner for C2 message rows.

Quality flags in this module are review cues, not semantic-correctness labels.
The only policy loop decision comes from the repository's canonical loop gate.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import shlex
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from albedo_eval_service.shared.loop_check import commands_of, loop_verdict  # noqa: E402

SCANNER_VERSION = "c3-stage-b-v3"
EXCERPT_CHARS = 240
SUPPORTED_ROLES = {"system", "developer", "user", "assistant", "tool"}
INFORMATIONAL_FLAGS = {
    "verification_signal",
    "action_change_after_error",
    "error_text_in_read_output",
    "ambiguous_error_text",
}


@dataclass(frozen=True)
class InvalidJSON:
    reason: str


def _decode_row(line: str) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        return InvalidJSON(f"invalid JSON at column {exc.colno}: {exc.msg}")


def _read_output_command(command: str) -> bool:
    """Whether every executed segment is only directory setup or reading/searching.

    This is deliberately lexical. Unknown shell constructs abstain instead of being
    interpreted, and therefore cannot turn an error into a source-output cue.
    """
    if not command.strip() or re.search(r"[;<>`$()\n]", command):
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&")
        lexer.whitespace_split = True
        lexer.commenters = ""
        shell_tokens = list(lexer)
    except ValueError:
        return False
    segments: list[list[str]] = [[]]
    for token in shell_tokens:
        if token in {"|", "&&"}:
            if not segments[-1]:
                return False
            segments.append([])
        elif token in {"&", "||"}:
            return False
        else:
            segments[-1].append(token)
    if not segments[-1]:
        return False
    saw_read = False
    for tokens in segments:
        if not tokens:
            return False
        executable = tokens[0].rsplit("/", 1)[-1]
        if executable == "cd":
            continue
        if executable in {
            "cat",
            "sed",
            "head",
            "tail",
            "nl",
            "grep",
            "rg",
            "egrep",
            "fgrep",
            "findstr",
        }:
            saw_read = True
            continue
        if executable == "git" and len(tokens) > 1 and tokens[1] in {"diff", "show"}:
            saw_read = True
            continue
        return False
    return saw_read


XML_OBSERVATION_RE = re.compile(
    r"^\s*<returncode>(-?\d+)</returncode>\s*<output>\s*(.*?)\s*</output>\s*$", re.S
)
BRACKET_RETURN_CODE_RE = re.compile(
    r"\[(?:The command completed|Command finished) with exit code (-?\d+)\.?\]", re.I
)
READ_RESULT_PROSE_RE = re.compile(r"^\s*Here(?:'|’)s the result of running\s+`", re.I)


def _observation_envelope(observation: str) -> tuple[str, int | None]:
    """Return conservative provenance kind and an explicit return code if present."""
    match = XML_OBSERVATION_RE.fullmatch(observation)
    if match:
        return "xml_output", int(match.group(1))
    codes = {int(value) for value in BRACKET_RETURN_CODE_RE.findall(observation)}
    if len(codes) == 1:
        return "bracket_output", codes.pop()
    if READ_RESULT_PROSE_RE.match(observation):
        return "read_result_prose", None
    return "unknown", None


ENV_RE = re.compile(
    r"(?:\bno module named\b|\bcommand not found\b|\bnot found\b|\bno such file\b|"
    r"\bpermission denied\b|\battributeerror\b|\bimporterror\b|"
    r"\bmodulenotfounderror\b|\btimed out\b|\btraceback\b)",
    re.I,
)
READ_CODE_RE = re.compile(
    r"(?:\b(?:cat|sed|head|tail|rg|grep|findstr)\b.*\.(?:py|rs|js|ts|go|java|c|h)\b|git\s+(?:diff|show))",
    re.I,
)
DEPENDENCY_RE = re.compile(
    r"(?:\b(?:pip|conda|npm|pnpm|yarn|cargo)\s+(?:install|add|update)|venv|virtualenv|dependency)",
    re.I,
)
VERIFY_RE = re.compile(
    r"(?:pytest|unittest|cargo\s+test|npm\s+test|go\s+test|git\s+diff|reproduc|verify|check)", re.I
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _excerpt(text: str) -> str:
    text = " ".join(text.split())
    return text[:EXCERPT_CHARS]


def _evidence(
    flag: str, reason: str, confidence: str, turn: int, command: str = "", observation: str = ""
) -> dict[str, Any]:
    return {
        "flag": flag,
        "reason": reason,
        "confidence": confidence,
        "turn_index": turn,
        "command_excerpt": _excerpt(command),
        "observation_excerpt": _excerpt(observation),
    }


def validate_row(row: Any, expected_split: str | None = None) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def bad(code: str, reason: str, turn: int = -1) -> None:
        errors.append(_evidence(code, reason, "certain", turn))

    if isinstance(row, InvalidJSON):
        bad("input_parse_error", row.reason)
        return errors
    if not isinstance(row, dict):
        bad("schema_invalid", "row is not an object")
        return errors
    messages = row.get("messages")
    boundary = row.get("completion_start")
    if not isinstance(messages, list) or not messages:
        bad("schema_invalid", "messages must be a non-empty list")
        return errors
    if (
        not isinstance(boundary, int)
        or isinstance(boundary, bool)
        or not 0 <= boundary < len(messages)
    ):
        bad("completion_boundary_invalid", "completion_start must index an existing message")
        boundary = len(messages)
    if not isinstance(row.get("trajectory_id"), str) or not row.get("trajectory_id"):
        bad("trajectory_id_invalid", "trajectory_id must be a non-empty string")
    split = row.get("split")
    if not isinstance(split, str) or not split:
        bad("split_provenance_invalid", "split must be a non-empty string")
    elif expected_split and split != expected_split:
        bad(
            "split_provenance_invalid",
            f"row split {split!r} does not match input split {expected_split!r}",
        )
    supervised = 0
    for index, msg in enumerate(messages):
        if (
            not isinstance(msg, dict)
            or not isinstance(msg.get("role"), str)
            or not isinstance(msg.get("content"), str)
        ):
            bad("schema_invalid", "each message needs string role and content", index)
            continue
        if msg["role"] not in SUPPORTED_ROLES:
            bad("role_invalid", "unsupported message role", index)
        loss = msg.get("loss")
        if type(loss) is not bool:
            bad("loss_flag_invalid", "message loss must be boolean", index)
            continue
        expected = msg["role"] == "assistant" and index >= boundary
        if loss != expected:
            bad("loss_mask_mismatch", f"loss must be {expected} for this role and boundary", index)
        supervised += int(loss)
    if supervised == 0:
        bad("loss_mask_mismatch", "row has no supervised assistant continuation")
    return errors


def analyze_row(row: Any, expected_split: str | None = None) -> dict[str, Any]:
    invalid = validate_row(row, expected_split)
    row = row if isinstance(row, dict) else {}
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    boundary = row.get("completion_start") if not invalid else len(messages)
    assistant_turns = [
        m["content"]
        for i, m in enumerate(messages)
        if i >= boundary
        and isinstance(m, dict)
        and m.get("role") == "assistant"
        and isinstance(m.get("content"), str)
    ]
    verdict = loop_verdict(assistant_turns)
    policy: list[dict[str, Any]] = []
    if verdict.looped:
        policy.append(
            _evidence(
                "canonical_loop",
                "; ".join(verdict.reasons),
                "certain",
                boundary,
                verdict.commands[0].command if verdict.commands else "",
            )
        )

    quality: list[dict[str, Any]] = []
    # One entry per assistant turn, never one per code block. All blocks share one observation.
    pairs: list[tuple[int, str, str]] = []
    for i in range(max(0, boundary), len(messages)):
        msg = messages[i]
        if (
            not isinstance(msg, dict)
            or msg.get("role") != "assistant"
            or not isinstance(msg.get("content"), str)
        ):
            continue
        observation = ""
        if (
            i + 1 < len(messages)
            and isinstance(messages[i + 1], dict)
            and messages[i + 1].get("role") in {"user", "tool"}
        ):
            observation = str(messages[i + 1].get("content", ""))
        pairs.append((i, "\n".join(commands_of([msg["content"]])), observation))

    cmds = [command for _, command, _ in pairs if command]
    run = 1
    for j in range(1, len(pairs)):
        if pairs[j][1] and pairs[j][1] == pairs[j - 1][1]:
            run += 1
            if run == 5 and not verdict.looped:
                i, command, obs = pairs[j]
                quality.append(
                    _evidence(
                        "near_loop",
                        "same action across 5 assistant turns; canonical gate passes",
                        "high",
                        i,
                        command,
                        obs,
                    )
                )
        else:
            run = 1
    for j in range(1, len(pairs)):
        _, previous, prior_obs = pairs[j - 1]
        i, command, obs = pairs[j]
        if (
            command
            and command == previous
            and prior_obs.strip()
            and prior_obs.strip() == obs.strip()
        ):
            quality.append(
                _evidence(
                    "stagnation_retry",
                    "same action and full observation across turns; outer whitespace ignored",
                    "medium",
                    i,
                    command,
                    obs,
                )
            )
            break

    first_code = next((i for i, c, _ in pairs if READ_CODE_RE.search(c)), None)
    early_dependency = [
        (i, c, o)
        for i, c, o in pairs
        if DEPENDENCY_RE.search(c) and (first_code is None or i < first_code)
    ]
    if len(early_dependency) >= 3:
        i, c, o = early_dependency[2]
        quality.append(
            _evidence(
                "environment_detour",
                "3+ runtime-pattern turns before a code-like read; relevance unknown",
                "medium",
                i,
                c,
                o,
            )
        )

    env_pairs: list[tuple[int, str, str]] = []
    for i, command, obs in pairs:
        if not ENV_RE.search(obs):
            continue
        envelope, return_code = _observation_envelope(obs)
        read_command = _read_output_command(command)
        if read_command and (
            envelope == "read_result_prose"
            or envelope in {"xml_output", "bracket_output"}
            and return_code == 0
        ):
            quality.append(
                _evidence(
                    "error_text_in_read_output",
                    "error-shaped text occurs in a recognized read/search result, "
                    "not a runtime failure",
                    "high",
                    i,
                    command,
                    obs,
                )
            )
        elif read_command and envelope == "unknown":
            quality.append(
                _evidence(
                    "ambiguous_error_text",
                    "read/search observation contains error-shaped text but has unknown provenance",
                    "low",
                    i,
                    command,
                    obs,
                )
            )
        else:
            env_pairs.append((i, command, obs))
    if env_pairs:
        i, c, o = env_pairs[0]
        quality.append(
            _evidence(
                "environment_error",
                "observation contains an environment/runtime error pattern",
                "medium",
                i,
                c,
                o,
            )
        )
        later = next(((ni, nc, no) for ni, nc, no in pairs if ni > i and nc and nc != c), None)
        if later:
            quality.append(
                _evidence(
                    "action_change_after_error",
                    "action changes after an error; recovery/progress remains unknown",
                    "medium",
                    later[0],
                    later[1],
                    later[2],
                )
            )

    # A search returning error-shaped source text is not command/output mismatch.
    # Truly unrelated/echoed observations need semantic linkage and remain deferred.
    if any(VERIFY_RE.search(c) for c in cmds):
        i, c, o = next((p for p in pairs if VERIFY_RE.search(p[1])))
        quality.append(
            _evidence(
                "verification_signal",
                "action text contains a verification/test/diff pattern; outcome is not inferred",
                "low",
                i,
                c,
                o,
            )
        )

    # Deduplicate evidence generated from repeated commands while retaining turn-local support.
    seen: set[tuple[str, int]] = set()
    quality = [
        e
        for e in quality
        if not ((e["flag"], e["turn_index"]) in seen or seen.add((e["flag"], e["turn_index"])))
    ]
    informational = [e for e in quality if e["flag"] in INFORMATIONAL_FLAGS]
    risks = [e for e in quality if e["flag"] not in INFORMATIONAL_FLAGS]
    disposition = "invalid" if invalid or policy else ("review" if risks else "eligible")
    metadata = {
        key: row.get(key)
        for key in (
            "trajectory_id",
            "split",
            "source",
            "sample_phase",
            "origin",
            "sample_id",
            "task_group",
        )
    }
    return {
        **metadata,
        "completion_start": row.get("completion_start"),
        "content_hash": _hash(messages) if not invalid else None,
        "policy_flags": policy,
        "quality_flags": quality,
        "informational_flags": informational,
        "risk_flags": risks,
        "review_reasons": sorted({e["flag"] for e in risks}),
        "invalid_reasons": invalid,
        "suggested_disposition": disposition,
        "canonical_loop_stats": {
            "n_commands": verdict.n_cmds,
            "duplicate_command_ratio": verdict.dup_cmd_ratio,
            "max_consecutive_command_run": verdict.max_cmd_run,
        },
    }


def deterministic_rows(path: Path, sample_size: int) -> Iterable[tuple[int, Any]]:
    if sample_size == 0:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                yield line_no, _decode_row(line)
        return
    heap: list[tuple[int, int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            score = int.from_bytes(
                hashlib.sha256(f"{line_no}:".encode() + line.encode("utf-8")).digest(), "big"
            )
            item = (-score, -line_no, line)
            if len(heap) < sample_size:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    for _, neg_line, line in sorted(heap, key=lambda x: -x[1]):
        yield -neg_line, _decode_row(line)


def scan(
    input_path: Path,
    output_dir: Path,
    sample_size: int,
    split: str | None,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    input_hash = _file_hash(input_path)
    plan_counts: Counter[str] = Counter()
    if plan_path:
        with plan_path.open(encoding="utf-8") as handle:
            for line in handle:
                plan = json.loads(line)
                plan_counts[str(plan.get("trajectory_id"))] += 1
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    dispositions: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    invalid_flags: Counter[str] = Counter()
    subgroup = {field: Counter() for field in ["source", "sample_phase", "origin", "split"]}
    count = loops = 0
    # Disk spool avoids retaining raw rows or evidence manifests in RAM in full mode.
    # Only plan counts and compact content-hash/provenance indexes grow with row count.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", newline="\n") as spool:
        for line_no, row in deterministic_rows(input_path, sample_size):
            result = analyze_row(row, split)
            result["input_line"] = line_no
            tid = str(result.get("trajectory_id"))
            result["sampling_plan_slots"] = plan_counts.get(tid, 0) if plan_path else None
            if result["content_hash"] is not None:
                hashes[result["content_hash"]].append(tid)
            spool.write(_stable_json(result) + "\n")
            count += 1
            loops += bool(result["policy_flags"])
            dispositions[result["suggested_disposition"]] += 1
            flags.update({e["flag"] for e in result["quality_flags"]})
            invalid_flags.update({e["flag"] for e in result["invalid_reasons"]})
            for field, counts in subgroup.items():
                counts[str(result.get(field))] += 1
        if _file_hash(input_path) != input_hash:
            raise RuntimeError("input changed during scan; output not published")
        output_dir.mkdir(parents=True)
        spool.seek(0)
        with (output_dir / "review-manifest.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as manifest:
            for line in spool:
                result = json.loads(line)
                group = hashes.get(result["content_hash"], [])
                result["content_duplicate"] = len(group) > 1
                result["content_duplicate_trajectory_ids"] = (
                    sorted(set(group)) if len(group) > 1 else []
                )
                manifest.write(_stable_json(result) + "\n")
    summary = {
        "scanner_version": SCANNER_VERSION,
        "input": str(input_path.resolve()),
        "input_sha256": input_hash,
        "selection": {
            "method": "full" if sample_size == 0 else "sha256-smallest-over-lines",
            "requested_rows": sample_size,
            "selected_rows": count,
            "not_a_corpus_estimate": sample_size != 0,
        },
        "counts": {
            "rows": count,
            "dispositions": dict(sorted(dispositions.items())),
            "quality_flags": dict(sorted(flags.items())),
            "risk_flags": {k: v for k, v in sorted(flags.items()) if k not in INFORMATIONAL_FLAGS},
            "informational_flags": {
                k: v for k, v in sorted(flags.items()) if k in INFORMATIONAL_FLAGS
            },
            "invalid_flags": dict(sorted(invalid_flags.items())),
            "canonical_loop_rows": loops,
            "content_duplicate_groups": sum(len(v) > 1 for v in hashes.values()),
        },
        "subgroups": {field: dict(sorted(counts.items())) for field, counts in subgroup.items()},
        "input_integrity": "sha256 unchanged before/after scan",
        "invalid_line_policy": "blank/invalid JSON and non-objects reported invalid, not skipped",
        "sampling_plan": {
            "path": str(plan_path.resolve()),
            "note": "plan repetitions are intentional epoch slots, not raw-row duplicates",
        }
        if plan_path
        else None,
        "limitations": [
            "Quality flags are deterministic review cues, not semantic correctness judgments.",
            "Exit code 0 is not a passing-test label; edit-like commands are not fixes.",
            "Exact content dedup covers selected rows only; provenance is retained.",
            "Eligible means no detected review risk, not certified good training data.",
            "Environment patterns may be expected reproduction failures or source text.",
            "Action change is not recovery; a code-like read need not be task-relevant.",
            "Split strings and content hashes do not certify task/context-group separation.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=256,
        help="0 scans all rows; positive values select deterministically",
    )
    parser.add_argument("--split", choices=("train", "dev"))
    parser.add_argument("--sampling-plan", type=Path)
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative")
    summary = scan(args.input, args.output, args.sample_size, args.split, args.sampling_plan)
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
