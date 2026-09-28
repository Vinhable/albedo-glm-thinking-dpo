#!/usr/bin/env python3
"""Corrected per-turn behavior detection for trajectory segmentation.

Three defects in the v1 path are fixed here.

1. Fence pairing. `edit_detection.EDIT_BLOCK_RE` is ```` ```(?:bash|sh)?[ \t]*\n(.*?)``` ````.
   When a turn shows a ```` ```python ```` block before its ```` ```bash ```` command, the opening
   python fence does not match, the engine then matches the python block's *closing* fence as an
   opener, and captures the prose between the two blocks instead of the command. The real command
   is never inspected, so the turn is never seen as an edit. Blocks are now paired by walking
   fence markers in order, exactly as a Markdown reader would.

2. Heredoc bodies. Detecting edits on whitespace-flattened command text makes ordinary comparisons
   inside a heredoc body (`if x > 1.5:`) look like a shell redirect into `1.5`. Command lines are
   now extracted heredoc-aware, and the body of a heredoc is never scanned for edit syntax.

3. Scratch files vs source. `REPO_EDIT_RE` treats every write outside /tmp and /dev as a repository
   edit, so creating `reproduce_issue.py` was labelled `edit`. In this corpus those reproduction
   and debug scripts are the single most common write target, and labelling them `edit` also
   flipped every later test turn from `reproduce_test` to `verify_test`. A write is a source edit
   only when it lands on a file that already existed, which is decided from the paths named in the
   prompt and in the environment observations seen so far rather than from the file's name. Naming
   heuristics were tried first and rejected: they misclassify real repository files such as
   `tests/test_core.py` or a Go `main.go`. Writes to paths the model itself invented are scratch
   work, and belong to reproducing or verifying instead.

The label vocabulary is unchanged: inspect_diagnose, reason_plan, reproduce_test, edit,
verify_inspect, verify_test, finalize.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import audit_rollouts as rollout_audit  # noqa: E402

SHELL_LANGUAGES = {"bash", "sh", "shell"}
TAG_BLOCK_RE = re.compile(r"<([a-z_]*bash[a-z_]*)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(?P<delim>[A-Za-z_][A-Za-z0-9_]*)['\"]?")

# In-place modification of something that already exists. These never create scratch files.
IN_PLACE_RE = re.compile(
    r"\bsed\s+-i"
    r"|\bstr_replace\b"
    r"|\bgit\s+apply\b|\bapplypatch\b|\bpatch\s+-p\d"
    r"|\b(?:perl|ruby)\s+-[a-zA-Z]*i\b"
    r"|\bfileinput\.input\([^)]*inplace",
    re.IGNORECASE,
)
# Any write of file content, in place or not.
WRITE_RE = re.compile(
    r"\bsed\s+-i|\bstr_replace\b|\bgit\s+apply\b|\bapplypatch\b|\bpatch\s+-p\d"
    r"|\b(?:perl|ruby)\s+-[a-zA-Z]*i\b|\bfileinput\.input\([^)]*inplace"
    r"|\bcat\s*>>?|\btee\s+(?!-a\b)"
    r"|\.write_text\s*\(|\.write_bytes\s*\(|\.writelines\s*\("
    r"|\bopen\s*\([^)]*['\"][wa]\+?['\"]"
    r"|\bshutil\.(?:copy|copyfile|copy2|move)\s*\(|\bos\.(?:replace|rename)\s*\("
    r"|\bcp\s+[\w./~-]+\s+[\w./~-]+|\bmv\s+[\w./~-]+\s+[\w./~-]+"
    r"|(?<![-=<>0-9&])>>?\s*(?!/dev/|&)[\w./~$-]*[\w-]\.[A-Za-z]\w*",
    re.IGNORECASE,
)
# A plausible file extension: lowercase and short. Keeps attribute access such as `t.Any` or
# `best_match.matched_slice` from being read as a write target.
FILE_EXTENSION_RE = re.compile(r"\.[a-z0-9]{1,5}$")
# Where the write lands. Used only to tell a scratch artifact from a source file.
TARGET_RE = re.compile(
    r"(?:>>?|\btee(?:\s+-a)?|\bcat\s*>>?)\s+([\w./~$-]*[\w-]\.[A-Za-z]\w*)"
    r"|\bsed\s+-i[^\s]*\s+(?:-e\s+\S+\s+)*([\w./~$-]*[\w-]\.[A-Za-z]\w*)"
    r"|\b(?:open|write_text|write_bytes)\s*\(\s*['\"]([\w./~$-]*[\w-]\.[A-Za-z]\w*)",
    re.IGNORECASE,
)
SCRATCH_PATH_RE = re.compile(r"^(?:/tmp/|/dev/|~/|/root/)", re.IGNORECASE)
# Any path-like token, used to learn which files the environment has already shown to exist.
PATH_TOKEN_RE = re.compile(r"[\w./~$-]*[\w-]\.[A-Za-z]\w*")
# The Albedo submission artifact is never a repository source file.
SUBMISSION_ARTIFACT = "patch.txt"
# Running a scratch script counts as reproducing or verifying, like running a test runner does.
RUN_SCRATCH_RE = re.compile(
    r"\b(?:python3?|node|go\s+run|ruby|php|cargo\s+run|java|bash|sh)\s+[\w./~$-]*"
    r"(?:reproduce|repro|test_|debug|check_|verify_|minimal|example|demo|poc|issue|bug|problem)"
    r"[\w./~$-]*\.[A-Za-z]\w*",
    re.IGNORECASE,
)
SUBMIT_TOKENS = ("albedo submit", "albedo activate", "submit_marker")


FENCE_RE = re.compile(r"```([^\n`]*)")


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Pair fence markers in document order and return (language, raw body) for each block.

    Markers are scanned wherever they appear, not only at the start of a line: these turns
    routinely open a block immediately after a closing `</think>` on the same line.
    """
    source = text or ""
    blocks: list[tuple[str, str]] = []
    opener: re.Match[str] | None = None
    for marker in FENCE_RE.finditer(source):
        if opener is None:
            opener = marker
            continue
        newline = source.find("\n", opener.end())
        start = opener.end() if newline == -1 else newline + 1
        blocks.append((opener.group(1).strip().lower(), source[start:marker.start()]))
        opener = None
    return blocks


UNPAIRED_SHELL_RE = re.compile(
    r"```(?:bash|sh|shell)[ \t]*\r?\n(.*?)(?:```|\Z)", re.IGNORECASE | re.DOTALL
)


def shell_blocks(text: str) -> list[str]:
    """Raw bodies of the blocks a turn would actually execute, newlines preserved."""
    blocks = [body for language, body in fenced_blocks(text) if language in SHELL_LANGUAGES]
    if not blocks:
        # A turn that leaves an earlier code block unclosed has an odd number of fence markers,
        # which shifts strict pairing and hides its shell block. Fall back to locating the shell
        # opener directly, still on raw text so heredoc handling below stays correct.
        blocks = [match.group(1) for match in UNPAIRED_SHELL_RE.finditer(text or "")]
    blocks.extend(match.group(2) for match in TAG_BLOCK_RE.finditer(text or ""))
    return blocks


def command_lines(block: str) -> list[str]:
    """Lines of a shell block with heredoc bodies removed."""
    lines = block.split("\n")
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        kept.append(line)
        match = HEREDOC_RE.search(line)
        if match:
            delimiter = match.group("delim")
            index += 1
            while index < len(lines) and lines[index].strip() != delimiter:
                index += 1
        index += 1
    return kept


def turn_commands(text: str) -> list[str]:
    """Executable command lines of a turn, heredoc bodies excluded."""
    commands: list[str] = []
    for block in shell_blocks(text):
        commands.extend(line for line in command_lines(block) if line.strip())
    return commands


def write_targets(command: str) -> list[str]:
    targets: list[str] = []
    for match in TARGET_RE.finditer(command):
        target = next((group for group in match.groups() if group), None)
        if target and FILE_EXTENSION_RE.search(target):
            targets.append(target)
    return targets


def normalise(path: str) -> str:
    return path.lstrip("./").rstrip("/")


class TrajectoryClassifier:
    """Labels the turns of one continuation, carrying the file knowledge between them.

    A write counts as a source edit when the path was already visible in the task prompt or in an
    environment observation, and as scratch work when the model invented the path itself.
    """

    def __init__(self, prefix: list[dict] | None = None) -> None:
        self.known: set[str] = set()
        self.created: set[str] = set()
        self.edit_seen = False
        if prefix:
            self.warm_up(prefix)

    def warm_up(self, messages: list[dict]) -> None:
        """Replay the prompt prefix so the continuation starts from the real trajectory state.

        Two things carry over and both matter. Files the model created earlier are already known
        to be its own scratch work, and an edit already made before the cut point means the
        continuation is verifying rather than reproducing. v1 started every continuation from a
        blank state, which mislabelled the `pre_edit` and `at_edit` phases.
        """
        for message in messages:
            content = str(message.get("content") or "")
            if str(message.get("role") or "").lower() == "assistant":
                self.classify(content, final_turn=False)
            else:
                self.observe(content)

    def observe(self, text: str) -> None:
        for token in PATH_TOKEN_RE.findall(text or ""):
            path = normalise(token)
            if path and path not in self.created:
                self.known.add(path)
                self.known.add(path.rsplit("/", 1)[-1])

    def _target_kind(self, target: str) -> str:
        path = normalise(target)
        base = path.rsplit("/", 1)[-1]
        if SCRATCH_PATH_RE.match(target) or base == SUBMISSION_ARTIFACT:
            return "scratch"
        if path in self.created or base in self.created:
            return "scratch"
        if path in self.known or base in self.known:
            return "source"
        return "scratch"

    def _classify_write(self, command: str) -> str:
        """'' when the command writes nothing, else 'source' or 'scratch'."""
        if not WRITE_RE.search(command):
            return ""
        targets = write_targets(command)
        if not targets:
            # A write whose path we cannot resolve. In-place syntax needs an existing file;
            # anything else is ambiguous, and an unresolved write is more often a real edit.
            return "source"
        kinds = {self._target_kind(target) for target in targets}
        if "source" in kinds:
            return "source"
        for target in targets:
            path = normalise(target)
            self.created.add(path)
            self.created.add(path.rsplit("/", 1)[-1])
        return "scratch"

    def classify(self, text: str, *, final_turn: bool) -> tuple[str, dict[str, bool]]:
        commands = turn_commands(text)
        # An in-place modification implies the file exists even if we never saw it named.
        for command in commands:
            if IN_PLACE_RE.search(command):
                for target in write_targets(command):
                    path = normalise(target)
                    if path not in self.created and path.rsplit("/", 1)[-1] not in self.created:
                        self.known.add(path)
                        self.known.add(path.rsplit("/", 1)[-1])

        writes = [self._classify_write(command) for command in commands]
        source_edit = "source" in writes
        scratch_write = "scratch" in writes
        ran_test = any(rollout_audit.TEST_RE.search(command) for command in commands)
        ran_scratch = any(RUN_SCRATCH_RE.search(command) for command in commands)
        diffed = any(rollout_audit.GIT_DIFF_RE.search(command) for command in commands)
        targeted = any(rollout_audit._is_targeted_inspect(command) for command in commands)
        submitted = any(token in (text or "").lower() for token in SUBMIT_TOKENS)

        flags = {
            "has_edit": source_edit,
            "has_scratch_write": scratch_write,
            "has_test": ran_test or ran_scratch,
            "has_targeted_inspect": targeted,
            "has_diff": diffed,
            "has_submit": submitted,
        }

        if submitted or (final_turn and not commands):
            label = "finalize"
        elif source_edit:
            label = "edit"
        elif ran_test or ran_scratch or scratch_write:
            label = "verify_test" if self.edit_seen else "reproduce_test"
        elif diffed or targeted or commands:
            label = "verify_inspect" if self.edit_seen else "inspect_diagnose"
        else:
            label = "reason_plan"

        self.edit_seen |= source_edit
        return label, flags
