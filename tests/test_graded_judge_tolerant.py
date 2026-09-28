"""Whitespace-tolerant logprob reading used by the local graded-judge runners."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import graded_judge  # noqa: E402
from albedo_eval_service import judge_api  # noqa: E402
from albedo_eval_service.shared import verdict_levels  # noqa: E402

RAW = '{"answers": [{"asked": "q_01", "reason": "runs `cat ./x.py`", "verdict": "O"}]}'


def entries_for(text: str) -> list[dict]:
    """One token per character, with a top-logprob distribution on the verdict letter."""
    out = []
    for ch in text:
        entry = {"token": ch, "logprob": 0.0}
        if ch == "O":
            entry["top_logprobs"] = [{"token": "O", "logprob": math.log(0.6)},
                                     {"token": "T", "logprob": math.log(0.4)}]
        out.append(entry)
    return out


def test_strict_reader_rejects_a_dropped_space():
    tokens = entries_for(RAW.replace("cat ./", "cat./"))
    assert not verdict_levels.read_verdict_logprobs(RAW, tokens).ok


def test_tolerant_reader_recovers_the_same_scores():
    original = judge_api.read_verdict_logprobs
    try:
        graded_judge.enable_whitespace_tolerant_logprobs(max_skipped=4)
        tolerant = judge_api.read_verdict_logprobs(RAW, entries_for(RAW.replace("cat ./", "cat./")))
        exact = verdict_levels.read_verdict_logprobs(RAW, entries_for(RAW))
        assert tolerant.ok and tolerant.scores == exact.scores
        assert math.isclose(tolerant.scores["q_01"], 0.6 * 0.70 + 0.4 * 1.0)
    finally:
        judge_api.read_verdict_logprobs = original


def test_tolerant_reader_still_rejects_other_mismatches():
    original = judge_api.read_verdict_logprobs
    try:
        graded_judge.enable_whitespace_tolerant_logprobs(max_skipped=4)
        assert not judge_api.read_verdict_logprobs(RAW, entries_for(RAW.replace("cat", "cut"))).ok
        too_many = RAW.replace("runs `cat ./x.py`", "runs      `cat ./x.py`")
        assert not judge_api.read_verdict_logprobs(too_many, entries_for(RAW.replace(" ", ""))).ok
    finally:
        judge_api.read_verdict_logprobs = original
