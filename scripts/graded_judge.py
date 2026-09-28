"""Production-parity graded judging of one trajectory document, for local runners.

Mirrors one side of `judge_api._score_samples` at the merged upstream revision (graded A-T judge,
commit 60d002b): a truncated, abandoned, reserved-token or looped document scores 0 without a
judge call; otherwise `_judge_side` asks the judge with `logprobs` + `top_logprobs: 20` on the
logprob provider pins (ambient, then alibaba) and each question scores the expectation of the
letter distribution. The amputated-thinking multiplier is applied last, as production does.

Everything is delegated to the service's own functions, so a later upstream change to the judge
reaches the local runners through the next merge instead of drifting from a copy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from albedo_config import JudgeSettings  # noqa: E402
from albedo_config.models import JUDGE_MODELS  # noqa: E402
from albedo_eval_service.judge_api import (  # noqa: E402
    _corrupted_side,
    _judge_side,
    _looped_side,
    reserved_token_leak,
)
from albedo_eval_service.judge_core import (  # noqa: E402
    AMPUTATED_THINKING_MULTIPLIER,
    amputated_thinking,
    response_score,
)
from albedo_eval_service.judge_llm_client import JudgeLLMClient  # noqa: E402
from albedo_eval_service.shared.loop_check import loop_verdict_for_document  # noqa: E402
from albedo_eval_service.shared.observation_format import is_abandoned, is_truncated  # noqa: E402
from albedo_eval_service.shared.verdict_levels import SCORING_MODE  # noqa: E402


async def judge_document(
    *,
    client: JudgeLLMClient,
    settings: JudgeSettings,
    document: str,
    questions: list[dict[str, Any]],
    side: str = "candidate",
    judge_models: list[str] | None = None,
) -> dict[str, Any]:
    """Score one trajectory document the way production scores one side of a sample."""
    models = list(judge_models or JUDGE_MODELS)
    if is_truncated(document):
        per_judge, records = _corrupted_side(
            side=side, questions=questions, judge_models=models,
            reason="output truncated mid-generation")
    elif is_abandoned(document):
        per_judge, records = _corrupted_side(
            side=side, questions=questions, judge_models=models,
            reason="turn unusable after repeated attempts; the benchmark abandons here")
    elif leak := reserved_token_leak(document):
        per_judge, records = _corrupted_side(
            side=side, questions=questions, judge_models=models,
            reason=f"reserved template token in output: {leak}")
    elif (looping := loop_verdict_for_document(document)).looped:
        per_judge, records = _looped_side(
            side=side, questions=questions, judge_models=models, verdict=looping)
    else:
        per_judge, records = await _judge_side(
            client=client, settings=settings, side=side, response_text=document,
            questions=questions, judge_models=models,
            repeats=int(getattr(settings, "judge_repeats", 1) or 1),
        )
    score = response_score(per_judge, questions)
    amputated = amputated_thinking(document)
    if amputated and score is not None:
        score = round(score * AMPUTATED_THINKING_MULTIPLIER, 6)
    first = records[0] if records else {}
    return {
        "scoring_mode": SCORING_MODE,
        "score": score,
        "parse_ok": all(r.get("parse_ok") for r in records) and score is not None,
        "amputated_thinking": amputated,
        "provider": first.get("provider"),
        "answers": first.get("answers") or {},
        "scores": first.get("scores") or {},
        "distributions": first.get("distributions") or {},
        "explanations": first.get("explanations") or {},
        "error": first.get("error"),
        "deterministic_zero": bool(first.get("corrupted") or first.get("looped")),
        "zero_reason": first.get("corruption_reason") or (
            "; ".join(first.get("loop_reasons") or []) if first.get("looped") else None),
        "repeats_held": first.get("repeats_held"),
        "disputed": first.get("disputed"),
    }


TOLERANT_READS = {"rescued": 0, "skipped_chars": 0}


def _align_ignoring_whitespace(raw: str, spanned: str, max_skipped: int) -> str | None:
    """The prefix of `spanned` (concatenated logprob tokens) that covers all of `raw`, when the two
    differ only by at most `max_skipped` whitespace characters; None otherwise."""
    i = j = skipped = 0
    while i < len(raw):
        if j < len(spanned) and raw[i] == spanned[j]:
            i += 1
            j += 1
        elif raw[i].isspace():
            i += 1
            skipped += 1
        elif j < len(spanned) and spanned[j].isspace():
            j += 1
            skipped += 1
        else:
            return None
        if skipped > max_skipped:
            return None
    TOLERANT_READS["skipped_chars"] += skipped
    return spanned[:j]


def enable_whitespace_tolerant_logprobs(max_skipped: int = 4) -> None:
    """Local runners only: accept a logprob reading whose tokens differ from the content by a few
    whitespace characters.

    Production rejects any reading whose concatenated tokens do not reproduce the content
    exactly. GLM-5.2 providers drop the space in sequences like ``cat ./path`` from the token
    stream while keeping it in the content (measured 2026-09-27: `('cat'), ('./', [46, 47])` under
    ``cat ./graphene/...``), so every document whose judge explanation quotes such a command fails
    on every retry. The verdict letters and their distributions are untouched by a missing space;
    this re-reads them over the token-aligned text. Any other mismatch still fails, as upstream."""
    from albedo_eval_service import judge_api
    from albedo_eval_service.shared import verdict_levels

    strict = verdict_levels.read_verdict_logprobs

    def tolerant(raw, entries):
        reading = strict(raw, entries)
        if reading.ok or not entries or not str(reading.error).startswith("logprob tokens do not reproduce"):
            return reading
        spanned = "".join(str(entry.get("token") or "") for entry in entries)
        aligned = _align_ignoring_whitespace(raw or "", spanned, max_skipped)
        if aligned is None:
            return reading
        rescued = strict(aligned, entries)
        if rescued.ok:
            TOLERANT_READS["rescued"] += 1
        return rescued

    judge_api.read_verdict_logprobs = tolerant


def spent_usd(client: JudgeLLMClient) -> float:
    """Every attempt this client paid for, including retries and rejected logprob readings."""
    return sum(float(record.get("cost") or 0.0) for record in client.usage_records)
