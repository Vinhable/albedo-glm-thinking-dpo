#!/usr/bin/env python3
"""Pilot: GLM-5.2 regenerates weak-King tasks with its reasoning on, in the candidate format.

Why
    Teacher SFT on GLM references failed because those references carry no reasoning (the King
    learnt to think less), and King-vs-challenger pairs carry no signal. Here GLM-5.2 writes each
    turn with its own reasoning, stored exactly as a Qwen3.6 candidate turn:
    `reasoning` + `\\n</think>\\n\\n` + `action`. The tasks are those where the King 127 scored low
    and GLM's own reference scored high, so a GLM rollout should beat the King's by a clear margin.

Stages
    select  (no API)  pick `--count` tasks from the crawled graded-era runs:
                      King mean <= --king-max over its two rollouts, both rollouts <= --king-rollout-max,
                      every GLM reference run >= --glm-min-each and the best >= --glm-min-best
                      (scores from HF `dendriteholdings/albedo`), spread over sources and phases.
    run     (paid)    per task: GLM turns + the production simulator (no repo grounding, see below),
                      then production's graded judge on the task's own checklist. With
                      --rejudge-king, the King's first rollout is judged too, to check that the local
                      judge reproduces production's score (the comparison is only fair if it does).

Rendering matches production: a later turn's context holds only the earlier actions (the chat
template drops past reasoning), observations come from the simulator, and a turn stops the
trajectory on submission, truncation or the task's horizon.

Grounding: production grounds the simulator in the real repository through the repo-context
service. Pass `--repo-context-url http://127.0.0.1:8093` with that service running locally (see
`docs/LOCAL_GROUNDING.md`); without it observations are ungrounded, unlike the King's.

    py -3 scripts/pilot_glm_thinking_teacher.py select --out E:/albedo-storage-temp/glm-thinking-pilot-20260927
    py -3 scripts/pilot_glm_thinking_teacher.py run --out E:/albedo-storage-temp/glm-thinking-pilot-20260927
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from albedo_config import JudgeSettings  # noqa: E402
from albedo_eval_service.judge_api import (  # noqa: E402
    ObservationSimulationService,
    SimulateObservationRequest,
)
from albedo_eval_service.judge_llm_client import JudgeLLMClient  # noqa: E402
from albedo_eval_service.repo_context_client import RepoContextClient  # noqa: E402
from albedo_eval_service.remote.generation import format_scored_trajectory  # noqa: E402
from albedo_eval_service.shared.observation_format import (  # noqa: E402
    detect_format,
    first_bash_block,
    is_abandoned,
    truncation_notice,
    wrap,
)
from albedo_eval_service.shared.submit_protocol import is_exact_submission  # noqa: E402
from albedo_eval_service.simulator.prompt_simulator import COMPLETE_MARKER  # noqa: E402
from build_gen_vinhable import rollouts, run_index  # noqa: E402
from graded_judge import TOLERANT_READS, enable_whitespace_tolerant_logprobs, judge_document, spent_usd  # noqa: E402

OPENROUTER = "https://openrouter.ai/api/v1"
GLM_MODEL = "z-ai/glm-5.2"
TURN_TOKEN_LIMIT = 4096  # production's max_new_tokens per candidate turn

# Steers GLM at generation time only. v2 showed GLM skipping its reasoning on ~35% of agent turns
# even at medium effort; the King thinks on 99.7% of turns (median ~220 tokens, p99 ~2,300).
THINKING_PROTOCOL = """\
REASONING PROTOCOL (strict, applies to every single turn):
- You MUST reason privately before every response. Never skip reasoning, not even for simple steps
  such as listing files, reading a file, or re-running a command.
- In that reasoning: state what the latest observation shows, what it means for the task, and why
  the next command is the right one.
- Keep the reasoning focused and concise: about 50 to 300 words, never more than 1,500 words.
- After reasoning, write the visible response exactly in the format the instructions above require,
  with exactly one bash code block."""

# v4 (probe "nudge5" on Baidu): reasoning on every turn, but inside real trajectories it ran to a
# mean of 663 Qwen tokens (King 299 on the same tasks) and cut 3/15 rollouts at 4,096.
REASONING_NUDGE_V4 = (
    "\n\n[Reasoning requirement for this turn, including routine steps like verifying an edit: before "
    "replying, reason privately through all five points below, writing at least four full sentences "
    "for each point (about 400 to 700 words in total, never more than 1,500): "
    "1) Observation: what exactly this output shows, line by line where it matters. "
    "2) Evidence: what it confirms or rules out about the bug and the fix so far. "
    "3) Open questions: the hypotheses still open and what could still be wrong. "
    "4) Options: at least two candidate next commands and their trade-offs. "
    "5) Decision: why the chosen command is the best next step. "
    "Then answer in the required format.]")

# v5: the same five points, one notch lighter, aiming at ~300-400 Qwen tokens and nothing near 4,096.
REASONING_NUDGE = REASONING_NUDGE_V4.replace(
    "at least four full sentences", "at least two full sentences").replace(
    "about 400 to 700 words in total, never more than 1,500", "about 200 to 400 words in total, never more than 700")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


# --------------------------------------------------------------------------------------- select


def reference_scores(hf_root: Path) -> dict[tuple[str, str], list[float]]:
    """(eval_run_id, sample_id) -> the GLM reference runs' own scores."""
    import pyarrow.parquet as pq

    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in sorted((hf_root / "data" / "glm_5_2").glob("*.parquet")):
        table = pq.read_table(path, columns=["sample_id", "eval_run_id", "score"])
        for row in table.to_pylist():
            if row["score"] is not None:
                scores[(row["eval_run_id"], row["sample_id"])].append(float(row["score"]))
    return scores


def assistant_count(turns: list[dict[str, Any]]) -> int:
    return sum(1 for t in turns if t.get("role") == "assistant")


def select(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    glm = reference_scores(args.hf_root)
    held_out = tuple(args.held_out)
    counts: Counter[str] = Counter()
    candidates = []
    for run in run_index([args.crawl_root]):
        if run["eval_run_id"].startswith(held_out):
            continue
        directory = run["directory"]
        submit = {}
        with (directory / "generated-samples.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                submit[str(row["sample_id"]).split("#", 1)[0]] = (row.get("submit_command"), row.get("submit_marker"))
        questions: dict[str, list[dict[str, Any]]] = {}
        with (directory / "scoring-results.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                questions.setdefault(str(row["sample_id"]).split("#", 1)[0], row.get("questions") or [])
        for sid, info, found in rollouts(run, counts):
            if info.get("scoring_mode") != "graded_20":
                counts["skip:not_graded"] += 1
                continue
            kings = [r for r in found if r["side"] == "king"]
            ref = glm.get((run["eval_run_id"], sid), [])
            if len(kings) < 2 or not ref or not questions.get(sid):
                counts["skip:missing_king_or_reference"] += 1
                continue
            king_scores = [r["score"] for r in kings]
            if args.strategy == "weak_king" and (
                    st.mean(king_scores) > args.king_max or max(king_scores) > args.king_rollout_max
                    or max(ref) < args.glm_min_best or min(ref) < args.glm_min_each):
                counts["skip:thresholds"] += 1
                continue
            if sid in args.exclude_samples:
                counts["skip:already_piloted"] += 1
                continue
            command, marker = submit.get(sid, (None, None))
            candidates.append({
                "task_id": f"{run['eval_run_id']}:{sid}",
                "eval_run_id": run["eval_run_id"],
                "sample_id": sid,
                "source": sid.split("/", 1)[0],
                "sample_phase": info.get("sample_phase"),
                "question_mode": info.get("question_mode"),
                "context": kings[0]["context"],
                "submit_command": command or "",
                "submit_marker": marker or "",
                "questions": questions[sid],
                "horizon": max(assistant_count(r["completion"]) for r in kings),
                "king_rollouts": [
                    {"replica": r["replica"], "score": r["score"], "amputated": r["amputated"],
                     "assistant_turns": assistant_count(r["completion"]),
                     "document": format_scored_trajectory(
                         [dict(m) for m in r["context"]]
                         + [{**t, "score_target": True} if t["role"] == "assistant"
                            else {**t, "environment_observation": True} for t in r["completion"]])}
                    for r in kings
                ],
                "king_mean": st.mean(king_scores),
                "glm_reference_scores": ref,
                # robust gaps: the second-best reference run against the better King rollout, so one
                # lucky reference run is not enough
                "gap_glm_over_king": sorted(ref, reverse=True)[min(1, len(ref) - 1)] - max(king_scores),
                "gap_king_over_glm": min(king_scores) - max(ref),
            })
            counts["candidate"] += 1
    if args.strategy == "gap":
        chosen = choose_by_gap(candidates, args, counts)
    else:
        chosen = choose_spread(candidates, args)
    with (args.out / "pilot-input.jsonl").open("w", encoding="utf-8") as handle:
        for c in chosen:
            handle.write(json.dumps(c, ensure_ascii=False) + "\n")
    summary = {
        "strategy": args.strategy, "candidates": len(candidates), "chosen": len(chosen), "counts": dict(counts),
        "chosen_by_role": dict(Counter(c.get("role", "main") for c in chosen)),
        "chosen_by_source": dict(Counter(c["source"] for c in chosen)),
        "chosen_by_phase": dict(Counter(c["sample_phase"] for c in chosen)),
        "chosen_king_mean": st.mean(c["king_mean"] for c in chosen) if chosen else None,
        "chosen_horizon": dict(Counter(c["horizon"] for c in chosen)),
    }
    for role in ("main", "contrast"):
        part = [c for c in chosen if c.get("role", "main") == role]
        if part:
            summary[f"{role}_king_mean"] = round(st.mean(c["king_mean"] for c in part), 3)
            summary[f"{role}_glm_ref_best_mean"] = round(st.mean(max(c["glm_reference_scores"]) for c in part), 3)
            summary[f"{role}_gap_glm_over_king_min"] = round(min(c["gap_glm_over_king"] for c in part), 3)
    (args.out / "select-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def choose_by_gap(candidates: list[dict[str, Any]], args: argparse.Namespace, counts: Counter) -> list[dict[str, Any]]:
    """`--count` tasks: (1 - contrast share) where GLM's reference clearly beat both King rollouts,
    largest robust gap first, split over sample phases like the candidate pool; the rest where
    the King clearly beat every reference run (future King-chosen pairs)."""
    n_contrast = round(args.count * args.contrast_share)
    n_main = args.count - n_contrast
    main_pool = [c for c in candidates if c["gap_glm_over_king"] >= args.min_gap]
    contrast_pool = [c for c in candidates if c["gap_king_over_glm"] >= args.min_gap]
    if len(contrast_pool) < n_contrast:  # top up with the King's next-clearest wins
        extra = sorted((c for c in candidates if c not in contrast_pool and c["gap_king_over_glm"] > 0),
                       key=lambda c: -c["gap_king_over_glm"])
        contrast_pool += extra[: n_contrast - len(contrast_pool)]
    counts["pool:main"] = len(main_pool)
    counts["pool:contrast"] = len(contrast_pool)
    phase_share = Counter(c["sample_phase"] for c in main_pool)
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for phase, n in phase_share.items():
        quota = round(n_main * n / len(main_pool))
        ranked = sorted((c for c in main_pool if c["sample_phase"] == phase), key=lambda c: -c["gap_glm_over_king"])
        taken = 0
        for c in ranked:
            if taken >= quota:
                break
            if c["sample_id"] not in seen:
                chosen.append({**c, "role": "main"})
                seen.add(c["sample_id"])
                taken += 1
    for c in sorted(main_pool, key=lambda c: -c["gap_glm_over_king"]):  # rounding remainder
        if sum(1 for x in chosen if x["role"] == "main") >= n_main:
            break
        if c["sample_id"] not in seen:
            chosen.append({**c, "role": "main"})
            seen.add(c["sample_id"])
    for c in sorted(contrast_pool, key=lambda c: -c["gap_king_over_glm"]):
        if sum(1 for x in chosen if x["role"] == "contrast") >= n_contrast:
            break
        if c["sample_id"] not in seen:
            chosen.append({**c, "role": "contrast"})
            seen.add(c["sample_id"])
    return chosen


def choose_spread(candidates: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """The pilot's pick: spread over (source, phase), deterministic."""
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    buckets: dict[tuple, list] = defaultdict(list)
    for c in candidates:
        buckets[(c["source"], c["sample_phase"])].append(c)
    chosen: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    while len(chosen) < args.count and any(buckets.values()):
        for key in sorted(buckets, key=str):
            while buckets[key]:
                c = buckets[key].pop()
                if c["sample_id"] not in seen_samples:
                    chosen.append(c)
                    seen_samples.add(c["sample_id"])
                    break
            if len(chosen) >= args.count:
                break
    return chosen


# ------------------------------------------------------------------------------------------ run


class GlmClient:
    """Direct OpenRouter chat call with reasoning on; records the cost OpenRouter reports."""

    def __init__(self, key: str, args: argparse.Namespace):
        self.key = key
        self.args = args
        self.records: list[dict[str, Any]] = []
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0))

    async def close(self) -> None:
        await self.http.aclose()

    def steered(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """The request GLM gets: the production context plus `--glm-system-suffix` on its system
        message. Only generation sees the suffix; stored data and judge documents keep the
        production prompt."""
        out = [dict(m) for m in messages]
        suffix = self.args.glm_system_suffix.strip()
        if suffix:
            if out and out[0]["role"] == "system":
                out[0]["content"] = f"{out[0]['content'].rstrip()}\n\n{suffix}"
            else:
                out.insert(0, {"role": "system", "content": suffix})
        # GLM ignores reasoning instructions in the system prompt but follows one appended to the
        # latest observation (provider probes, 2026-09-27)
        if self.args.glm_user_suffix and out and out[-1]["role"] == "user":
            out[-1]["content"] = out[-1]["content"] + self.args.glm_user_suffix
        return out

    async def turn(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": GLM_MODEL,
            "messages": self.steered(messages),
            "max_tokens": TURN_TOKEN_LIMIT,
            "temperature": self.args.temperature,
            "reasoning": {"enabled": True, "effort": self.args.reasoning_effort},
            "usage": {"include": True},
        }
        if self.args.glm_providers:
            # listed providers only, in order: the default router picked Wafer/Mistral, which drop
            # reasoning most often and cost ~3x Baidu
            payload["provider"] = {"order": self.args.glm_providers.split(","), "allow_fallbacks": False}
        last_error = None
        for attempt in range(4):
            started = time.monotonic()
            try:
                response = await self.http.post(
                    f"{OPENROUTER}/chat/completions", json=payload,
                    headers={"Authorization": f"Bearer {self.key}"})
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(5 * (attempt + 1))
                continue
            usage = body.get("usage") or {}
            self.records.append({
                "purpose": "glm_turn", "provider": body.get("provider"),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
                "cost": float(usage.get("cost") or 0.0),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            })
            choices = body.get("choices") or []
            if not choices:
                last_error = f"no choices: {str(body.get('error'))[:200]}"
                continue
            message = choices[0].get("message") or {}
            reasoning = message.get("reasoning") or ""
            content = message.get("content") or ""
            if not reasoning and "</think>" in content:  # some providers inline the reasoning
                reasoning, content = content.split("</think>", 1)
                reasoning = reasoning.replace("<think>", "")
            return {
                "reasoning": reasoning.strip(),
                "action": content.strip(),
                "finish_reason": choices[0].get("finish_reason"),
                "provider": body.get("provider"),
                "usage": self.records[-1],
            }
        raise RuntimeError(f"GLM turn failed after retries: {last_error}")

    async def good_turn(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], int, list[str]]:
        """Resample a turn whose reasoning is empty, over the reasoning cap, or cut at the token limit.

        This is data generation, not an eval: a bad sample is simply drawn again. After
        `--turn-retries` redraws the last one is kept and carries its defect."""
        rejected: list[str] = []
        for _ in range(self.args.turn_retries + 1):
            out = await self.turn(messages)
            if out["finish_reason"] == "length":
                rejected.append("truncated")
            elif self.args.retry_only_truncated:
                return out, len(rejected), rejected
            elif not out["reasoning"]:
                rejected.append("empty_reasoning")
            elif out["usage"]["reasoning_tokens"] > self.args.reasoning_cap:
                rejected.append("reasoning_over_cap")
            else:
                return out, len(rejected), rejected
        return out, len(rejected) - 1, rejected


def candidate_text(turn: dict[str, Any]) -> str:
    """The turn as a Qwen3.6 candidate writes it after `<think>\\n`."""
    return f"{turn['reasoning']}\n</think>\n\n{turn['action']}"


def submitted(task: dict[str, Any], text: str) -> bool:
    if task["submit_command"]:
        return is_exact_submission(text, task["submit_command"])
    return COMPLETE_MARKER in text


async def generate(task: dict[str, Any], glm: GlmClient, simulator: ObservationSimulationService,
                   args: argparse.Namespace) -> dict[str, Any]:
    context = [{"role": m["role"], "content": m["content"]} for m in task["context"]]
    convo = list(context)  # what GLM sees: earlier actions only
    sim_convo = list(context)  # what the simulator sees in production: earlier turns verbatim
    fmt = detect_format(task["sample_id"], context)
    turns: list[dict[str, Any]] = []
    stop = "horizon"
    for index in range(task["horizon"]):
        out, redraws, rejected = await glm.good_turn(convo)
        text = candidate_text(out)
        record = {"role": "assistant", "content": text, "reasoning": out["reasoning"], "action": out["action"],
                  "finish_reason": out["finish_reason"], "reasoning_tokens": out["usage"]["reasoning_tokens"],
                  "completion_tokens": out["usage"]["completion_tokens"], "provider": out["provider"],
                  "redraws": redraws, "rejected_draws": rejected}
        if out["finish_reason"] == "length":
            record["content"] = truncation_notice(TURN_TOKEN_LIMIT)
            record["truncated"] = True
            turns.append(record)
            stop = "truncated"
            break
        turns.append(record)
        if is_abandoned(out["action"]):
            stop = "abandoned"
            break
        if submitted(task, out["action"]):
            stop = "submitted"
            if index + 1 < task["horizon"]:
                turns.append({"role": "user", "content": wrap(task["submit_marker"] or COMPLETE_MARKER, fmt)})
            break
        if index + 1 >= task["horizon"]:
            break
        observation = await simulator.simulate(SimulateObservationRequest(
            eval_run_id=f"glm-thinking-pilot:{task['eval_run_id']}", sample_id=task["sample_id"],
            prompt="", assistant_output=text, messages=sim_convo))
        turns.append({"role": "user", "content": observation})
        # production re-renders history through the chat template, which keeps only the action
        convo = convo + [{"role": "assistant", "content": out["action"]}, {"role": "user", "content": observation}]
        sim_convo = sim_convo + [{"role": "assistant", "content": text}, {"role": "user", "content": observation}]
    return {"turns": turns, "stop": stop}


def document(task: dict[str, Any], turns: list[dict[str, Any]]) -> str:
    rendered = [{"role": m["role"], "content": m["content"]} for m in task["context"]]
    for t in turns:
        if t["role"] == "assistant":
            rendered.append({"role": "assistant", "content": t["content"], "score_target": True})
        else:
            rendered.append({"role": "user", "content": t["content"], "environment_observation": True})
    return format_scored_trajectory(rendered)


async def key_status(key: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.get(f"{OPENROUTER}/key", headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        return dict(response.json().get("data") or {})


async def run(args: argparse.Namespace) -> None:
    key = args.key_file.read_text(encoding="utf-8").strip()
    tasks = read_jsonl(args.out / "pilot-input.jsonl")
    results_path = args.out / "results.jsonl"
    done = {r["task_id"] for r in read_jsonl(results_path) if r.get("status") == "ok"}
    pending = [t for t in tasks if t["task_id"] not in done]
    start_key = await key_status(key)
    if float(start_key.get("limit_remaining") or 0) <= args.floor_usd:
        raise SystemExit(f"key at or below the ${args.floor_usd} floor; nothing started")
    settings = JudgeSettings(openrouter_api_key=key, max_concurrency_per_model=args.concurrency, judge_repeats=1,
                             repo_context_url=args.repo_context_url)
    if args.repo_context_url:
        async with httpx.AsyncClient(timeout=10) as http:
            (await http.get(f"{args.repo_context_url.rstrip('/')}/health")).raise_for_status()
    elif not args.allow_ungrounded:
        raise SystemExit("no --repo-context-url: pass --allow-ungrounded to run without grounding")
    glm = GlmClient(key, args)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()
    async with JudgeLLMClient(settings) as client:
        repo_context = RepoContextClient(settings) if args.repo_context_url else None
        simulator = ObservationSimulationService(settings, client, repo_context=repo_context)

        async def one(task: dict[str, Any]) -> None:
            async with semaphore:
                t0 = time.monotonic()
                glm_mark, client_mark = len(glm.records), len(client.usage_records)
                try:
                    generated = await generate(task, glm, simulator, args)
                    async def judged_once_more(doc: str, side: str) -> tuple[dict[str, Any], int]:
                        """A reading the judge could not parse is asked again, up to --judge-retries times."""
                        for attempt in range(args.judge_retries + 1):
                            out = await judge_document(client=client, settings=settings, document=doc,
                                                       questions=task["questions"], side=side)
                            if out["parse_ok"] or out["deterministic_zero"]:
                                return out, attempt
                        return out, attempt

                    judged, judge_retries = await judged_once_more(document(task, generated["turns"]), "candidate")
                    king_rejudge = None
                    if args.rejudge_king:
                        first = task["king_rollouts"][0]
                        king_out, _ = await judged_once_more(first["document"], "king")
                        king_rejudge = {"replica": first["replica"], "production_score": first["score"],
                                        "local_score": king_out["score"], "parse_ok": king_out["parse_ok"]}
                    result = {
                        "task_id": task["task_id"], "status": "ok", "stop": generated["stop"],
                        "sample_id": task["sample_id"], "source": task["source"], "sample_phase": task["sample_phase"],
                        "horizon": task["horizon"], "turns": generated["turns"],
                        "glm_score": judged["score"], "glm_parse_ok": judged["parse_ok"], "judge_retries": judge_retries,
                        "glm_judge_error": judged["error"],
                        "glm_amputated": judged["amputated_thinking"], "glm_zero_reason": judged["zero_reason"],
                        "king_scores": [r["score"] for r in task["king_rollouts"]], "king_mean": task["king_mean"],
                        "glm_reference_scores": task["glm_reference_scores"], "king_rejudge": king_rejudge,
                    }
                except Exception as exc:  # recorded, the rest of the batch goes on
                    result = {"task_id": task["task_id"], "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                result["cost_usd"] = {
                    "glm": sum(r["cost"] for r in glm.records[glm_mark:]),
                    "simulator_and_judge": sum(float(r.get("cost") or 0) for r in client.usage_records[client_mark:]),
                }
                result["elapsed_seconds"] = round(time.monotonic() - t0, 1)
                append_jsonl(results_path, result)
                print(json.dumps({k: result.get(k) for k in ("task_id", "status", "stop", "glm_score", "king_mean",
                                                             "cost_usd", "elapsed_seconds", "error")}), flush=True)

        await asyncio.gather(*(one(t) for t in pending))
        simulator_cost = sum(float(r.get("cost") or 0) for r in client.usage_records if r.get("purpose") == "simulate")
        judge_cost = spent_usd(client) - simulator_cost
    await glm.close()
    end_key = await key_status(key)
    summarize(args, start_key, end_key, sum(r["cost"] for r in glm.records), simulator_cost, judge_cost,
              time.monotonic() - started)


def summarize(args, start_key, end_key, glm_cost, simulator_cost, judge_cost, elapsed) -> None:
    results = [r for r in read_jsonl(args.out / "results.jsonl") if r.get("status") == "ok"]
    turns = [t for r in results for t in r["turns"] if t["role"] == "assistant"]
    reasoning_tokens = [t["reasoning_tokens"] for t in turns if not t.get("truncated")]
    scored = [r for r in results if r["glm_score"] is not None]
    rejudged = [r["king_rejudge"] for r in results if r.get("king_rejudge") and r["king_rejudge"]["local_score"] is not None]
    summary = {
        "tasks_ok": len(results),
        "errors": sum(1 for r in read_jsonl(args.out / "results.jsonl") if r.get("status") == "error"),
        "stops": dict(Counter(r["stop"] for r in results)),
        "assistant_turns": len(turns),
        "reasoning_tokens_per_turn": {"mean": st.mean(reasoning_tokens) if reasoning_tokens else None,
                                      "p50": quantile(reasoning_tokens, .5), "p90": quantile(reasoning_tokens, .9),
                                      "max": max(reasoning_tokens) if reasoning_tokens else None},
        "turns_empty_reasoning": sum(1 for t in turns if not t.get("reasoning")),
        "redraws_total": sum(int(t.get("redraws") or 0) for t in turns),
        "rejected_draw_reasons": dict(Counter(x for t in turns for x in t.get("rejected_draws") or [])),
        "turns_kept_with_defect": sum(1 for t in turns if (t.get("rejected_draws") or [])
                                      and len(t["rejected_draws"]) > int(t.get("redraws") or 0)),
        "providers": dict(Counter(str(t.get("provider")) for t in turns)),
        "judge_retries_total": sum(int(r.get("judge_retries") or 0) for r in results),
        "glm_unscored": sum(1 for r in results if r["glm_score"] is None),
        "turns_truncated": sum(1 for t in turns if t.get("truncated")),
        "turns_without_one_bash_block": sum(1 for t in turns if not t.get("truncated") and not first_bash_block(t["action"])),
        "turns_starting_THOUGHT": sum(1 for t in turns if t.get("action", "").lstrip().startswith("THOUGHT")),
        "glm_score_mean": st.mean(r["glm_score"] for r in scored) if scored else None,
        "king_mean_of_same_tasks": st.mean(r["king_mean"] for r in scored) if scored else None,
        "glm_beats_king_mean_by_0.05": sum(1 for r in scored if r["glm_score"] >= r["king_mean"] + 0.05),
        "glm_below_king_mean": sum(1 for r in scored if r["glm_score"] < r["king_mean"]),
        "glm_amputated": sum(1 for r in results if r.get("glm_amputated")),
        "king_rejudge_parity": {
            "n": len(rejudged),
            "mean_abs_diff": st.mean(abs(k["local_score"] - k["production_score"]) for k in rejudged) if rejudged else None,
            "max_abs_diff": max(abs(k["local_score"] - k["production_score"]) for k in rejudged) if rejudged else None,
        },
        "cost_usd": {"glm": round(glm_cost, 4), "simulator": round(simulator_cost, 4), "judge": round(judge_cost, 4),
                     "attributed_total": round(glm_cost + simulator_cost + judge_cost, 4),
                     "key_debit": round(float(end_key.get("usage") or 0) - float(start_key.get("usage") or 0), 4),
                     "key_remaining": end_key.get("limit_remaining")},
        "elapsed_seconds": round(elapsed, 1),
        "settings": {"reasoning_effort": args.reasoning_effort, "reasoning_cap": args.reasoning_cap,
                     "turn_retries": args.turn_retries, "judge_retries": args.judge_retries,
                     "glm_system_suffix": args.glm_system_suffix, "glm_user_suffix": args.glm_user_suffix,
                     "glm_providers": args.glm_providers, "rejudge_king": args.rejudge_king},
    }
    if results:
        summary["key_debit_per_task"] = round(summary["cost_usd"]["key_debit"] / len(results), 4)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--out", type=Path, required=True)
    sel.add_argument("--crawl-root", type=Path, default=Path("E:/albedo-storage-temp/raw-rollouts-graded-20260927"))
    sel.add_argument("--hf-root", type=Path, default=Path("E:/albedo-storage-temp/hf-albedo-reference-20260927"))
    sel.add_argument("--held-out", nargs="*", default=["719dbe80", "3a48801e"], help="eval id prefixes to skip")
    sel.add_argument("--count", type=int, default=20)
    sel.add_argument("--king-max", type=float, default=0.7)
    sel.add_argument("--king-rollout-max", type=float, default=0.85)
    sel.add_argument("--glm-min-best", type=float, default=0.9)
    sel.add_argument("--glm-min-each", type=float, default=0.7)
    sel.add_argument("--seed", type=int, default=20260927)
    sel.add_argument("--strategy", choices=["weak_king", "gap"], default="weak_king",
                     help="weak_king: the pilot's thresholds; gap: largest GLM-over-King gaps plus a contrast share")
    sel.add_argument("--min-gap", type=float, default=0.15, help="gap strategy: minimum robust gap")
    sel.add_argument("--contrast-share", type=float, default=0.10,
                     help="gap strategy: share of tasks where the King clearly beat every reference run")
    sel.add_argument("--exclude-from", type=Path, nargs="*", default=[],
                     help="pilot-input.jsonl files whose samples are skipped")
    runp = sub.add_parser("run")
    runp.add_argument("--out", type=Path, required=True)
    runp.add_argument("--key-file", type=Path, default=Path.home() / ".ssh" / "openrouter_new_key.txt")
    runp.add_argument("--concurrency", type=int, default=4)
    runp.add_argument("--temperature", type=float, default=0.6)
    runp.add_argument("--reasoning-effort", default="low", choices=["minimal", "low", "medium", "high"])
    runp.add_argument("--glm-system-suffix", default="",
                      help="appended to the system prompt GLM sees (not stored); GLM ignored this in v3")
    runp.add_argument("--glm-user-suffix", default=REASONING_NUDGE,
                      help="appended to the latest observation GLM sees (not stored); '' to disable")
    runp.add_argument("--reasoning-cap", type=int, default=2300,
                      help="redraw a turn whose reasoning exceeds this many tokens (King 127's p99 is ~2,280)")
    runp.add_argument("--turn-retries", type=int, default=0,
                      help="redraws of a turn with empty/over-cap reasoning or cut at the token limit")
    runp.add_argument("--judge-retries", type=int, default=2, help="re-asks of a judge reading that failed to parse")
    runp.add_argument("--retry-only-truncated", action="store_true",
                      help="redraw a turn only when it hit the token limit (with --turn-retries > 0)")
    rej = sub.add_parser("rejudge", help="judge again the rollouts of --out whose GLM score is missing")
    rej.add_argument("--out", type=Path, required=True)
    rej.add_argument("--key-file", type=Path, default=Path.home() / ".ssh" / "openrouter_new_key.txt")
    rej.add_argument("--attempts", type=int, default=4)
    rej.add_argument("--concurrency", type=int, default=16)
    rej.add_argument("--floor-usd", type=float, default=0.25)
    rej.add_argument("--tolerant-logprobs", action=argparse.BooleanOptionalAction, default=True,
                     help="accept logprob readings that differ from the content only by a few whitespace chars")
    rej.add_argument("--max-whitespace-skip", type=int, default=4)
    runp.add_argument("--glm-providers", default="", help="comma-separated OpenRouter provider order")
    runp.add_argument("--rejudge-king", action=argparse.BooleanOptionalAction, default=True)
    runp.add_argument("--floor-usd", type=float, default=0.25, help="refuse to start at or below this key balance")
    runp.add_argument("--repo-context-url", default="", help="local repo-context service, e.g. http://127.0.0.1:8093")
    runp.add_argument("--allow-ungrounded", action="store_true", help="run with an ungrounded simulator")
    args = parser.parse_args()
    if args.stage == "select":
        args.exclude_samples = {json.loads(line)["sample_id"] for path in args.exclude_from
                                for line in path.open(encoding="utf-8") if line.strip()}
        select(args)
    elif args.stage == "rejudge":
        asyncio.run(rejudge(args))
    else:
        asyncio.run(run(args))


async def rejudge(args: argparse.Namespace) -> None:
    """Ask the judge again for rollouts it could not read; results go to `rejudge.jsonl`
    (`results.jsonl` is left as written)."""
    key = args.key_file.read_text(encoding="utf-8").strip()
    if args.tolerant_logprobs:
        enable_whitespace_tolerant_logprobs(args.max_whitespace_skip)
    tasks = {t["task_id"]: t for t in read_jsonl(args.out / "pilot-input.jsonl")}
    scored = {r["task_id"] for r in read_jsonl(args.out / "rejudge.jsonl") if r["glm_score"] is not None}
    todo = [r for r in read_jsonl(args.out / "results.jsonl")
            if r.get("status") == "ok" and r["glm_score"] is None and r["stop"] != "truncated"
            and r["task_id"] not in scored]
    start = await key_status(key)
    if float(start.get("limit_remaining") or 0) <= args.floor_usd:
        raise SystemExit(f"key at or below the ${args.floor_usd} floor; nothing started")
    settings = JudgeSettings(openrouter_api_key=key, max_concurrency_per_model=args.concurrency, judge_repeats=1)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with JudgeLLMClient(settings) as client:
        async def one(r: dict[str, Any]) -> None:
            async with semaphore:
                task = tasks[r["task_id"]]
                doc = document(task, r["turns"])
                out, errors = None, []
                for _ in range(args.attempts):
                    out = await judge_document(client=client, settings=settings, document=doc,
                                               questions=task["questions"], side="candidate")
                    if out["parse_ok"] or out["deterministic_zero"]:
                        break
                    errors.append(str(out["error"])[:160])
                row = {"task_id": r["task_id"], "glm_score": out["score"], "glm_parse_ok": out["parse_ok"],
                       "glm_amputated": out["amputated_thinking"], "provider": out["provider"],
                       "failed_attempts": errors, "tolerant_logprobs": args.tolerant_logprobs}
                append_jsonl(args.out / "rejudge.jsonl", row)
                print(json.dumps({k: row[k] for k in ("task_id", "glm_score", "provider")} | {"fails": len(errors)}),
                      flush=True)

        await asyncio.gather(*(one(r) for r in todo))
    end = await key_status(key)
    rows = read_jsonl(args.out / "rejudge.jsonl")
    print(json.dumps({"attempted_now": len(todo), "tolerant_reads": TOLERANT_READS,
                      "scored_in_rejudge_file": sum(1 for r in rows if r["glm_score"] is not None),
                      "key_debit": round(float(end.get("usage") or 0) - float(start.get("usage") or 0), 4),
                      "key_remaining": end.get("limit_remaining")}, indent=2))


if __name__ == "__main__":
    main()
