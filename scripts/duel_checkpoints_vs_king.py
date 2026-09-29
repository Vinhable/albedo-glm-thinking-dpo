#!/usr/bin/env python3
"""Local duel: trained checkpoints against King 127, production's loop and graded judge.

select  (no API)  `--count` held-out graded-era tasks from the crawled runs, phases mixed like
                  production (65/15/20 cold/pre_edit/at_edit), sources spread, never a sample
                  already used for data (`--exclude-from`). Each task keeps the production run's
                  context, submit protocol, milestone-ladder questions, both King 127 rollouts with
                  their production scores, and its production horizon (recomputed with production's
                  `assign_horizons` over the whole run).
run     (paid)    for every `--policy LABEL=URL` (a `vllm serve` of the checkpoint, reached through
                  an ssh tunnel) and every task, `--replicas` rollouts through production's own turn
                  loop (`remote/worker.py`: re-ask of unusable turns with retry feedback, abandonment,
                  truncation, submission, missing-command observations), the simulator with local
                  repo grounding, then production's graded judge on the task's questions. The King
                  side reuses production's rollouts and judges them again here, so all sides share
                  judge provider and reading rules.
summary           per policy: mean score, task-level delta against the King (local rejudge and
                  production scores), wins/ties/losses, a bootstrap interval.

    py -3 scripts/duel_checkpoints_vs_king.py select --out E:/albedo-storage-temp/duel-20260928 \
        --exclude-from E:/albedo-storage-temp/glm-thinking-*/pilot-input.jsonl
    py -3 scripts/duel_checkpoints_vs_king.py run --out E:/albedo-storage-temp/duel-20260928 \
        --policy sft=http://127.0.0.1:8001 --policy dpo=http://127.0.0.1:8002 \
        --repo-context-url http://127.0.0.1:8093 --judge-providers baidu,streamlake
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
from types import SimpleNamespace
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from albedo_config import JudgeSettings  # noqa: E402
from albedo_config.models import JUDGE_LOGPROB_PROVIDER_PINS, JUDGE_MODELS  # noqa: E402
from albedo_eval_service.evaluator.shared.questions import assign_horizons  # noqa: E402
from albedo_eval_service.judge_api import ObservationSimulationService, SimulateObservationRequest  # noqa: E402
from albedo_eval_service.judge_llm_client import JudgeLLMClient  # noqa: E402
from albedo_eval_service.modelstore.canonical_model_config import (  # noqa: E402
    canonical_generation_config,
    canonical_max_model_len,
)
from albedo_eval_service.remote import worker as production  # noqa: E402
from albedo_eval_service.remote.dataset import EvalSample  # noqa: E402
from albedo_eval_service.remote.generation import GenerationResult  # noqa: E402
from albedo_eval_service.repo_context_client import RepoContextClient  # noqa: E402
from albedo_eval_service.shared.observation_format import first_bash_block  # noqa: E402
from build_gen_vinhable import rollouts, run_index  # noqa: E402
from graded_judge import TOLERANT_READS, enable_whitespace_tolerant_logprobs, judge_document  # noqa: E402
from pilot_glm_thinking_teacher import append_jsonl, assistant_count, key_status, read_jsonl  # noqa: E402
from vinhable_dpo_data import Renderer  # noqa: E402

PHASE_MIX = {"cold": 0.65, "pre_edit": 0.15, "at_edit": 0.20}  # production's sample mix
MAX_NEW_TOKENS = 4096  # production's max_new_tokens per turn
IM_END = 248046  # QWEN3_IM_END_TOKEN_ID, production's stop token


# ------------------------------------------------------------------------------------- select


def select(args: argparse.Namespace) -> None:
    from albedo_eval_service.remote.generation import format_scored_trajectory

    args.out.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run in run_index([args.crawl_root]):
        directory = run["directory"]
        submit, questions = {}, {}
        for row in read_jsonl(directory / "generated-samples.jsonl"):
            submit[str(row["sample_id"]).split("#", 1)[0]] = (row.get("submit_command"), row.get("submit_marker"))
        for row in read_jsonl(directory / "scoring-results.jsonl"):
            questions.setdefault(str(row["sample_id"]).split("#", 1)[0], row.get("questions") or [])
        found_run = list(rollouts(run, counts))
        # production's horizons: phase strata over every sample of the run (assign_horizons)
        contexts = {sid: next((r["context"] for r in found if r["side"] == "king"), None) for sid, _, found in found_run}
        horizons = assign_horizons([SimpleNamespace(sample_id=sid, messages=ctx)
                                    for sid, ctx in contexts.items() if ctx is not None])
        for sid, info, found in found_run:
            if info.get("scoring_mode") != "graded_20":
                counts["skip:not_graded"] += 1
                continue
            kings = sorted((r for r in found if r["side"] == "king"), key=lambda r: r["replica"])
            if len(kings) < 2 or not questions.get(sid) or sid not in horizons:
                counts["skip:missing_king_or_questions"] += 1
                continue
            if sid in args.exclude_samples:
                counts["skip:used_for_data"] += 1
                continue
            if sid in seen:
                counts["skip:repeat_sample"] += 1
                continue
            longest_king = max(assistant_count(r["completion"]) for r in kings)
            if longest_king > horizons[sid]:
                counts["skip:horizon_mismatch"] += 1
                continue
            seen.add(sid)
            command, marker = submit.get(sid, (None, None))
            candidates.append({
                "task_id": f"{run['eval_run_id']}:{sid}", "eval_run_id": run["eval_run_id"], "sample_id": sid,
                "source": sid.split("/", 1)[0], "sample_phase": info.get("sample_phase"),
                "context": kings[0]["context"], "submit_command": command or "", "submit_marker": marker or "",
                "questions": questions[sid], "horizon": horizons[sid],
                "king_rollouts": [
                    {"replica": r["replica"], "score": r["score"], "amputated": r["amputated"],
                     "assistant_turns": assistant_count(r["completion"]),
                     "document": format_scored_trajectory(
                         [dict(m) for m in r["context"]]
                         + [{**t, "score_target": True} if t["role"] == "assistant"
                            else {**t, "environment_observation": True} for t in r["completion"]])}
                    for r in kings],
                "king_mean": st.mean(r["score"] for r in kings),
            })
            counts["candidate"] += 1
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    by_phase: dict[str, list] = defaultdict(list)
    for c in candidates:
        by_phase[c["sample_phase"]].append(c)
    chosen: list[dict[str, Any]] = []
    for phase, share in PHASE_MIX.items():  # random inside a phase: sources come in the pool's (production's) mix
        chosen += by_phase.get(phase, [])[: round(args.count * share)]
    with (args.out / "duel-input.jsonl").open("w", encoding="utf-8") as handle:
        for c in chosen:
            handle.write(json.dumps(c, ensure_ascii=False) + "\n")
    summary = {"candidates": len(candidates), "chosen": len(chosen), "counts": dict(counts),
               "by_phase": dict(Counter(c["sample_phase"] for c in chosen)),
               "by_source": dict(Counter(c["source"] for c in chosen)),
               "by_horizon": dict(Counter(c["horizon"] for c in chosen)),
               "king_production_mean": round(st.mean(c["king_mean"] for c in chosen), 4) if chosen else None}
    (args.out / "select-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------------------- run


class RemoteVllm:
    """production's VllmServerGenerator._complete against an already running `vllm serve`."""

    def __init__(self, url: str, renderer: Renderer, sampling: dict[str, Any], max_model_len: int):
        self.client = httpx.Client(base_url=url.rstrip("/"), timeout=httpx.Timeout(900.0, connect=30.0),
                                   limits=httpx.Limits(max_connections=1024, max_keepalive_connections=1024))
        self.renderer, self.sampling, self.max_model_len = renderer, sampling, max_model_len
        self.completion_tokens: list[int] = []

    def generate(self, samples: list[EvalSample]) -> list[GenerationResult]:
        return [self._complete(s) for s in samples]

    def _complete(self, sample: EvalSample) -> GenerationResult:
        if len(self.renderer.encode(sample.prompt)) >= self.max_model_len - 64:
            return GenerationResult(sample.sample_id, "", truncated=True)
        body = {"model": "candidate", "prompt": sample.prompt, "max_tokens": MAX_NEW_TOKENS,
                "stop_token_ids": [IM_END], **self.sampling}
        try:
            for attempt in range(3):
                try:
                    response = self.client.post("/v1/completions", json=body)
                    break
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                    time.sleep(5)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return GenerationResult(sample.sample_id, "", f"{type(exc).__name__}: {exc}")
        choice = payload["choices"][0]
        tokens = int((payload.get("usage") or {}).get("completion_tokens") or 0)
        self.completion_tokens.append(tokens)
        return GenerationResult(sample_id=sample.sample_id, text=choice.get("text") or "",
                                truncated=choice.get("finish_reason") == "length" and tokens >= MAX_NEW_TOKENS)


def first_sample(task: dict[str, Any], replica: int) -> EvalSample:
    """The rollout's first-turn sample, rendered like production (canonical tokenizer's template)."""
    from albedo_eval_service.remote.dataset import format_messages

    messages = [{"role": m["role"], "content": m["content"]} for m in task["context"]]
    return EvalSample(sample_id=f"{task['sample_id']}#r{replica}", messages=messages,
                      prompt=format_messages(messages, tokenizer_path=str(production._CANONICAL_TOKENIZER_PATH),
                                             enable_thinking=True),
                      submit_marker=task["submit_marker"], submit_command=task["submit_command"])


async def trajectory(task: dict[str, Any], replica: int, generator: RemoteVllm, simulator, sim_slots,
                     label: str) -> GenerationResult:
    """production's `_generate_trajectories.trajectory` + `_simulate_observations` for one rollout."""
    sample = first_sample(task, replica)
    horizon = task["horizon"]
    results: list[list[GenerationResult]] = [[] for _ in range(horizon)]
    observations: list[dict] = [{} for _ in range(horizon)]
    side = "challenger"
    for turn in range(horizon):
        result = (await asyncio.to_thread(production._generate_retrying_bad_turns, generator, [sample]))[0]
        results[turn].append(result)
        if turn == horizon - 1:
            break
        key = (side, result.sample_id)
        if result.error:
            observed = production.ObservationResult(result.sample_id, "", result.error)
        elif result.truncated:
            observed = production.ObservationResult(result.sample_id, "")
        elif production._assistant_submitted(sample, result.text):
            observed = production.ObservationResult(result.sample_id, production._completion_observation(sample))
        elif not first_bash_block(result.text):
            observed = production.ObservationResult(result.sample_id, production._missing_command_observation(sample))
        else:
            try:
                async with sim_slots:
                    text = await simulator.simulate(SimulateObservationRequest(
                        eval_run_id=f"duel-{label}:{task['eval_run_id']}", sample_id=sample.sample_id,
                        prompt=sample.prompt, assistant_output=result.text, messages=sample.messages or []))
                observed = production.ObservationResult(result.sample_id, text)
            except Exception as exc:
                observed = production.ObservationResult(result.sample_id, "", f"{type(exc).__name__}: {exc}")
        observations[turn][key] = observed
        following = production._next_turn_samples([sample], [result], {key: observed}, side=side)
        if not following:
            break
        sample = following[0]
    first = first_sample(task, replica)
    return production._merge_trajectory_results([first], results, observations, side=side,
                                                token_limit=MAX_NEW_TOKENS, horizons={first.sample_id: horizon})[0]


async def run(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor

    # every rollout's blocking vLLM call runs in a thread (production's loop is synchronous): the default
    # executor's ~32 threads would cap the engines at 32 requests in flight
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=args.generation_threads))
    key = args.key_file.read_text(encoding="utf-8").strip()
    tasks = read_jsonl(args.out / "duel-input.jsonl")[: args.limit or None]
    results_path = args.out / "results.jsonl"
    generated_path = args.out / "generated.jsonl"  # --generate-only: rollouts waiting for the `score` stage
    done = {(r["label"], r["task_id"], r["replica"]) for r in read_jsonl(results_path) if r.get("status") == "ok"}
    generated_keys = {(r["label"], r["task_id"], r["replica"]) for r in read_jsonl(generated_path)
                      if r.get("status") == "generated"}
    policies = dict(p.split("=", 1) for p in args.policy)
    start_key = await key_status(key)
    remaining = float(start_key.get("limit_remaining") or 0)
    if remaining <= args.floor_usd:
        raise SystemExit(f"key at or below the ${args.floor_usd} floor; nothing started")
    pin_judge(args)
    settings = JudgeSettings(openrouter_api_key=key, max_concurrency_per_model=args.concurrency, judge_repeats=1,
                             repo_context_url=args.repo_context_url)
    async with httpx.AsyncClient(timeout=10) as http:
        (await http.get(f"{args.repo_context_url.rstrip('/')}/health")).raise_for_status()
        for label, url in policies.items():
            (await http.get(f"{url.rstrip('/')}/health")).raise_for_status()
    renderer = Renderer()
    generation = canonical_generation_config()
    sampling = {"temperature": float(generation["temperature"]), "top_p": float(generation["top_p"]),
                "top_k": int(generation["top_k"])}
    generators = {label: RemoteVllm(url, renderer, sampling, canonical_max_model_len())
                  for label, url in policies.items()}
    sim_slots, judge_slots = asyncio.Semaphore(args.concurrency), asyncio.Semaphore(args.concurrency)
    spent = {"usd": 0.0}
    started = time.monotonic()

    async with JudgeLLMClient(settings) as client:
        simulator = ObservationSimulationService(settings, client, repo_context=RepoContextClient(settings))

        async def judged(task: dict[str, Any], document: str) -> tuple[dict[str, Any], int]:
            async with judge_slots:
                for attempt in range(args.judge_retries + 1):
                    out = await judge_document(client=client, settings=settings, document=document,
                                               questions=task["questions"], side="candidate")
                    if out["parse_ok"] or out["deterministic_zero"]:
                        return out, attempt
                return out, attempt

        def over_budget() -> bool:
            spent["usd"] = sum(float(r.get("cost") or 0) for r in client.usage_records)
            return spent["usd"] >= args.max_spend

        async def policy_one(label: str, task: dict[str, Any], replica: int) -> None:
            if (label, task["task_id"], replica) in done or over_budget():
                return
            if args.generate_only and (label, task["task_id"], replica) in generated_keys:
                return
            t0, mark = time.monotonic(), len(client.usage_records)
            try:
                merged = await trajectory(task, replica, generators[label], simulator, sim_slots, label)
                if merged.error:
                    raise RuntimeError(merged.error)
                turns = merged.turns or []
                assistant = [t for t in turns if t.get("role") == "assistant" and t.get("score_target")]
                row = {"label": label, "task_id": task["task_id"], "replica": replica, "status": "generated",
                       "document": merged.text, "assistant_turns": len(assistant), "horizon": task["horizon"],
                       "truncated": merged.truncated, "abandoned": any(t.get("abandoned") for t in assistant),
                       "retry_feedbacks": sum(1 for t in turns if t.get("retry_feedback")),
                       "sample_phase": task["sample_phase"], "source": task["source"], "turns": turns}
                if not args.generate_only:
                    row.update(scored_fields(*await judged(task, merged.text)))
                    del row["document"]
            except Exception as exc:
                row = {"label": label, "task_id": task["task_id"], "replica": replica, "status": "error",
                       "error": f"{type(exc).__name__}: {exc}"}
            row["cost_usd"] = round(sum(float(r.get("cost") or 0) for r in client.usage_records[mark:]), 4)
            row["elapsed_seconds"] = round(time.monotonic() - t0, 1)
            append_jsonl(generated_path if args.generate_only else results_path, row)
            print(json.dumps({k: row.get(k) for k in ("label", "task_id", "replica", "status", "score",
                                                      "assistant_turns", "cost_usd", "error")}), flush=True)

        async def king_one(task: dict[str, Any], rollout: dict[str, Any]) -> None:
            if ("king", task["task_id"], rollout["replica"]) in done or over_budget():
                return
            mark = len(client.usage_records)
            out, retries = await judged(task, rollout["document"])
            row = {"label": "king", "task_id": task["task_id"], "replica": rollout["replica"], "status": "ok",
                   "score": out["score"], "production_score": rollout["score"], "parse_ok": out["parse_ok"],
                   "judge_retries": retries, "judge_error": out["error"], "amputated": out["amputated_thinking"],
                   "provider": out.get("provider"), "assistant_turns": rollout["assistant_turns"],
                   "horizon": task["horizon"], "sample_phase": task["sample_phase"], "source": task["source"],
                   "cost_usd": round(sum(float(r.get("cost") or 0) for r in client.usage_records[mark:]), 4)}
            append_jsonl(results_path, row)
            print(json.dumps({k: row.get(k) for k in ("label", "task_id", "replica", "score", "production_score")}),
                  flush=True)

        jobs = [] if args.generate_only else [king_one(t, r) for t in tasks for r in t["king_rollouts"]]
        jobs += [policy_one(label, t, replica) for t in tasks for label in policies
                 for replica in range(1, args.replicas + 1)]
        await asyncio.gather(*jobs)
    if args.generate_only:  # judged later, all sides at once, by the `score` stage
        print(json.dumps({"generated_rows": sum(1 for r in read_jsonl(generated_path)),
                          "simulator_usd": round(sum(float(r.get("cost") or 0) for r in client.usage_records), 4)}))
        return
    end_key = await key_status(key)
    summarize(args, key_debit=float(end_key.get("usage") or 0) - float(start_key.get("usage") or 0),
              key_remaining=end_key.get("limit_remaining"), elapsed=time.monotonic() - started,
              completion_tokens={label: g.completion_tokens for label, g in generators.items()})


def scored_fields(out: dict[str, Any], retries: int) -> dict[str, Any]:
    return {"status": "ok", "score": out["score"], "parse_ok": out["parse_ok"], "judge_retries": retries,
            "judge_error": out["error"], "amputated": out["amputated_thinking"], "zero_reason": out["zero_reason"],
            "provider": out.get("provider")}


async def score(args: argparse.Namespace) -> None:
    """Judge, in one pass, every rollout `run --generate-only` wrote and both King rollouts of each task."""
    key = args.key_file.read_text(encoding="utf-8").strip()
    tasks = {t["task_id"]: t for t in read_jsonl(args.out / "duel-input.jsonl")}
    results_path = args.out / "results.jsonl"
    done = {(r["label"], r["task_id"], r["replica"]) for r in read_jsonl(results_path) if r.get("status") == "ok"}
    generated = {}
    for r in read_jsonl(args.out / "generated.jsonl"):
        if r.get("status") == "generated":
            generated[(r["label"], r["task_id"], r["replica"])] = r
    todo = [r for k, r in generated.items() if k not in done and r["task_id"] in tasks]
    kings = [(t, r) for tid, t in tasks.items() if any(g["task_id"] == tid for g in generated.values())
             for r in t["king_rollouts"] if ("king", tid, r["replica"]) not in done]
    start_key = await key_status(key)
    if float(start_key.get("limit_remaining") or 0) <= args.floor_usd:
        raise SystemExit(f"key at or below the ${args.floor_usd} floor; nothing started")
    pin_judge(args)
    settings = JudgeSettings(openrouter_api_key=key, max_concurrency_per_model=args.concurrency, judge_repeats=1)
    slots = asyncio.Semaphore(args.concurrency)
    started = time.monotonic()
    async with JudgeLLMClient(settings) as client:
        async def judged(task: dict[str, Any], document: str) -> tuple[dict[str, Any], int]:
            async with slots:
                for attempt in range(args.judge_retries + 1):
                    out = await judge_document(client=client, settings=settings, document=document,
                                               questions=task["questions"], side="candidate")
                    if out["parse_ok"] or out["deterministic_zero"]:
                        return out, attempt
                return out, attempt

        def spent() -> float:
            return sum(float(r.get("cost") or 0) for r in client.usage_records)

        async def rollout_one(row: dict[str, Any]) -> None:
            if spent() >= args.max_spend:
                return
            mark = len(client.usage_records)
            out = {k: v for k, v in row.items() if k != "document"}
            out.update(scored_fields(*await judged(tasks[row["task_id"]], row["document"])))
            out["judge_usd"] = round(sum(float(r.get("cost") or 0) for r in client.usage_records[mark:]), 4)
            append_jsonl(results_path, out)
            print(json.dumps({k: out.get(k) for k in ("label", "task_id", "replica", "score")}), flush=True)

        async def king_one(task: dict[str, Any], rollout: dict[str, Any]) -> None:
            if spent() >= args.max_spend:
                return
            mark = len(client.usage_records)
            out = {"label": "king", "task_id": task["task_id"], "replica": rollout["replica"],
                   "production_score": rollout["score"], "assistant_turns": rollout["assistant_turns"],
                   "horizon": task["horizon"], "sample_phase": task["sample_phase"], "source": task["source"],
                   **scored_fields(*await judged(task, rollout["document"]))}
            out["judge_usd"] = round(sum(float(r.get("cost") or 0) for r in client.usage_records[mark:]), 4)
            append_jsonl(results_path, out)
            print(json.dumps({k: out.get(k) for k in ("label", "task_id", "replica", "score", "production_score")}),
                  flush=True)

        await asyncio.gather(*(rollout_one(r) for r in todo), *(king_one(t, r) for t, r in kings))
    end_key = await key_status(key)
    summarize(args, key_debit=float(end_key.get("usage") or 0) - float(start_key.get("usage") or 0),
              key_remaining=end_key.get("limit_remaining"), elapsed=time.monotonic() - started)


def pin_judge(args: argparse.Namespace) -> None:
    if args.judge_providers:  # the logprob judge's provider pins, for every side alike
        for model in JUDGE_MODELS:
            pins = {"allow_fallbacks": False, "order": args.judge_providers.split(",")}
            if args.judge_quantizations:
                pins["quantizations"] = args.judge_quantizations.split(",")
            JUDGE_LOGPROB_PROVIDER_PINS[model] = pins
    enable_whitespace_tolerant_logprobs(args.max_whitespace_skip)


async def probe_judge(args: argparse.Namespace) -> None:
    """Judge the first King rollout of the first `--limit` tasks with the chosen provider pins and
    compare with production's score: the pins must return usable logprobs before a paid run."""
    key = args.key_file.read_text(encoding="utf-8").strip()
    pin_judge(args)
    settings = JudgeSettings(openrouter_api_key=key, max_concurrency_per_model=args.concurrency, judge_repeats=1)
    tasks = read_jsonl(args.out / "duel-input.jsonl")[: args.limit or 3]
    async with JudgeLLMClient(settings) as client:
        async def one(task):
            rollout = task["king_rollouts"][0]
            out = await judge_document(client=client, settings=settings, document=rollout["document"],
                                       questions=task["questions"], side="candidate")
            return {"task_id": task["task_id"], "production": rollout["score"], "local": out["score"],
                    "parse_ok": out["parse_ok"], "provider": out.get("provider"), "error": str(out["error"])[:160]}
        rows = await asyncio.gather(*(one(t) for t in tasks))
        cost = sum(float(r.get("cost") or 0) for r in client.usage_records)
    for row in rows:
        print(json.dumps(row))
    print(json.dumps({"providers_seen": dict(Counter(str(r.get("provider")) for r in client.usage_records
                                                    if r.get("purpose") != "simulate")),
                      "tolerant_reads": dict(TOLERANT_READS), "cost_usd": round(cost, 4)}))


def summarize(args: argparse.Namespace, key_debit: float | None = None, key_remaining: Any = None,
              elapsed: float | None = None, completion_tokens: dict | None = None) -> None:
    rows = [r for r in read_jsonl(args.out / "results.jsonl") if r.get("status") == "ok"]
    latest: dict[tuple, dict] = {}
    for r in rows:  # a rerun's row replaces an earlier one
        latest[(r["label"], r["task_id"], r["replica"])] = r
    per_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    production_king: dict[str, list[float]] = defaultdict(list)
    for (label, task_id, _), r in latest.items():
        if r["score"] is not None:
            per_task[task_id][label].append(float(r["score"]))
        if label == "king":
            production_king[task_id].append(float(r["production_score"]))
    labels = sorted({label for label, _, _ in latest} - {"king"})
    rng = random.Random(0)
    summary: dict[str, Any] = {"tasks": len(per_task), "labels": labels}
    king_local = {t: st.mean(v["king"]) for t, v in per_task.items() if v.get("king")}
    summary["king"] = {
        "local_rejudge_mean": round(st.mean(king_local.values()), 4) if king_local else None,
        "production_mean": round(st.mean(st.mean(v) for v in production_king.values()), 4) if production_king else None,
        "rejudge_minus_production_mean": round(st.mean(
            king_local[t] - st.mean(production_king[t]) for t in king_local if t in production_king), 4)
        if king_local else None,
        "unscored_readings": sum(1 for (l, _, _), r in latest.items() if l == "king" and r["score"] is None),
    }
    for label in labels:
        mine = {t: st.mean(v[label]) for t, v in per_task.items() if v.get(label) and len(v[label]) == args.replicas}
        paired = [t for t in mine if t in king_local]
        deltas = [mine[t] - king_local[t] for t in paired]
        deltas_prod = [mine[t] - st.mean(production_king[t]) for t in mine if t in production_king]
        boots = sorted(st.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000)) if deltas else []
        label_rows = [r for (l, _, _), r in latest.items() if l == label]
        summary[label] = {
            "tasks_complete": len(mine), "rollouts": len(label_rows),
            "unscored_readings": sum(1 for r in label_rows if r["score"] is None),
            "mean": round(st.mean(mine.values()), 4) if mine else None,
            "delta_vs_king_local": round(st.mean(deltas), 4) if deltas else None,
            "delta_ci95": [round(boots[50], 4), round(boots[1949], 4)] if boots else None,
            "delta_vs_king_production": round(st.mean(deltas_prod), 4) if deltas_prod else None,
            "wins_ties_losses_0.05": [sum(d > 0.05 for d in deltas), sum(abs(d) <= 0.05 for d in deltas),
                                      sum(d < -0.05 for d in deltas)],
            "by_phase": {ph: round(st.mean(mine[t] - king_local[t] for t in paired
                                           if latest[(label, t, 1)]["sample_phase"] == ph), 4)
                         for ph in PHASE_MIX if any(latest[(label, t, 1)]["sample_phase"] == ph for t in paired)},
            "truncated_rollouts": sum(1 for r in label_rows if r.get("truncated")),
            "abandoned_rollouts": sum(1 for r in label_rows if r.get("abandoned")),
            "amputated_rollouts": sum(1 for r in label_rows if r.get("amputated")),
            "retry_feedbacks": sum(int(r.get("retry_feedbacks") or 0) for r in label_rows),
            "mean_assistant_turns": round(st.mean(r["assistant_turns"] for r in label_rows), 2) if label_rows else None,
        }
        if completion_tokens and completion_tokens.get(label):
            summary[label]["completion_tokens_per_turn_mean"] = round(st.mean(completion_tokens[label]), 1)
    summary["providers"] = dict(Counter(str(r.get("provider")) for r in latest.values()))
    summary["tolerant_reads"] = dict(TOLERANT_READS)
    summary["cost_usd"] = {"key_debit": round(key_debit, 4) if key_debit is not None else None,
                           "attributed": round(sum(float(r.get("cost_usd") or 0) for r in latest.values()), 4),
                           "key_remaining": key_remaining}
    summary["elapsed_seconds"] = round(elapsed, 1) if elapsed else None
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--out", type=Path, required=True)
    sel.add_argument("--crawl-root", type=Path, default=Path("E:/albedo-storage-temp/raw-rollouts-graded-20260927"))
    sel.add_argument("--count", type=int, default=100)
    sel.add_argument("--seed", type=int, default=20260928)
    sel.add_argument("--exclude-from", type=Path, nargs="*", default=[],
                     help="jsonl files (pilot-input, duel-input) whose sample ids are excluded")
    runp = sub.add_parser("run")
    runp.add_argument("--out", type=Path, required=True)
    runp.add_argument("--policy", action="append", required=True, help="LABEL=URL of a vllm serve (repeatable)")
    runp.add_argument("--replicas", type=int, default=2, help="rollouts per task and policy (production: 2)")
    runp.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke)")
    runp.add_argument("--key-file", type=Path, default=Path.home() / ".ssh" / "openrouter_new_key.txt")
    runp.add_argument("--concurrency", type=int, default=32, help="simulator and judge calls in flight, each")
    runp.add_argument("--generation-threads", type=int, default=512,
                      help="rollouts whose vLLM turn can be in flight at once (vLLM batches them)")
    runp.add_argument("--judge-retries", type=int, default=2)
    runp.add_argument("--judge-providers", default="", help="OpenRouter order for the logprob judge, e.g. baidu,streamlake")
    runp.add_argument("--judge-quantizations", default="", help="quantization filter for the judge ('' = any)")
    runp.add_argument("--max-whitespace-skip", type=int, default=4)
    runp.add_argument("--repo-context-url", default="http://127.0.0.1:8093")
    runp.add_argument("--floor-usd", type=float, default=0.25)
    runp.add_argument("--max-spend", type=float, default=45.0, help="stop starting new work past this many USD")
    summ = sub.add_parser("summary")
    summ.add_argument("--out", type=Path, required=True)
    summ.add_argument("--replicas", type=int, default=2)
    runp.add_argument("--generate-only", action="store_true",
                      help="roll out and simulate only (generated.jsonl); judge later with the `score` stage")
    scor = sub.add_parser("score", help="judge every generated rollout and the King rollouts, then summarize")
    scor.add_argument("--out", type=Path, required=True)
    scor.add_argument("--replicas", type=int, default=2)
    scor.add_argument("--key-file", type=Path, default=Path.home() / ".ssh" / "openrouter_new_key.txt")
    scor.add_argument("--concurrency", type=int, default=32)
    scor.add_argument("--judge-retries", type=int, default=2)
    scor.add_argument("--judge-providers", default="")
    scor.add_argument("--judge-quantizations", default="")
    scor.add_argument("--max-whitespace-skip", type=int, default=4)
    scor.add_argument("--floor-usd", type=float, default=0.25)
    scor.add_argument("--max-spend", type=float, default=25.0)
    probe = sub.add_parser("probe-judge")
    probe.add_argument("--out", type=Path, required=True)
    probe.add_argument("--limit", type=int, default=3)
    probe.add_argument("--key-file", type=Path, default=Path.home() / ".ssh" / "openrouter_new_key.txt")
    probe.add_argument("--concurrency", type=int, default=8)
    probe.add_argument("--judge-providers", default="")
    probe.add_argument("--judge-quantizations", default="")
    probe.add_argument("--max-whitespace-skip", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "probe-judge":
        asyncio.run(probe_judge(args))
        return
    if args.stage == "score":
        asyncio.run(score(args))
        return
    if args.stage == "select":
        args.exclude_samples = {json.loads(line)["sample_id"] for path in args.exclude_from
                                for line in path.open(encoding="utf-8") if line.strip()}
        select(args)
    elif args.stage == "summary":
        summarize(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
