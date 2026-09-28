#!/usr/bin/env python3
"""Probe which OpenRouter providers make GLM-5.2 actually reason on an agent turn.

Sends the same mid-trajectory context (a pilot task's production prompt, cut after several
steps) to each provider, pinned with no fallback, `--repeats` times at the given reasoning effort,
and reports per provider: how often reasoning came back, its length, the visible format, cost and
latency. The context is the production prompt as is; nothing is added to it.

    py -3 scripts/probe_glm_providers.py --pilot E:/albedo-storage-temp/glm-thinking-pilot-v2-20260927
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from albedo_eval_service.shared.observation_format import first_bash_block  # noqa: E402

OPENROUTER = "https://openrouter.ai/api/v1"


def mid_trajectory_context(pilot: Path, min_steps: int) -> tuple[str, list[dict[str, str]]]:
    """A v2 task's production context plus its first `min_steps` generated steps, the way the
    candidate saw them (earlier reasoning dropped, actions and observations kept)."""
    tasks = {json.loads(line)["task_id"]: json.loads(line) for line in (pilot / "pilot-input.jsonl").open(encoding="utf-8")}
    for line in (pilot / "results.jsonl").open(encoding="utf-8"):
        result = json.loads(line)
        turns = result.get("turns") or []
        if result.get("status") != "ok" or sum(t["role"] == "assistant" for t in turns) < min_steps + 1:
            continue
        messages = [{"role": m["role"], "content": m["content"]} for m in tasks[result["task_id"]]["context"]]
        steps = 0
        for turn in turns:
            if turn["role"] == "assistant":
                if steps == min_steps:
                    break
                messages.append({"role": "assistant", "content": turn["action"]})
                steps += 1
            else:
                messages.append({"role": "user", "content": turn["content"]})
        if messages[-1]["role"] == "user":
            return result["sample_id"], messages
    raise SystemExit("no task with enough steps")


async def call(http: httpx.AsyncClient, key: str, provider: str, messages, effort: str) -> dict:
    payload = {
        "model": "z-ai/glm-5.2", "messages": messages, "max_tokens": 4096, "temperature": 0.6,
        "reasoning": {"enabled": True, "effort": effort}, "usage": {"include": True},
        "provider": {"only": [provider], "allow_fallbacks": False},
    }
    started = time.monotonic()
    try:
        response = await http.post(f"{OPENROUTER}/chat/completions", json=payload,
                                   headers={"Authorization": f"Bearer {key}"})
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"provider": provider, "error": f"{type(exc).__name__}: {exc}"}
    if response.status_code != 200 or not body.get("choices"):
        return {"provider": provider, "error": f"HTTP {response.status_code}: {str(body.get('error'))[:160]}"}
    message = body["choices"][0].get("message") or {}
    usage = body.get("usage") or {}
    reasoning = (message.get("reasoning") or "").strip()
    content = (message.get("content") or "").strip()
    return {
        "provider": provider, "served_by": body.get("provider"),
        "reasoning_chars": len(reasoning),
        "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "one_bash_block": bool(first_bash_block(content)),
        "starts_thought": content.startswith("THOUGHT"),
        "finish_reason": body["choices"][0].get("finish_reason"),
        "cost": float(usage.get("cost") or 0.0),
        "seconds": round(time.monotonic() - started, 1),
        "reasoning_head": reasoning[:160],
    }


async def main_async(args: argparse.Namespace) -> None:
    key = args.key_file.read_text(encoding="utf-8").strip()
    sample_id, messages = mid_trajectory_context(args.pilot, args.steps)
    print(f"context: {sample_id}, {len(messages)} messages, {sum(len(m['content']) for m in messages)} chars")
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as http:
        results = await asyncio.gather(*(call(http, key, p, messages, args.effort)
                                         for p in args.providers for _ in range(args.repeats)))
    out = args.pilot.parent / f"glm-provider-probe-{args.effort}-step{args.steps}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{'provider':11s} ok reasoned | reasoning tok (each)          | bash | cost/call  | sec")
    for p in args.providers:
        rows = [r for r in results if r["provider"] == p]
        good = [r for r in rows if "error" not in r]
        errors = [r["error"] for r in rows if "error" in r]
        if not good:
            print(f"{p:11s} 0/{len(rows)} errors: {errors[:1]}")
            continue
        reasoned = sum(1 for r in good if r["reasoning_chars"] > 0)
        toks = [r["reasoning_tokens"] for r in good]
        print(f"{p:11s} {len(good)}/{len(rows)} {reasoned}/{len(good)}      | {str(toks):30s} | "
              f"{sum(r['one_bash_block'] for r in good)}/{len(good)} | ${st.mean(r['cost'] for r in good):.5f} | "
              f"{st.mean(r['seconds'] for r in good):.0f}" + (f"  errors: {errors[:1]}" if errors else ""))
    print(f"\ntotal cost ${sum(r.get('cost', 0) for r in results):.4f}; raw rows: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".ssh" / "openrouter_new_key.txt")
    parser.add_argument("--providers", nargs="+",
                        default=["streamlake", "baidu", "decart", "deepinfra", "z-ai", "alibaba"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--steps", type=int, default=5, help="generated steps before the probed turn")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
