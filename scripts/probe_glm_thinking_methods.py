#!/usr/bin/env python3
"""Probe ways to get GLM-5.2 to write King-length reasoning on an agent turn.

The v1-v3 pilots and the provider probe showed GLM reasoning ~0-90 tokens per agent turn at any
effort, while King 127 reasons ~220 tokens (median). Methods compared, same contexts and providers:

  budget   reasoning channel with a token budget (`reasoning.max_tokens`) instead of an effort level
  nudge    effort high + an instruction appended to the last user message (recency beats system)
  inline   reasoning channel off; GLM writes its analysis in <analysis>...</analysis> before the
           answer, 150-400 words; the analysis would become the think block

Generation-side steering only: stored contexts stay production's.

    py -3 scripts/probe_glm_thinking_methods.py --pilot E:/albedo-storage-temp/glm-thinking-pilot-v2-20260927
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from albedo_eval_service.shared.observation_format import first_bash_block  # noqa: E402
from probe_glm_providers import mid_trajectory_context  # noqa: E402

NUDGE = ("\n\n[Before you reply: think it through carefully in your private reasoning first, at least "
         "150 words. Analyze what this output shows, what it implies for the bug, and plan the next "
         "step. Then answer in the required format.]")
NUDGE2 = ("\n\n[Before you reply, reason privately and in depth, 150 to 300 words, even if the next step "
          "looks routine: (1) what exactly this output shows, (2) what it confirms or rules out about the "
          "bug, (3) what could still go wrong, (4) why the next command is the best one. Then answer in "
          "the required format.]")
NUDGE3 = ("\n\n[Before you reply, reason privately and in depth: 250 to 500 words, even if the next step "
          "looks routine, and never more than 1,200 words. Cover (1) what exactly this output shows, "
          "(2) what it confirms or rules out about the bug, (3) the hypotheses still open and what could "
          "go wrong, (4) why the next command is the best one. Then answer in the required format.]")
NUDGE4 = ("\n\n[Reasoning requirement for this turn, including routine steps like verifying an edit: before "
          "replying, reason privately through all five points below, writing at least three full sentences "
          "for each point (about 300 to 500 words in total, never more than 1,200): "
          "1) Observation: what exactly this output shows, line by line where it matters. "
          "2) Evidence: what it confirms or rules out about the bug and the fix so far. "
          "3) Open questions: the hypotheses still open and what could still be wrong. "
          "4) Options: at least two candidate next commands and their trade-offs. "
          "5) Decision: why the chosen command is the best next step. "
          "Then answer in the required format.]")
NUDGE5 = NUDGE4.replace("at least three full sentences", "at least four full sentences").replace(
    "about 300 to 500 words in total, never more than 1,200", "about 400 to 700 words in total, never more than 1,500")
INLINE = """\
ANALYSIS PROTOCOL (every turn): before your answer, write your private analysis between <analysis>
and </analysis> tags, 150 to 400 words: what the latest output shows, what it implies for the task,
the hypotheses you are weighing, and why the next command is the right one. After </analysis>,
write the answer exactly in the format required above, with exactly one bash code block."""
ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL)


def request(method: str, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    msgs = [dict(m) for m in messages]
    if method == "budget":
        return msgs, {"enabled": True, "max_tokens": 1024}
    if method == "nudge":
        msgs[-1]["content"] += NUDGE
        return msgs, {"enabled": True, "effort": "high"}
    if method == "nudge2":
        msgs[-1]["content"] += NUDGE2
        return msgs, {"enabled": True, "effort": "high"}
    if method == "nudge3":
        msgs[-1]["content"] += NUDGE3
        return msgs, {"enabled": True, "effort": "high"}
    if method == "nudge4":
        msgs[-1]["content"] += NUDGE4
        return msgs, {"enabled": True, "effort": "high"}
    if method == "nudge5":
        msgs[-1]["content"] += NUDGE5
        return msgs, {"enabled": True, "effort": "high"}
    if method == "inline":
        msgs[0]["content"] = f"{msgs[0]['content'].rstrip()}\n\n{INLINE}"
        return msgs, {"enabled": False}
    raise ValueError(method)


async def call(http, key, provider, method, messages) -> dict:
    msgs, reasoning = request(method, messages)
    payload = {"model": "z-ai/glm-5.2", "messages": msgs, "max_tokens": 4096, "temperature": 0.6,
               "reasoning": reasoning, "usage": {"include": True},
               "provider": {"only": [provider], "allow_fallbacks": False}}
    started = time.monotonic()
    try:
        response = await http.post("https://openrouter.ai/api/v1/chat/completions", json=payload,
                                   headers={"Authorization": f"Bearer {key}"})
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"provider": provider, "method": method, "error": f"{type(exc).__name__}: {exc}"}
    if response.status_code != 200 or not body.get("choices"):
        return {"provider": provider, "method": method, "error": f"HTTP {response.status_code}: {str(body.get('error'))[:160]}"}
    message = body["choices"][0].get("message") or {}
    usage = body.get("usage") or {}
    reasoning_text = (message.get("reasoning") or "").strip()
    content = (message.get("content") or "").strip()
    analysis = ANALYSIS_RE.search(content)
    thinking = analysis.group(1).strip() if (method == "inline" and analysis) else reasoning_text
    answer = content[analysis.end():].strip() if (method == "inline" and analysis) else content
    return {
        "provider": provider, "method": method,
        "thinking_words": len(thinking.split()), "thinking_chars": len(thinking),
        "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
        "one_bash_block": bool(first_bash_block(answer)) and answer.count("```bash") == 1,
        "analysis_tag": bool(analysis), "finish_reason": body["choices"][0].get("finish_reason"),
        "cost": float(usage.get("cost") or 0.0), "seconds": round(time.monotonic() - started, 1),
        "thinking_head": thinking[:300], "answer_head": answer[:200],
    }


async def main_async(args: argparse.Namespace) -> None:
    key = args.key_file.read_text(encoding="utf-8").strip()
    contexts = {s: mid_trajectory_context(args.pilot, s)[1] for s in args.steps}
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as http:
        jobs = [(s, p, m) for s in args.steps for p in args.providers for m in args.methods for _ in range(args.repeats)]
        results = await asyncio.gather(*(call(http, key, p, m, contexts[s]) for s, p, m in jobs))
    for (s, _, _), r in zip(jobs, results):
        r["context_step"] = s
    out = args.pilot.parent / f"glm-thinking-methods-probe-{'-'.join(args.methods)}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{'method':7s} {'provider':10s} n  thinking words (each)          bash  cost/call")
    for m in args.methods:
        for p in args.providers:
            rows = [r for r in results if r["method"] == m and r["provider"] == p and "error" not in r]
            errs = [r["error"] for r in results if r["method"] == m and r["provider"] == p and "error" in r]
            if not rows:
                print(f"{m:7s} {p:10s} errors {errs[:1]}")
                continue
            words = [r["thinking_words"] for r in rows]
            print(f"{m:7s} {p:10s} {len(rows)}  {str(words):32s} {sum(r['one_bash_block'] for r in rows)}/{len(rows)}  "
                  f"${st.mean(r['cost'] for r in rows):.5f}" + (f"  errors {errs[:1]}" if errs else ""))
    print(f"total ${sum(r.get('cost', 0) for r in results):.4f}; rows: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".ssh" / "openrouter_new_key.txt")
    parser.add_argument("--providers", nargs="+", default=["baidu", "streamlake"])
    parser.add_argument("--methods", nargs="+", default=["budget", "nudge", "inline"])
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--repeats", type=int, default=2)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
