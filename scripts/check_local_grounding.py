#!/usr/bin/env python3
"""Check the local repo-context service against real production observations. No paid call.

For each pilot task, replay the King's first rollout turn by turn through production's
`ObservationSimulationService` wired to the local grounding service, with an LLM client that
refuses every call. A turn the grounding resolves exactly (or the absent-tool rule answers) returns
an observation without any model; that observation is compared with the one production recorded.
Turns that would need the LLM simulator are counted as `needs_llm` and skipped.

    py -3 scripts/check_local_grounding.py --pilot E:/albedo-storage-temp/glm-thinking-pilot-20260927
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from albedo_config import JudgeSettings  # noqa: E402
from albedo_eval_service.judge_api import ObservationSimulationService, SimulateObservationRequest  # noqa: E402
from albedo_eval_service.repo_context_client import RepoContextClient  # noqa: E402
from albedo_eval_service.shared.observation_format import first_bash_block  # noqa: E402
from build_gen_vinhable import rollouts, run_index  # noqa: E402


class NoLLM:
    """Stands in for JudgeLLMClient: any model call means the turn was not grounded exactly."""

    usage_records: list = []

    async def complete(self, *args, **kwargs):
        raise RuntimeError("needs_llm")

    async def score(self, *args, **kwargs):
        raise RuntimeError("needs_llm")


def norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").strip().splitlines())


async def main_async(args: argparse.Namespace) -> None:
    tasks = [json.loads(line) for line in (args.pilot / "pilot-input.jsonl").open(encoding="utf-8")]
    wanted = {t["eval_run_id"]: set() for t in tasks}
    for t in tasks:
        wanted[t["eval_run_id"]].add(t["sample_id"])
    settings = JudgeSettings(repo_context_url=args.url, openrouter_api_key="unused")
    repo_context = RepoContextClient(settings)
    simulator = ObservationSimulationService(settings, NoLLM(), repo_context=repo_context)
    http = httpx.AsyncClient(base_url=args.url, timeout=300)
    counts: Counter[str] = Counter()
    per_task = []
    mismatches = []
    for run in run_index([args.crawl_root]):
        if run["eval_run_id"] not in wanted:
            continue
        for sid, _, found in rollouts(run, Counter()):
            if sid not in wanted[run["eval_run_id"]]:
                continue
            king = next((r for r in found if r["side"] == "king"), None)
            if king is None:
                continue
            messages = [dict(m) for m in king["context"]]
            completion = king["completion"]
            task_counts: Counter[str] = Counter()
            for i, turn in enumerate(completion):
                if turn["role"] != "assistant" or i + 1 >= len(completion):
                    continue
                real = completion[i + 1]["content"]
                if not first_bash_block(turn["content"]):
                    task_counts["no_command"] += 1
                else:
                    try:
                        got = await simulator.simulate(SimulateObservationRequest(
                            eval_run_id="grounding-check", sample_id=sid, prompt="",
                            assistant_output=turn["content"], messages=messages))
                        same = norm(got) == norm(real)
                        task_counts["exact_match" if same else "exact_differs"] += 1
                        if not same and len(mismatches) < args.show:
                            mismatches.append({"sample_id": sid, "turn": i,
                                               "command": first_bash_block(turn["content"])[:200],
                                               "local": got[:600], "production": real[:600]})
                    except RuntimeError:
                        task_counts["needs_llm"] += 1
                        # what the LLM would have been given: a grounding block (transcribe from real
                        # output or repo context) or nothing (free simulation)
                        body = (await http.post("/repo-context", json={
                            "sample_id": sid, "assistant_output": turn["content"], "messages": messages})).json()
                        task_counts[f"llm_with_{body.get('kind') or 'none'}"
                                    + ("_context" if body.get("context") else "")] += 1
                messages = messages + [{"role": "assistant", "content": turn["content"]},
                                       {"role": "user", "content": real}]
            counts.update(task_counts)
            per_task.append({"sample_id": sid, **task_counts})
            print(json.dumps(per_task[-1]), flush=True)
    await repo_context.aclose()
    await http.aclose()
    grounded = counts["exact_match"] + counts["exact_differs"]
    summary = {
        "tasks": len(per_task), "turns": dict(counts),
        "grounded_share": grounded / max(1, grounded + counts["needs_llm"]),
        "match_rate_when_grounded": counts["exact_match"] / max(1, grounded),
        "mismatch_examples": mismatches,
    }
    (args.pilot / "grounding-check.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                    encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "mismatch_examples"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--crawl-root", type=Path, default=Path("E:/albedo-storage-temp/raw-rollouts-graded-20260927"))
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument("--show", type=int, default=8, help="mismatch examples kept in the report")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
