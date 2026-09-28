#!/usr/bin/env python3
"""Compare GLM pilot thinking with the King's on the same tasks, in Qwen3.6 tokens.

Provider-reported reasoning tokens use GLM's tokenizer; the King (and a model trained on this data)
counts in Qwen3.6's. For every pilot task this reports, per assistant turn, the reasoning length
and the whole turn (reasoning + `</think>` + action) in Qwen3.6 tokens, for the GLM rollout and for
both King rollouts of that task in production.

    py -3 scripts/compare_pilot_thinking.py --pilot E:/albedo-storage-temp/glm-thinking-pilot-v4-20260927
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT / "scripts", ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from build_gen_vinhable import rollouts, run_index  # noqa: E402

TOKENIZER = ROOT / "assets" / "tokenizers" / "Qwen3.6-35B-A3B" / "tokenizer.json"


def split(content: str) -> tuple[str, str, bool]:
    if "</think>" in content:
        head, tail = content.split("</think>", 1)
        return head.strip(), tail.strip(), True
    return content.strip(), "", False


def stats(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    q = lambda p: ordered[min(len(ordered) - 1, int(p * len(ordered)))]
    return {"n": len(values), "mean": round(st.mean(values)), "p10": q(.1), "p50": q(.5), "p90": q(.9), "max": ordered[-1],
            "zero": sum(1 for v in values if v == 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--crawl-root", type=Path, default=Path("E:/albedo-storage-temp/raw-rollouts-graded-20260927"))
    args = parser.parse_args()
    tok = Tokenizer.from_file(str(TOKENIZER))
    count = lambda text: len(tok.encode(text, add_special_tokens=False).ids) if text else 0

    results = [json.loads(l) for l in (args.pilot / "results.jsonl").open(encoding="utf-8") if l.strip()]
    results = [r for r in results if r.get("status") == "ok"]
    wanted = {(r["task_id"].split(":", 1)[0], r["sample_id"]) for r in results}

    glm_reason, glm_turn = [], []
    for r in results:
        for t in r["turns"]:
            if t["role"] != "assistant" or t.get("truncated"):
                continue
            glm_reason.append(count(t.get("reasoning") or ""))
            glm_turn.append(count(t["content"]) + 1)  # + <|im_end|>

    king_reason, king_turn, king_closed = [], [], Counter()
    for run in run_index([args.crawl_root]):
        for sid, _, found in rollouts(run, Counter()):
            if (run["eval_run_id"], sid) not in wanted:
                continue
            for r in found:
                if r["side"] != "king":
                    continue
                for t in r["completion"]:
                    if t["role"] != "assistant" or t.get("harness_text"):
                        continue
                    reasoning, _, closed = split(t["content"])
                    king_closed[closed] += 1
                    king_reason.append(count(reasoning))
                    king_turn.append(count(t["content"]) + 1)

    report = {
        "tasks": len(results),
        "glm_reasoning_qwen_tokens": stats(glm_reason),
        "king_reasoning_qwen_tokens_same_tasks": stats(king_reason),
        "glm_turn_qwen_tokens": stats(glm_turn),
        "king_turn_qwen_tokens_same_tasks": stats(king_turn),
        "glm_turns_over_4096": sum(1 for v in glm_turn if v > 4096),
        "king_think_closed_share": round(king_closed[True] / max(1, sum(king_closed.values())), 3),
    }
    (args.pilot / "thinking-comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
