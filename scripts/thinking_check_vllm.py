#!/usr/bin/env python3
"""Thinking check of a checkpoint with vLLM: does it still think like the King?

Generates the first turn of `--prompts` dev tasks, `--samples` times each, with the checkpoint's own
generation config (genesis: temperature 1.0, top_p 0.95, top_k 20) and production's per-turn limit
of 4,096 tokens, from the production prompt (canonical template, thinking on). Reports per model:
closed-think rate, empty-think turns (Albedo halves rollouts where most turns open with an empty
`</think>`), reasoning length in tokens, turns cut at 4,096, exactly-one-bash-block rate and the
`THOUGHT:` prefix share. Run once per checkpoint (the King, then each export), then compare the
JSON summaries.

No simulator: one turn only, which is where empty or runaway thinking shows first.

    python scripts/thinking_check_vllm.py --model /workspace/models/king127 --label king127 \
        --data /workspace/data/single/data --out /workspace/runs/think
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vinhable_dpo_data import Renderer, load_rows  # noqa: E402

BASH_BLOCK = re.compile(r"```bash\s*\n.*?```", re.S)


def repeated_fragment(text: str, size: int = 40, times: int = 8) -> bool:
    """A `size`-char fragment occurring `times`+ times: loops that stay on one line."""
    if len(text) < size * times:
        return False
    counts: dict[str, int] = {}
    for i in range(0, len(text) - size, size // 2):
        frag = text[i:i + size]
        counts[frag] = counts.get(frag, 0) + 1
        if counts[frag] >= times:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=20)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    # Shadeform images have no nvcc: keep vLLM on kernels that need no JIT build (the lab's fix)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    renderer = Renderer()
    rows, seen = [], set()
    for row in load_rows(args.data, "dev"):
        if row["pair_id"] not in seen and row.get("prefix_groups") == 1:
            seen.add(row["pair_id"])
            rows.append(row)
    rows = rows[: args.prompts]
    prompts = [{"prompt_token_ids": renderer.encode(renderer.prompt(row["prompt"]))} for row in rows]
    generation = json.loads((args.model / "generation_config.json").read_text(encoding="utf-8"))
    params = SamplingParams(n=args.samples, temperature=generation.get("temperature", 1.0),
                            top_p=generation.get("top_p", 0.95), top_k=generation.get("top_k", 20),
                            max_tokens=args.max_tokens, skip_special_tokens=False)
    llm = LLM(model=str(args.model), tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.85, enable_prefix_caching=True, trust_remote_code=False,
              # the custom all-reduce kernel failed CUDA-graph capture on the Shadeform VM ("invalid
              # argument", 2026-09-28); NCCL is fast enough for a short generation check
              disable_custom_all_reduce=True)
    outputs = llm.generate(prompts, params)

    args.out.mkdir(parents=True, exist_ok=True)
    turns = []
    with (args.out / f"{args.label}.jsonl").open("w", encoding="utf-8") as handle:
        for row, out in zip(rows, outputs):
            for completion in out.outputs:
                text = completion.text
                closed = "</think>" in text
                reasoning, action = text.split("</think>", 1) if closed else (text, "")
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                repeat = max((lines.count(l) for l in set(lines)), default=0)
                turn = {
                    "sample_uid": row["sample_uid"], "closed": closed,
                    "empty_think": text.lstrip().startswith("</think>"),
                    "reasoning_tokens": len(renderer.encode(reasoning)),
                    "action_chars": len(action),
                    "tokens": len(completion.token_ids), "cut": completion.finish_reason == "length",
                    "one_bash_block": len(BASH_BLOCK.findall(action)) == 1,
                    "thought_prefix": action.lstrip().startswith("THOUGHT"),
                    # degeneration seen in the first run: one line repeated dozens of times, or a
                    # fragment repeating inside one enormous line (grep alternations, sed chains)
                    "max_line_repeat": repeat,
                    "degenerate": repeat >= 5 or repeated_fragment(text),
                }
                turns.append(turn)
                handle.write(json.dumps({**turn, "text": text}, ensure_ascii=False) + "\n")
    n = max(1, len(turns))
    reasoning = sorted(t["reasoning_tokens"] for t in turns if t["closed"])
    summary = {
        "label": args.label, "model": str(args.model), "turns": len(turns),
        "closed_rate": sum(t["closed"] for t in turns) / n,
        "empty_think_rate": sum(t["empty_think"] for t in turns) / n,
        "cut_at_limit_rate": sum(t["cut"] for t in turns) / n,
        "one_bash_block_rate": sum(t["one_bash_block"] for t in turns) / n,
        "thought_prefix_rate": sum(t["thought_prefix"] for t in turns) / n,
        "degenerate_rate": sum(t["degenerate"] for t in turns) / n,
        "action_chars_p50": sorted(t["action_chars"] for t in turns)[len(turns) // 2] if turns else None,
        "action_chars_p90": sorted(t["action_chars"] for t in turns)[int(0.9 * len(turns))] if turns else None,
        "reasoning_tokens_mean": st.mean(reasoning) if reasoning else None,
        "reasoning_tokens_p50": reasoning[len(reasoning) // 2] if reasoning else None,
        "reasoning_tokens_p90": reasoning[int(0.9 * len(reasoning))] if reasoning else None,
    }
    (args.out / f"{args.label}.summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
