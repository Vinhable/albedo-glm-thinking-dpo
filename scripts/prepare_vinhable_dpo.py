#!/usr/bin/env python3
"""Check a `vinhable_*` dataset against the training contract and size the run. No GPU.

For every row: render each supervised turn as its own production-exact sequence
(vinhable_dpo_data.py) and verify that
  * the prompt ends with `<|im_start|>assistant\\n<think>\\n`;
  * the target decodes back to the turn's content + `<|im_end|>`, and holds `</think>`;
  * no earlier turn's reasoning appears in the rendered history;
  * each assistant turn is supervised once across a pair's rows.
Writes `<out>/rows.jsonl` (per-row sequence counts and lengths) and `<out>/summary.json` with the
token totals the trainer will process, and time estimates at a given throughput.

    py -3 scripts/prepare_vinhable_dpo.py --data E:/albedo-storage-temp/vinhable-glm-pairs-20260927/single \
        --out E:/albedo-storage-temp/vinhable-dpo-prep/single
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vinhable_dpo_data import GENERATION_PROMPT, Renderer, load_rows, row_example, supervised  # noqa: E402


def quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    q = lambda p: ordered[min(len(ordered) - 1, int(p * len(ordered)))]
    return {"p50": q(.5), "p90": q(.9), "p99": q(.99), "max": ordered[-1]} if ordered else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seq-tokens", type=int, default=65536)
    parser.add_argument("--train-tokens-per-sec", type=float, default=24000.0,
                        help="8xH200 fwd+bwd throughput the lab measured (attention + shared expert trainable)")
    parser.add_argument("--forward-speedup", type=float, default=3.0, help="no-grad forward vs fwd+bwd")
    parser.add_argument("--check-every", type=int, default=1, help="decode-check every Nth row (1 = all)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    renderer = Renderer()
    counts: Counter[str] = Counter()
    report: dict = {"data": str(args.data)}
    seen_targets = defaultdict(Counter)  # (pair, side) -> count of each supervised content
    with (args.out / "rows.jsonl").open("w", encoding="utf-8") as out:
        for split in ("train", "dev"):
            rows = load_rows(args.data, split)
            lengths, per_row, seqs = [], [], []
            for i, row in enumerate(rows):
                ex = row_example(renderer, row)
                for side in ("chosen", "rejected"):
                    for m in row[side]:
                        if supervised(m):
                            seen_targets[(row["pair_id"], side)][m["content"]] += 1
                    if not getattr(ex, side):
                        counts["error:side_without_sequence"] += 1
                if i % args.check_every == 0:
                    for side in ("chosen", "rejected"):
                        turns = [m for m in row[side] if supervised(m)]
                        for m, s in zip(turns, getattr(ex, side)):
                            prompt_text = renderer.tokenizer.decode(s.prompt_ids, skip_special_tokens=False)
                            if not prompt_text.endswith(GENERATION_PROMPT):
                                counts["error:prompt_suffix"] += 1
                            target = renderer.tokenizer.decode(s.target_ids, skip_special_tokens=False)
                            if target != m["content"] + "<|im_end|>":
                                counts["error:target_roundtrip"] += 1
                            if renderer.think_close not in s.target_ids:
                                counts["error:target_without_think_close"] += 1
                            # earlier reasoning must be gone from the history
                            for earlier in row[side]:
                                if earlier is m:
                                    break
                                if earlier["role"] == "assistant" and "</think>" in earlier["content"]:
                                    reasoning, action = earlier["content"].split("</think>", 1)
                                    reasoning = reasoning.strip()
                                    # a model may repeat its reasoning in the visible answer; only
                                    # text that exists nowhere but in the reasoning counts
                                    if len(reasoning) > 40 and reasoning[:200] in prompt_text \
                                            and reasoning[:200] not in action:
                                        counts["error:history_keeps_reasoning"] += 1
                over = ex.max_length > args.max_seq_tokens
                counts[f"{split}:rows"] += 1
                counts[f"{split}:rows_over_max_seq"] += over
                lengths.extend(s.length for s in ex.chosen + ex.rejected)
                per_row.append(ex.total_length)
                seqs.append(len(ex.chosen) + len(ex.rejected))
                out.write(json.dumps({"uid": ex.uid, "split": split, "sequences": len(ex.chosen) + len(ex.rejected),
                                      "chosen_seq": len(ex.chosen), "rejected_seq": len(ex.rejected),
                                      "max_len": ex.max_length, "total_len": ex.total_length,
                                      "target_tokens": [ex.tokens("chosen"), ex.tokens("rejected")],
                                      "over_max_seq": over}) + "\n")
            total = sum(per_row)
            report[split] = {
                "rows": len(rows), "sequences": sum(seqs), "sequences_per_row": round(st.mean(seqs), 2),
                "sequence_tokens": quantiles(lengths), "tokens_per_epoch": total,
                "epoch_hours_estimate": round(total * (1 + 1 / args.forward_speedup) / args.train_tokens_per_sec / 3600, 2)
                if split == "train" else None,
                "reference_hours_estimate": round(total / (args.train_tokens_per_sec * args.forward_speedup) / 3600, 2),
            }
    counts["error:turn_supervised_twice"] = sum(1 for c in seen_targets.values() for v in c.values() if v > 1)
    report["counts"] = dict(counts)
    report["errors"] = {k: v for k, v in counts.items() if k.startswith("error:") and v}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(f"contract violations: {report['errors']}")


if __name__ == "__main__":
    main()
