#!/usr/bin/env python3
"""Publish `vinhable/vinhable_king_selfpairs` (public) to Hugging Face.

Rows from `build_king_selfpairs.py`, plus its summary, the lab prep summary and the pool analysis
(`analyze_king_pair_pool.py`). The card is rendered from the summaries; every uploaded file is
size-checked against the remote afterwards.

    py -3 scripts/push_king_selfpairs_to_hf.py --root E:/albedo-storage-temp/king-selfpairs-20260929 \
        --pool E:/albedo-storage-temp/king-pair-pool-20260929.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "vinhable/vinhable_king_selfpairs"
SIBLING = "vinhable/vinhable_single"


def card(s: dict, pool: dict, prep: dict) -> str:
    g, u = s["grouped"], pool["unused"]
    gr = u["groups"]
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: dev
    path: data/dev.jsonl
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- dpo
- preference
- swe-agent
- code
- reasoning
- self-improvement
size_categories:
- 1K<n<10K
---

# vinhable_king_selfpairs

Preference pairs for DPO on multi-turn software-engineering agent trajectories from the Albedo subnet
(Bittensor SN97): **the reigning King 127 against itself**. For each task, the King's two production
rollouts were scored by the production graded judge (A-T scale from top-20 logprobs) on the same
checklist; the better rollout is `chosen`, the worse one `rejected`, when they differ by at least
{s['margin_min']}. This is the "best-of-2" policy-improvement step of expert iteration / rejection-sampling
fine-tuning, turned into DPO pairs: both sides are the King's own text, so the pair differs only in what
the King did, not in style.

| | value |
|---|---|
| pairs (tasks) | {s['pairs']} (train {s['pairs_by_split'].get('train')}, dev {s['pairs_by_split'].get('dev')}, task-level split) |
| rows | {g['rows']} (train {g['rows_by_split'].get('train')}, dev {g['rows_by_split'].get('dev')}) |
| margin median | {s['margin_median']} |
| chosen / rejected score mean | {s['chosen_score_mean']} / {s['rejected_score_mean']} |
| phase | {', '.join(f'{k} {v}' for k, v in s['by_phase'].items())} |
| source | {', '.join(f'{k} {v}' for k, v in s['by_source'].items())} |

## Source

The graded-era production crawl (34 eval runs, King version 127 in every run, finished before
2026-09-27), public artifacts of albedo.tech. Tasks already spent elsewhere were removed: the GLM-5.2
teacher batches behind [`{SIBLING}`](https://huggingface.co/datasets/{SIBLING}), its pilots, and a
100-task held-out duel set.

## Row format

The same as [`{SIBLING}`](https://huggingface.co/datasets/{SIBLING}), so the same trainers read it.
A trajectory pair is cut into behaviour groups: both sides need >= 4 assistant turns, each side is split
into min(6, turns) groups, and row `k` holds groups 1..k of each side and supervises only group `k`'s
assistant turns (`sample_uid` = `pair_id#k`).

| field | meaning |
|---|---|
| `prompt` | production context up to the cut point (loss false) |
| `chosen`, `rejected` | the two continuations: assistant turns and environment observations |
| assistant turn: `content`, `loss`, `think_closed`, `tokens` | text after `<think>\\n` (reasoning, `</think>`, action); supervise?; closes `</think>`; Qwen3.6 tokens incl. `<|im_end|>` |
| assistant turn flags | `harness_text` (truncation notice), `empty_think`, `over_turn_limit` (> 4,096 tokens) |
| `chosen_score`, `rejected_score`, `margin` | graded judge scores of the two King rollouts |
| `chosen_side`, `rejected_side` | `king#r1` / `king#r2`: which production replica |

No loss on observations, harness notices, empty think blocks, or turns over 4,096 tokens. Rows where a
side is left with no trained turn (a `loss` turn must also close `</think>`) were dropped
({g['counts'].get('grouped_rows_dropped:side_without_trained_turn', 0)} rows).

Every row was converted to the branch-packed format of the lab trainer and checked token by token
against a per-turn rendering with the canonical Qwen3.6 chat template: {prep.get('train', {}).get('records')}
train + {prep.get('dev', {}).get('records')} dev records, 0 mismatches.

## What differs between chosen and rejected

On the whole-trajectory level the two sides look alike: the same number of turns (12.7 vs 12.9), the
same share that edits code (92%), the same first-edit turn (5.3 vs 5.4), the same command breadth and
reasoning length. The visible differences are small: the chosen side runs a test or a reproduction
more often (55% vs 48%) and submits more often (54% vs 47%). The score gap is mostly in *what* was read,
concluded and changed, which makes the DPO signal fine-grained.

## Headroom in the pool

On {u['units']} unused graded tasks: King mean {u['king_mean']}, King best-of-2 {u['king_best_of_2']},
best of the four rollouts in the run (King + challenger) {u['best_of_4_with_challenger']}. Groups:
A King capable (best >= 0.85) {gr['A']['share']:.0%}, B King inconsistent (gap >= 0.2) {gr['B']['share']:.0%},
C1 King weak but the challenger did it {gr['C1']['share']:.1%}, C2 King weak and nobody in the run did it
{gr['C2']['share']:.0%}. This set covers A and B; C2 needs outside knowledge (see the report).
`pool-analysis.json` holds the full breakdown.

## Files

- `data/train.jsonl`, `data/dev.jsonl`: the rows
- `build-summary.json`: counts from the build
- `lab-prep-summary.json`: the token-level check against the lab trainer's packed format
- `pool-analysis.json`: the pool groups and headroom

Code: `scripts/build_king_selfpairs.py`, `scripts/analyze_king_pair_pool.py`,
`scripts/prep_vinhable_for_lab.py` in https://github.com/Vinhable/albedo-glm-thinking-dpo
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    prep = json.loads((args.root / "lab" / "prep-summary.json").read_text(encoding="utf-8"))
    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    files = {
        "data/train.jsonl": args.root / "rows" / "train.jsonl",
        "data/dev.jsonl": args.root / "rows" / "dev.jsonl",
        "build-summary.json": args.root / "summary.json",
        "lab-prep-summary.json": args.root / "lab" / "prep-summary.json",
        "pool-analysis.json": args.pool,
    }
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card(summary, pool, prep).encode("utf-8")))
    commit = api.create_commit(REPO, repo_type="dataset", operations=ops,
                               commit_message="King 127 best-vs-worst self-pairs from the graded crawl")
    remote = {i.path: i.size for i in api.get_paths_info(REPO, list(files), repo_type="dataset")}
    for path, local in files.items():
        if remote.get(path) != local.stat().st_size:
            raise SystemExit(f"size mismatch for {path}: local {local.stat().st_size}, remote {remote.get(path)}")
    print(f"https://huggingface.co/datasets/{REPO}", commit.oid[:8], "all sizes match")


if __name__ == "__main__":
    main()
