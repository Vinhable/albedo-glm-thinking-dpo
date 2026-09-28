#!/usr/bin/env python3
"""Publish `vinhable/vinhable_single` and `vinhable/vinhable_double` (public) to Hugging Face.

The card is rendered from each set's `summary.json`, and every uploaded file is size-checked
against the remote afterwards.

    py -3 scripts/push_vinhable_glm_pairs_to_hf.py --root E:/albedo-storage-temp/vinhable-glm-pairs-20260927
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

OWNER = "vinhable"
SETS = {
    "single": ("vinhable_single", "one pair per task: the comparison with the largest score gap"),
    "double": ("vinhable_double", "every qualifying comparison: up to two pairs per task, one per King rollout"),
}


def card(repo: str, variant: str, what: str, s: dict, sibling: str) -> str:
    a = s["thinking_audit"]
    g = s["grouped"]
    gc = g["counts"]
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
size_categories:
- n<1K
---

# {repo}

Preference pairs for DPO on multi-turn software-engineering agent trajectories from the Albedo
subnet (Bittensor SN97): **GLM-5.2 regenerated with its own reasoning vs the reigning King 127**
on the same task. This is the **{variant}** variant ({what}); its sibling is
[`{OWNER}/{sibling}`](https://huggingface.co/datasets/{OWNER}/{sibling}). Both share the same tasks
and the same task-level train/dev split, so they can be compared directly.

| | rows | train | dev | pairs | GLM chosen | King chosen | tasks |
|---|---:|---:|---:|---:|---:|---:|---:|
| {variant} | {g['rows']} | {g['rows_by_split'].get('train', 0)} | {g['rows_by_split'].get('dev', 0)} | {g['pairs']} | {g['pairs_by_direction'].get('glm_over_king', 0)} | {g['pairs_by_direction'].get('king_over_glm', 0)} | {s['tasks']} |

Median score gap {s['margin_median']:.3f} (minimum {s['margin_min']}); mean GLM score
{s['glm_score_mean']:.3f}, mean King score {s['king_score_mean']:.3f} over the pairs.

## Turn groups

Rows follow the turn-group layout of
[`{OWNER}/gen_vinhable`](https://huggingface.co/datasets/{OWNER}/gen_vinhable): each side needs at
least four assistant turns ({gc.get('grouped_drop:fewer_than_four_turns', 0)} pairs dropped) and is
cut into the same number of behaviour groups, `min(6, turns)`. Unlike `gen_vinhable`, the first
groups of the two sides need **not** share a behaviour: the sides come from different models and
may open differently. A pair yields one row per prefix length `k` (`sample_uid` = `pair_id#k<k>`,
`prefix_groups`, `group_count`, `is_full_trajectory`): both sides hold their first `k` groups and
**only group `k`'s assistant turns carry `loss: true`**, so each turn is a target exactly once
across a pair's rows; a turn's own `loss: false` (see below) is kept. Rows whose last group has no
supervisable turn on a side are dropped ({gc.get('grouped_rows_dropped:no_supervised_turn', 0)} rows).
Filter `is_full_trajectory == true` for one trajectory-level row per pair.

## How the data was made

* **Tasks.** Production evaluation samples of King 127 in the graded-judge era (runs finished
  2026-09-23 to 09-26), taken from Albedo's public artifacts. 90% were picked where the published
  GLM-5.2 reference beat both King rollouts by a wide margin, 10% where the King clearly beat every
  reference run (so that some pairs prefer the King and "looks like GLM" cannot become a shortcut).
* **GLM side.** GLM-5.2 continues the production prompt turn by turn with reasoning enabled
  (OpenRouter, provider Baidu). Each turn is stored exactly as a Qwen3.6 candidate writes it:
  `reasoning` + `\\n</think>\\n\\n` + action. Later turns see earlier turns the way production
  renders history (actions only). Environment observations come from production's simulator with
  repository grounding (real snapshot of the task's repo; commands run against it where possible).
  At generation time only, a short instruction appended to the latest observation asked GLM to
  reason before every turn; it is **not** in the stored data.
* **King side.** The King's own two production rollouts of the task, with their production scores.
* **Scores.** Albedo's graded A-T judge (GLM-5.2, top-20 logprobs) on the task's own published
  checklist. The King's scores are production's; the GLM rollout was scored locally with the same
  judge code. {s['tolerant_logprob_scored_pairs']} pairs have a GLM score read with a
  whitespace-tolerant logprob alignment (the provider's token stream dropped a space that the
  content kept; verdict letters are unaffected).
* **Pairs.** GLM is compared with each King rollout; a comparison qualifies when the gap is at least
  {s['margin_min']}. The higher score is `chosen`. Ties are dropped.

## Row schema

| Field | Meaning |
|---|---|
| `prompt` | production context up to the cut point; every message `loss: false` |
| `chosen`, `rejected` | continuations: assistant turns and environment observations (`role: user`) |
| `direction` | `glm_over_king` or `king_over_glm` |
| `chosen_side`, `rejected_side` | `glm` or `king#r1` / `king#r2` |
| `glm_score`, `king_score`, `margin` | scores of the two compared rollouts; `margin` = GLM - King |
| `king_scores_both_rollouts` | both King rollouts' production scores |
| `task_role` | `main` (GLM reference beat the King) or `contrast` (King beat the reference) |
| `sample_uid`, `prefix_groups`, `group_count`, `is_full_trajectory` | position of the row in its pair's group split |
| `chosen_behavior_sequence`, `rejected_behavior_sequence`, `aligned_groups`, `aligned_fraction` | behaviour labels of the groups so far |
| `split`, `task_id`, `eval_run_id`, `sample_id`, `source`, `sample_phase`, `question_mode`, `king_version` | provenance |
| assistant turn: `content`, `loss`, `think_closed`, `tokens` | text after `<think>\\n`; supervise?; closes `</think>`; Qwen3.6 tokens incl. `<|im_end|>` |
| assistant turn flags | `harness_text` (truncation notice), `empty_think`, `over_turn_limit` (> 4,096 tokens), `nudge_echo` (see below) |

## Loss and rendering contract

* Render every completion turn verbatim as `<|im_start|>assistant\\n<think>\\n` + `content` +
  `<|im_end|>`; do not pass it through the chat template.
* When a turn is supervised, its context should hold earlier turns **as production renders them**:
  the chat template drops earlier reasoning, so the model never sees its past thinking.
* `loss: false` on the prompt, observations, harness notices, turns with an empty think block,
  turns over 4,096 tokens, and `nudge_echo` turns: GLM turns ({a['glm']['nudge_echo_turns']} of
  {a['glm']['assistant_turns']}) whose visible answer repeated the headings of the generation-time
  reasoning instruction.

## Thinking audit

| | GLM turns | King turns |
|---|---:|---:|
| assistant turns | {a['glm']['assistant_turns']} | {a['king']['assistant_turns']} |
| supervised | {a['glm']['supervised_turns']} | {a['king']['supervised_turns']} |
| close the think block | {a['glm']['think_closed_rate']:.1%} | {a['king']['think_closed_rate']:.1%} |
| empty think block | {a['glm']['empty_think_turns']} | {a['king']['empty_think_turns']} |
| median turn length (Qwen3.6 tokens) | {a['glm']['turn_tokens_median']} | {a['king']['turn_tokens_median']} |

GLM reasons longer than the King on average (about 490 vs 360 reasoning tokens per turn on these
tasks).

## Caveats

* Small: meant as a signal test before scaling up.
* GLM was scored by a local run of the judge, the King by production; on re-judged King rollouts
  the local judge differed from production by about 0.06 on average, hence the 0.15 margin.
* Observations are simulated (grounded where the command could be executed on the snapshot).
* In `{OWNER}/vinhable_double`, pairs of the same task share the GLM side.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    api = HfApi()
    print("hf user:", api.whoami()["name"])
    for variant, (name, what) in SETS.items():
        repo = f"{OWNER}/{name}"
        folder = args.root / variant
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        sibling = SETS["double" if variant == "single" else "single"][0]
        readme = folder / "README.md"
        readme.write_text(card(repo, variant, what, summary, sibling), encoding="utf-8")
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        files = {"README.md": readme, "summary.json": folder / "summary.json",
                 "data/train.jsonl": folder / "train.jsonl", "data/dev.jsonl": folder / "dev.jsonl"}
        existing = set(api.list_repo_files(repo, repo_type="dataset"))
        stale = [p for p in existing if p.startswith(("data/grouped/", "data/trajectory/"))]  # earlier layouts
        commit = api.create_commit(
            repo_id=repo, repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
            + [CommitOperationDelete(path_in_repo=p) for p in stale],
            commit_message=f"{variant}: turn-group rows only ({summary['grouped']['rows']} rows, "
                           f"{summary['grouped']['pairs']} pairs), no first-group rule")
        info = api.dataset_info(repo, files_metadata=True)
        remote = {s.rfilename: s.size for s in info.siblings}
        mismatch = {k: (v.stat().st_size, remote.get(k)) for k, v in files.items() if remote.get(k) != v.stat().st_size}
        print(repo, "commit", commit.oid[:8], "private" if info.private else "public",
              "sizes OK" if not mismatch else f"SIZE MISMATCH {mismatch}")


if __name__ == "__main__":
    main()
