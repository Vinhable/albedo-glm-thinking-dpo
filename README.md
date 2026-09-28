# Albedo: GLM-5.2 teacher data with full thinking, and training King 127 on it

From ThanhVinhNguyen's Albedo working session. It covers 2026-09-26 (the idea) to 2026-09-28 (first GPU
results). It is for the lab (Fable and teammates), who will re-run the training on the lab's
infrastructure and duel the result against the King.

Numbers are **measured** unless marked otherwise. All scores are on the graded A–T judge (live since
2026-09-23). Shell commands and costs refer to our setup, so adapt paths to yours.

---

## 0. In one paragraph

We asked GLM-5.2 to **re-solve tasks where King CXXVII scored low, while writing full reasoning on every
turn**, and paired its trajectories with the King's own rollouts of the same task. That gives
**214 GLM-vs-King pairs** (public HF `vinhable/vinhable_single`). We trained King 127 on them with a
DPO objective whose NLL term on the chosen side dominates (in effect SFT on GLM's thinking, plus a DPO
term). **Run v3 (LR 2e-5, NLL weight 50, 70 steps)** is the first of our checkpoints that does all three:

- it moves toward the teacher (dev chosen log-ratio +0.30 nat/token);
- it **passes a local replica of the dedup gate** (`rel_struct` 0.0028, threshold 0.0025);
- it **still generates cleanly** (100% closed `</think>`, one bash block, no truncation or repetition, on
  a small first-turn check).

It has **not been duelled against the King**. The rented box was shut down before the weights were saved,
so the next step is to re-train v3 (~70 min on 8×H200) and run the duel.

## 1. Where this came from

- **`gen_vinhable` DPO failed** (lab runs r020/r027/r029, idea 29). Chosen and rejected were both
  on-policy King-family text, and the labels were mostly rollout luck. Drift stayed on the Adam noise line
  and dedup rejected them.
- **Lab r045** (SFT of King 127 on GLM-5.2 reference trajectories, LR 2e-5) was the first to pass the dedup
  replica. But it almost never closed `</think>` (0.8%). The GLM references in `dendriteholdings/albedo`
  were generated with reasoning off, so they teach an empty think block.
- **Idea (2026-09-26):** keep the part that worked (off-policy GLM text is what moves the weights) and fix
  the part that broke it (no thinking). Regenerate the teacher trajectories with GLM-5.2 **thinking on
  every turn**, in the same harness and grounding as production, **only on tasks where the King is weak**.
  Then pair them with the King's own rollouts of the same task (same checklist, same judge) for DPO.

## 2. Data generation

### 2.1 Pool: how many tasks have headroom

These counts come from HF `dendriteholdings/albedo`, split `glm_5_2`, revision `7bd9c247`, graded era
(King 127, 39 evals).

| Filter | Tasks |
|---|---|
| Pool (minus the lab's 200 duel samples) | 3,567 |
| King ≤ 0.7 and GLM reference ≥ 0.9 | 783 |
| King < 0.9 and GLM reference ≥ 0.9 | 1,836 |

- The GLM reference score is a best-of-3 pass rate on a checklist built from those same runs, so it is
  optimistic next to the King's graded score. See `results/data-generation/headroom-summary.json`.
- Our own crawl of 34 graded runs (`glm-regen-pool-survey.json`) has 3,333 tasks with two King rollouts.
  If GLM matched its reference, 1,072 of those tasks would give a pair where GLM beats both King rollouts
  by ≥ 0.15.
- King 127 thinking on these tasks: 366 tokens per turn on average (median 222, p99 2,278), 12.6 turns per
  rollout.

### 2.2 Local simulator grounding identical to production

`scripts/run_repo_context_local.sh` runs upstream's `repo_context_service` locally, and the pilot's
simulator calls it through `RepoContextClient`, as production does. `scripts/repo_context_windows.py`
is only a Windows shim (tempfile and long paths), so skip it on Linux. We checked parity with
`scripts/check_local_grounding.py`: 45 of 46 observations came out identical. Details are in
`docs/LOCAL_GROUNDING.md` (in Vietnamese).

### 2.3 Getting GLM-5.2 to actually think (pilots v1–v5, via OpenRouter)

| Pilot | Change | Tasks | Turns with empty reasoning | Reasoning tokens/turn | GLM vs King (same tasks) | Cost |
|---|---|---|---|---|---|---|
| v1 | default routing, effort low | 20 | 132/255 (52%) | 163 | 0.428 vs 0.577 | $2.74 |
| v2 | redraw bad turns, concurrency 32 | 20 | 95/275 (35%) | 126 | 0.698 vs 0.554 | $4.39 |
| v3 | system prompt demands thinking, no redraws | 15 | 109/195 (56%) | 72 | 0.475 vs 0.567 | $1.77 |
| v4 | **reasoning nudge appended to the latest observation + provider pinned to Baidu**, effort high | 15 | 0/199 | 641 | 0.499 vs 0.564 | $1.47 |
| **v5** | v4 with the nudge tuned (2 sentences × 5 points, ~200–400 words, never > 700), reasoning cap 2,300 | 15 | **0/202** | 540 | **0.765 vs 0.529** | $1.37 |

What we learned:
- GLM-5.2 skips or shortens reasoning on agent turns **regardless of `reasoning.effort` and of
  system-prompt instructions**.
- OpenRouter's default routing prefers providers (Wafer, Mistral) that drop reasoning most.
- What works is a short 5-point reasoning requirement appended to the latest observation, in the copy sent
  to GLM only (the stored trajectory keeps the real observation), with the provider pinned to Baidu and no
  fallbacks.
- Over-long reasoning (v4) cost score; capping it (v5) fixed that.
- The nudge text is `--glm-user-suffix` in `scripts/pilot_glm_thinking_teacher.py`.

### 2.4 A judge bug we hit (graded judge + GLM-5.2 providers)

Some providers' logprob token streams drop the space in text like `cat ./path` while the content keeps it.
`read_verdict_logprobs` needs an exact match, so the whole verdict read fails and the sample goes
unscored. Production has the same strictness.

We added a local tolerant reader: `graded_judge.enable_whitespace_tolerant_logprobs(max_skipped=4)` skips
at most 4 whitespace characters. A plain rejudge rescued 12 of 50 failures; the tolerant reader rescued
41 of 43 for $0.48. Only 25 dataset pairs rely on a tolerant read (`tolerant_logprob_scored_pairs`).

### 2.5 Batch of 300 tasks (2026-09-27)

- **Selection** (`select --strategy gap`, 90% main / 10% contrast):
  - 270 "main" tasks where the second-best GLM reference beats the better King rollout by the largest
    gaps (smallest selected gap 0.23; King mean 0.488, GLM reference 0.975);
  - 30 "contrast" tasks where the King beats every reference (King 0.936, GLM reference 0.798).
- **Run:** $24.81 (GLM $8.90, simulator $7.65, judge $7.46), about $0.083 per task, 33 minutes of wall
  time.
  - 3,772 GLM turns, 3 with empty reasoning; GLM reasoning averaged 492 Qwen tokens per turn vs the King's
    356.
  - 25 tasks were cut by the turn limit and re-run ($5.67); 50 judge failures were re-judged (§2.4).
- **Raw means:** GLM 0.612 vs King 0.537.
- **After merge** (`merge_glm_regen_results.py`): 163 tasks with a GLM > King pair and 56 with a King > GLM
  pair (gap ≥ 0.15); 79 tasks were ties and dropped.

### 2.6 Datasets (public HF)

| Dataset | Pairs | Rows (train / dev) | Directions |
|---|---|---|---|
| `vinhable/vinhable_single` (commit `f0149269`) | 214, one per task, largest gap | 1,218 (1,127 / 91) | 159 GLM > King, 55 King > GLM |
| `vinhable/vinhable_double` (commit `e7dcb94c`) | 363, one per qualifying King rollout | 2,065 (1,915 / 150) | 277 GLM > King, 86 King > GLM |

- **Format:** turn-group rows only. A task needs ≥ 4 turns and gets min(6, turns) groups; loss is on the
  last group only; there is no first-group behaviour match. The split is by task.
- **Loss masking:** `loss: false` on:
  - GLM turns with an empty think block;
  - turns over 4,096 tokens;
  - turns that echo the nudge's headings in the visible action (254 turns).
- **Scores:** GLM turns close `</think>` 100% of the time. Mean score across pairs is GLM 0.696 vs King
  0.483; the median gap is 0.33.
- **Builder:** `scripts/build_vinhable_glm_pairs.py`. The summaries are in
  `results/data-generation/datasets/`.

## 3. Training

### 3.1 Trainer (`scripts/train_vinhable_dpo.py`, DeepSpeed ZeRO-3, 8×H200)

**Rendering (`scripts/vinhable_dpo_data.py`).** Each supervised turn is its own sequence, rendered by the
**production chat template**. That template drops the reasoning of earlier assistant turns, and
observations are role `user`. So a turn is trained on the same context it sees at inference.

**Parameters and optimiser.**
- Only 1.410B parameters train: `self_attn`, `linear_attn`, `mlp.shared_expert(_gate)`, the layernorms
  and the final norm. Routed experts are frozen.
- torch AdamW (betas 0.9/0.95, no weight decay), cosine LR with 5 warmup steps and a floor of 0.1×LR.
- Gradient clipping at 1.0; 16 rows per step (2 per rank).

**Objective.** `--agg mean` means log-probs are length-normalised and scaled by N0 = 512.

```
h    = β·N0·[(logp_c − ref_c)/n_c − (logp_r − ref_r)/n_r]
loss = −(1 − ls)·logσ(h) − ls·logσ(−h) + nll_weight·(−logp_c/n_c)
```

Here β = 0.1 and label smoothing ls = 0.1. The reference is King 127, with log-probs computed once
before training (`--phase reference`).

**Checks.**
- Step-0 gate (|policy − ref| < 1e-3 per token).
- Dev eval every 10 steps.
- fp32 drift against the Adam noise line √Σlr².
- Slim exports at chosen steps, plus a reassembly back into a full checkpoint.

**Evaluation tools.**
- `scripts/thinking_check_vllm.py`: 16 dev prompts × 3 samples, first turn only.
- `scripts/check_dedup_gate_local.py`: upstream's own `fingerprint`/`decide` with a local secret. The
  statistics are on the same scale as the validator but not bit-identical.

**GPU lessons (Shadeform 8×H200, no nvcc).**
- **Dependencies.**
  - transformers' `HfDeepSpeedConfig` needs `accelerate`.
  - fla refuses Triton < 3.7.1 on Hopper (wrong GDN gradients), so use torch 2.14 cu130 with Triton 3.8.
  - `experts_implementation="grouped_mm"`.
- **ZeRO-3 rules.**
  - Every forward must go through `engine(...)`; calling the module directly makes ZeRO-3 overwrite
    gradients between backwards.
  - Every rank must issue the same collectives: one `lm_head` call per forward and equal
    forward/backward counts (pad with dummy micro-batches).
- **VM networking.**
  - The VM has no NVLS multicast, so set `NCCL_NVLS_ENABLE=0` for both training and vLLM.
  - vLLM's custom all-reduce failed CUDA-graph capture, so the check sets `disable_custom_all_reduce=True`.

Throughput was about 12–14k tokens/s, i.e. **~70 min for one epoch of `single` (70 steps)**. Our trainer
forwards each turn separately; the lab's branch-packed forward would be 4–5× cheaper at scale.

### 3.2 Runs (`vinhable_single`)

| Run | Settings | Dev accuracy | Dev log-ratio chosen / rejected (nat/token) | Local dedup `rel_struct` (threshold 0.0025) | First-turn thinking check |
|---|---|---|---|---|---|
| 1 | 2 epochs, LR 5e-6, NLL 0.1 | peak 0.73, 0.64 at the end | −0.067 / (not recorded here) | step 140: 0.00084, TRIVIAL-EDIT | step 140: 18.8% of turns cut at 4,096, degenerate loops; 81% one bash block |
| v2 | 1 epoch, LR 5e-6, NLL 0.5 | 0.63–0.69 | −0.023 / −0.043 | 0.0004 → 0.0006, TRIVIAL-EDIT | not run (vLLM error, since fixed) |
| **v3** | 1 epoch, **LR 2e-5, NLL 50** | 0.68 from step 10 on | **+0.296 / +0.066** | step 10 0.00166, step 20 0.00237, **step 35 0.00281, step 50 0.00287, step 70 0.00281: PASS** | see below |

**v3 thinking check** (16 dev prompts × 3 samples, first turn):

| Checkpoint | Closed `</think>` | One bash block | Cut at limit | Degenerate | Reasoning tokens (mean) |
|---|---|---|---|---|---|
| King 127 | 100% | 100% | 0% | 0% | 108 |
| v3 step 10 | 89.6% | 89.6% | 0% | 0% | 133 |
| v3 step 20 | 100% | 100% | 0% | 0% | 83 |
| v3 step 35 | 100% | 95.8% | 4.2% | 2.1% | 133 |
| **v3 step 50** | 100% | 100% | 0% | 0% | 122 |
| **v3 step 70** | 100% | 100% | 0% | 0% | 130 |

Density was 0.66–0.81 (threshold 0.20) and F_struct 0.16–0.23 (noise threshold 0.10) across v3. The raw
files are in `results/training/`.

### 3.3 Why runs 1 and v2 did not move, and v3 did

At h ≈ 0 the DPO gradient on each chosen token is (1 − 2·ls)/2 · β·N0/n = 0.4 × 51.2/n ≈ **20.5/n**. The
NLL term adds only `nll_weight/n`. So NLL 0.1 and 0.5 were 0.5% and 2.4% of the chosen-side pull. DPO
reached its margin mostly by pushing the rejected (King) text down, and the chosen log-ratio went
*negative*.

- With Adam, each parameter moves about LR per step whatever the gradient norm. When the step direction is
  not consistent, the displacement grows like √steps: v2's `rel_struct` rose by √2 from step 20 to 40, a
  random walk.
- **Run 1's collapse** (cut turns, loops) is the classic sign of the likelihood mass on both sides going
  down.
- **v3 sets NLL 50**, about 2.4× the DPO chosen pull at start, and more once the DPO term saturates.
  Together with LR 2e-5 (r045's value), the chosen side went up at once (+0.25 at step 10). The structural
  change then cleared the gate by step 35.
- The grad norm reads 400–700 in v3 only because the loss is scaled by 50. It is clipped to 1.0, and Adam
  normalises it anyway.
- `drift_over_noise` stayed at about 1.06 in every run, so it **does not separate** these runs:
  per-parameter RMS moves at about LR regardless of direction. `rel_struct` does separate them.

## 4. What is still unknown, and the risks

1. **Whether v3 beats the King.** Nothing above measures task score. SFT on a GLM teacher lost before
   (E6: −0.221).
2. **The dedup margin is thin: 1.12–1.15×.** Our script treats a margin under 20% as unsafe because the
   validator's secret differs. `rel_struct` flattened after step 35 as the cosine LR decayed.
3. **The thinking check is small** (48 first turns). It says nothing about later turns, horizon behaviour
   or answer quality.
4. **Dev accuracy was flat at 0.68 from step 10.** The preference part saturates early, and later steps
   mostly learn GLM's style. Reasoning rose from 108 to 130 tokens, far from GLM's ~500, and we do not know
   whether longer is better here.
5. **The teacher can leak its prompt.** The nudge echo is masked out, but GLM's reasoning is shaped by the
   nudge ("Observation / Evidence / Open questions / Options / Decision" style).
6. **25 pairs rely on the tolerant judge read**, which production does not do.

## 5. Proposed next steps

1. **Re-train v3** (70 steps) with our script or the lab trainer using the same objective; keep exports
   at steps 35, 50 and 70. With our script:

   ```
   export WORK=/workspace/vinhable CODE=/path/to/overlaid/albedo   # CODE must hold scripts/, assets/ and src/
   STAGE=setup     bash scripts/run_vinhable_dpo_8xh200.sh   # venvs, King 127 rev e920362b, both datasets
   STAGE=reference bash scripts/run_vinhable_dpo_8xh200.sh   # King log-probs of every row
   STAGE=sweep TAG=v3 EPOCHS=1 LR=2e-5 NLL=50 EXPORT_STEPS=10,20,35,50 \
                   bash scripts/run_vinhable_dpo_8xh200.sh   # train, then reassemble and check each export
   STAGE=dedup     bash scripts/run_vinhable_dpo_8xh200.sh   # local dedup gate on every export (CPU; can run beside the sweep)
   ```

   Other defaults: β 0.1, `--agg mean`, N0 512, ls 0.1, 2 rows per rank, seed 20260927.
   - If you port to the branch-packed trainer, keep the **per-token mean and N0 scaling**, or rescale NLL
     so that it stays about 2–3× the DPO chosen gradient.
2. **Duel step 70 (and step 50) against King 127**: 2 × 100 samples, graded judge, margin +0.025.
3. **Only if the duel is at least even: widen the dedup margin before submitting.**
   - Options: `MIN_LR_RATIO=0.3`–`0.5` (the LR floor, default 0.1) or a constant LR after warmup; `vinhable_double` (1.7× rows); or a
     second epoch at a moderate LR.
   - Target a local `rel_struct` of at least 1.3× the threshold, and re-check thinking health each time.
4. **Scale the data only after a positive duel.**
   - At about $0.083 per task (GLM + simulator + judge), 1,000–1,500 tasks cost ~$100–125; 12k tasks cost
     ~$1,000.
   - The current pool has about 800–1,800 weak-King tasks (§2.1). More needs new graded runs.
   - Scale in steps (1k, then more) and duel after each.
5. **If the duel loses**, run ablations before spending on data:
   - pure SFT (DPO off) vs v3;
   - drop the 55 King > GLM pairs vs keep them;
   - reasoning-length control;
   - a multi-turn health check in the real harness.

## 6. Repository layout and use

This repo is an **overlay** on upstream `github.com/tony-dendrite/albedo` at commit **`e638fdc`**.

**Setup.**
1. Clone upstream at that commit.
2. Copy this repo's `scripts/`, `tests/`, `ops/`, `docs/` and `assets/` into it.
3. `git apply patches/judge_llm_client_usage_records.patch`, which records per-call usage and cost in
   `JudgeLLMClient`; the pilot's cost accounting reads it.
4. Scripts import upstream packages from `src/` (`albedo_eval_service`, `albedo_config`,
   `model_validation`, `repo_context_service`).

| Path | What it is |
|---|---|
| `scripts/pilot_glm_thinking_teacher.py` | Main generator. Stages: `select` (weak-King and gap/contrast task selection), `run` (GLM-5.2 with the nudge, Baidu pinned, local grounding, graded judge), `rejudge` (`--tolerant-logprobs`) |
| `scripts/graded_judge.py` | Graded judge through production `_judge_side`, plus the whitespace-tolerant logprob reader |
| `scripts/merge_glm_regen_results.py`, `scripts/build_vinhable_glm_pairs.py`, `scripts/push_vinhable_glm_pairs_to_hf.py` | Merge batches, build `single`/`double` turn-group pairs, publish to HF |
| `scripts/analyze_glm_reference_headroom.py`, `scripts/survey_glm_regen_pool.py`, `scripts/compare_pilot_thinking.py`, `scripts/probe_glm_*.py` | Pool sizing, provider and nudge probes, thinking-length comparison |
| `scripts/run_repo_context_local.sh`, `scripts/repo_context_windows.py`, `scripts/check_local_grounding.py` | Local grounding service and parity check |
| `scripts/vinhable_dpo_data.py`, `scripts/prepare_vinhable_dpo.py` | Production-template renderer, per-turn sequences, dataset contract check |
| `scripts/train_vinhable_dpo.py` | Trainer (`--phase reference` / `--phase train`, `--local-cpu` for the tiny-model tests) |
| `scripts/reassemble_trained_checkpoint.py`, `scripts/thinking_check_vllm.py`, `scripts/check_dedup_gate_local.py` | Post-training: full checkpoint, generation health, dedup replica |
| `scripts/run_vinhable_dpo_8xh200.sh` | Stages: setup, prep, reference, smoke, train, sweep, dedup, reassemble, think, full |
| `ops/vinhable-dpo/machine.sh`, `ops/vinhable-dpo/watch.sh` | Push code to a rented box and run stages in tmux; low-frequency watcher |
| `tests/` | CPU tests: two-pass gradient equals autograd, padding invariance, pack budgets, renderer, tolerant reader. They need `scripts/make_tiny_qwen35moe.py` |
| `assets/tokenizers/Qwen3.6-35B-A3B/` | Canonical tokenizer and chat template used by the renderer |
| other `scripts/*.py` | Helpers imported by the above (rollout crawling and indexing, behaviour-group splitting) |
| `results/data-generation/` | Pool counts, pilot v1–v5 summaries, batch 300 summaries, dataset summaries |
| `results/training/` | `metrics.jsonl` of runs 1/v2/v3, thinking-check outputs, local dedup reports |
| `docs/VINHABLE_DPO_RUN.md`, `docs/LOCAL_GROUNDING.md` | Runbook and grounding notes (in Vietnamese; the runbook predates v3, so trust this README for the v3 settings) |

**Secrets.**
- The OpenRouter key is read from `--key-file`. The grounding service reads a GitHub token from the
  environment (`ALBEDO_REPO_CONTEXT_GITHUB_TOKEN`).
- Neither is in this repo.
