# Albedo: GLM-5.2 teacher data with full thinking, and training King 127 on it

From ThanhVinhNguyen's Albedo working session. It covers 2026-09-26 (the idea) to 2026-09-28 (first GPU
results). It is for the lab (Fable and teammates), who will re-run the training on the lab's
infrastructure and duel the result against the King.

Numbers are **measured** unless marked otherwise. All scores are on the graded A–T judge (live since
2026-09-23). Shell commands and costs refer to our setup, so adapt paths to yours.

> **Update 2026-09-29: read §7 first.** Both v3 (DPO + NLL 50) and pure SFT were re-trained on the lab
> trainer and duelled against King 127 (100 tasks × 2 rollouts):
> - **v3 ties the King:** −0.0005, CI95 [−0.025, +0.025].
> - **SFT loses:** −0.058, CI95 [−0.095, −0.023].
>
> The diagnosis (§7.2–7.3) is that imitating GLM-5.2 transfers its style but not its skill, and caps near
> the King. Two points in §1–§5 below no longer hold:
> - the GLM reference scores are not comparable with the King's score, because the checklist is built from
>   those reference runs;
> - scaling GLM data is not the next step.
>
> §7.4–7.6 propose a new data direction: **King self-improvement** (best-vs-worst King pairs, published as
> [`vinhable/vinhable_king_selfpairs`](https://huggingface.co/datasets/vinhable/vinhable_king_selfpairs)),
> plus **hint-guided King regeneration** for the tasks where the King is systematically weak. The duel
> rollouts are public:
> [`vinhable/albedo_duel_step70_vs_king127`](https://huggingface.co/datasets/vinhable/albedo_duel_step70_vs_king127).

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
(*Update: done on 2026-09-29, v3 ties the King; see §7.1.*)

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
  (*Update: "optimistic" understates it. The checklist is extracted from, and pruned to, what these runs
  did, so these scores cannot be compared with the King's at all. See §7.3.*)
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

**v3 training curves** are in `results/training/single-v3/curves.svg`, drawn by
`scripts/plot_training_curves.py`. Means over step ranges (train batches of 16 rows):

| Steps | Loss | NLL part (50 × chosen NLL) | DPO part | Chosen NLL | Chosen / rejected log-ratio | Accuracy | Grad norm |
|---|---|---|---|---|---|---|---|
| 1–5 | 38.05 | 34.24 | 3.81 | 0.685 | +0.093 / +0.013 | 0.76 | 509 |
| 6–10 | 35.71 | 28.07 | 7.65 | 0.561 | +0.291 / −0.044 | 0.80 | 214 |
| 11–20 | 34.65 | 24.55 | 10.10 | 0.491 | +0.269 / +0.000 | 0.73 | 220 |
| 21–30 | 33.98 | 24.85 | 9.13 | 0.497 | +0.294 / −0.017 | 0.74 | 213 |
| 31–40 | 32.61 | 24.70 | 7.90 | 0.494 | +0.348 / −0.046 | 0.81 | 184 |
| 41–50 | 33.74 | 23.50 | 10.26 | 0.470 | +0.304 / +0.026 | 0.71 | 200 |
| 51–60 | 31.81 | 22.30 | 9.49 | 0.446 | +0.330 / +0.025 | 0.72 | 197 |
| 61–70 | 31.43 | 22.25 | 9.20 | 0.445 | +0.350 / +0.029 | 0.75 | 187 |

- **The loss falls because of the NLL term**: chosen NLL goes from 0.685 to 0.445 nat/token, with most of
  the drop in the first 10 steps.
- **The DPO part rises**, from 3.8 to about 9–10. This is expected with h = 51.2 × (log-ratio difference)
  and label smoothing 0.1:
  - once |h| reaches about 10–15, a misordered pair costs about 0.9·|h|, and a correct pair still costs
    0.1·h;
  - the smoothing term's gradient pushes large correct margins back down.
  - So in v3 the DPO term acts mostly as a regulariser on the margin, while NLL does the moving.

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

## 4. What is still unknown, and the risks (as of 2026-09-28; §7 answers item 1)

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

## 5. Proposed next steps (superseded by §7.6)

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
4. **Scale the data only after a positive duel, in milestones, and keep the best milestone rather than
   the biggest.** More data is not guaranteed to help: more rows also means more steps, more drift toward
   GLM's style, and weaker pairs once the strongest-gap tasks are used up. A 4k or 5k set that duels best
   is the one to develop, even if 12k is affordable. See §5.1.
5. **If the duel loses**, run ablations before spending on data:
   - pure SFT (DPO off) vs v3;
   - drop the 55 King > GLM pairs vs keep them;
   - reasoning-length control;
   - a multi-turn health check in the real harness.

### 5.1 Scaling ladder

Estimates are extrapolated linearly from the 300-task batch:
- **API cost** is about $0.10 per task. That is $0.083 for the run, plus re-running truncated tasks and
  re-judging: $30.96 for 300 tasks.
- **Yield** is 214 pairs (0.71 per task) and 1,127 train rows (3.76 per task) with the `single` builder.
- **Training time** is 3.7 s per row on 8×H200 with our per-turn trainer (70 min for 1,127 rows), at about
  $32/h (Shadeform). The lab's branch-packed forward should be several times faster; that is expected,
  not measured.

| Milestone (tasks generated) | API | Pairs | Train rows | Steps / epoch (16 rows) | 8×H200 hours / epoch (our trainer) |
|---|---|---|---|---|---|
| 300 (done) | $31 | 214 | 1,127 | 70 | 1.2 |
| 1,000 | ~$100 | ~710 | ~3,760 | ~235 | ~3.9 |
| 2,000 | ~$200 | ~1,430 | ~7,500 | ~470 | ~7.8 |
| 4,000 | ~$400 | ~2,850 | ~15,000 | ~940 | ~15.6 |
| 5,000 | ~$500 | ~3,570 | ~18,800 | ~1,175 | ~19.5 |
| 7,000 | ~$700 | ~5,000 | ~26,300 | ~1,645 | ~27 |
| 8,000 | ~$800 | ~5,700 | ~30,100 | ~1,880 | ~31 |
| 12,000 | ~$1,200 | ~8,560 | ~45,100 | ~2,820 | ~47 |

How to run the ladder so that milestones can be compared:

- **Nested sets.**
  - Rank candidate tasks once, by the robust GLM-over-King gap, keeping ~10% contrast tasks.
  - Generate in that order. Each milestone is then a prefix of the next, and only the added tasks cost
    money.
- **One fixed duel set.** Hold out duel tasks that never enter any milestone. Duel every trained milestone
  on that same set (2 × 100 samples) against King 127.
- **Fixed recipe.**
  - Keep v3's objective and 1 epoch at every milestone.
  - Larger sets take more steps at the same LR, so they drift further from the King. Check the thinking
    health and local dedup of each; if a large set degrades, try it again at a lower LR rather than
    dropping it.
- **Not every milestone needs training.**
  - Train 1k, 2k and 4k first.
  - If the duel score still rises at 4k, bracket the peak with 5k, 7k and 8k.
  - Go to 12k only if 8k still beats 7k by more than duel noise (roughly 0.01–0.02 on 100 samples; confirm
    with the second eval).
  - If the score peaks at 4k–5k, develop that set (dedup margin, LR, contrast share) instead of adding
    data.
- **The pool is the limit, not money.**
  - The current graded pool has 783 tasks with King ≤ 0.7 and GLM ≥ 0.9, and 1,836 with King < 0.9
    (§2.1).
  - Beyond ~2k tasks the ladder must either relax the selection (smaller gaps, stronger King, more ties,
    so fewer and weaker pairs per task: expect the 0.71 yield to fall) or use newly crawled graded runs.
    Each eval run adds ~100 tasks; our last crawl found 50 runs in 5 days.
  - `double` (a pair per King rollout) adds about 1.7× rows from the same tasks at no API cost. Try it at a
    milestone before paying for more tasks.

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
| `scripts/reassemble_trained_checkpoint.py`, `scripts/thinking_check_vllm.py`, `scripts/check_dedup_gate_local.py`, `scripts/plot_training_curves.py` | Post-training: full checkpoint, generation health, dedup replica, training-curve SVG |
| `scripts/run_vinhable_dpo_8xh200.sh` | Stages: setup, prep, reference, smoke, train, sweep, dedup, reassemble, think, full; lab trainer: lab-setup, lab-prep, lab-nll, lab-train; ablation |
| `scripts/prep_vinhable_for_lab.py` | Our rows → the lab trainer's branch-packed records, checked token by token against our per-turn rendering |
| `scripts/duel_checkpoints_vs_king.py`, `scripts/export_duel_rounds.py`, `ops/duel_watch.sh` | Offline duel vs the King with production's turn loop, simulator and graded judge (stages select / run `--generate-only` / score / summary); export as `able_e6`-style rounds |
| `scripts/analyze_king_pair_pool.py`, `scripts/build_king_selfpairs.py`, `scripts/push_king_selfpairs_to_hf.py`, `scripts/measure_king_nll.py` | §7.4: pool groups and headroom, the King best-vs-worst dataset, its HF card, the King-NLL gate proxy |
| `analysis/` | Ad-hoc scripts behind the numbers in §7.2–7.3 (local paths; they read the duel rollouts, the datasets and the HF reference split) |
| `results/duel-step70/`, `results/king-selfpairs/` | Duel summary and task selection; self-pair build, lab-prep check and pool analysis |
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

---

## 7. Update 2026-09-29: the duel, what it means, and a new data direction

### 7.1 Duel: v3 and SFT (lab trainer, step 70) vs King 127

**Training.** Both arms used the lab's branch-packed FSDP2 trainer on the same `vinhable_single` rows,
converted by `scripts/prep_vinhable_for_lab.py` (1,127 + 91 rows, 0 token mismatches against our per-turn
rendering). Shared settings: 16 rows per step, LR 2e-5, 5 warmup steps, cosine to 0.1×, 70 steps (1 epoch),
the same seed and the same trainable families as §3.1.

| Arm | Objective | Local dedup `rel_struct` (threshold 0.0025) | First-turn thinking check |
|---|---|---|---|
| **v3** | DPO + NLL 50 (`--loss dpo`) | step 35: 0.00262; steps 50 and 70: about 0.0027 | clean (100% closed, 0 degenerate) |
| **SFT** | NLL on the chosen side only (`--loss sft`) | step 35: 0.00309; step 70: 0.00323 (1.29×) | clean; reasoning mean 132 tokens |

Two trainer incidents:
- **The original SFT run went NaN from step 53 onward.** The duelled SFT checkpoint was continued from the
  step-50 snapshot with **fresh AdamW moments**, which may exaggerate its drift in steps 50–70.
- **v3 skipped 9 of 70 steps on non-finite gradient norms.**

We patched the lab's `train.py` in two places:
1. A NaN guard: `if math.isfinite(gn): opt.step()`, otherwise log the batch and skip the step on every
   rank.
2. `--start-step` to resume the schedule from a snapshot.

The root cause is not known. Suspects are the pinned stack (torch 2.11, transformers 5.11, fla-core 0.5.2,
triton 3.7.1, `grouped_mm`) or specific batches.

**Weights.** v3 step 70 is public at
[`vinhable/albedo-king127-v3-step70-delta`](https://huggingface.co/vinhable/albedo-king127-v3-step70-delta).
- It holds only the 486 of 1,045 tensors that differ from King CXXVII (2.8 GB), plus `reconstruct.py`.
- To rebuild: download the King at revision `e920362b`, then run `python reconstruct.py <king_dir> . <out>
  --verify-base`. The round trip was verified on all tensors and config files.
- The SFT checkpoint was not kept.

**Duel.** `scripts/duel_checkpoints_vs_king.py` ran production's turn loop against each checkpoint:
- **Tasks:** 100 held-out graded tasks in production's phase mix (cold 65, pre_edit 15, at_edit 20), with
  production horizons (62 tasks at 12 turns, 38 at 16).
- **Rollouts and environment:** 2 rollouts per task, the production simulator, and local grounding.
- **Judge:** the graded GLM-5.2 judge with production's provider pins.
- **King side:** the King's two production rollouts, re-judged locally. The local rejudge gave 0.7967 vs
  0.7942 in production.

| Side | Mean | Δ vs King (CI95) | W/T/L at ±0.05 | cold / pre_edit / at_edit |
|---|---|---|---|---|
| King 127 | 0.7967 | — | — | — |
| **v3 step 70** | **0.7963** | **−0.0005 [−0.025, +0.025]** | 23 / 42 / 31 | −0.008 / +0.065 / −0.022 |
| SFT step 70 | 0.7370 | −0.058 [−0.095, −0.023] | 24 / 23 / 50 | −0.073 / +0.016 / −0.062 |

- **Unscored readings:** 15 of 600 ("logprob tokens do not reproduce the content"), spread over all sides.
- **Cost:** about $26 of API.
- **Data:** every trajectory, score and the manifest are at
  [`vinhable/albedo_duel_step70_vs_king127`](https://huggingface.co/datasets/vinhable/albedo_duel_step70_vs_king127).
  Summaries are in `results/duel-step70/`.
- **Noise:** two rollouts of the same task differ by 0.14–0.17 on average. With 100 tasks the CI is about
  ±0.025, the same size as the win margin, so a real winner needs about +0.04, or more tasks.

### 7.2 What the rollouts show

The scripts are in `analysis/`. Bash commands are classified by a regex (the same for every model):
- **explore:** reads and searches;
- **edit:** writes to files;
- **test:** test, build and reproduction runs;
- **submit.**

| | explore / edit / submit | Rollouts that never edit | Rollouts that submit | Sub-commands per explore turn | First edit turn (median) |
|---|---|---|---|---|---|
| King | 53% / 27% / 13% | 13 / 200 | 113 | 7.8 | 4 |
| v3 | 68% / 18% / 7% | 30 / 200 | 85 | 6.3 | 6 |
| SFT | 76% / 14% / 3% | 51 / 200 | 53 | 5.1 | 6 |
| GLM, supervised turns of `vinhable_single` | 75% / 15% / 2% | — | — | 2.9 (batch 300) | 6 (batch 300) |

- **SFT copies the teacher's action mix almost exactly, and v3 lands in between.**
  - In 284 of the 1,127 rows the King is chosen and GLM rejected (`king_over_glm`). Our hypothesis is that
    DPO pushes GLM's habits down there, while SFT gets almost no gradient from those rows.
  - This is a hypothesis. The ablation that would test it is DPO on the 843 `glm_over_king` rows only.
- **SFT's deficit is concentrated:** the 10 worst tasks, mostly cold mini-coder, hold 74% of it.
- **Reasoning length is not the problem.** Chars per turn: SFT 1,863, v3 1,825, King 1,549. Thinking is
  clean in all three.
- **"Thin" actions are a correlate, not a cause.** Within the same task, command breadth barely relates to
  score: GLM reference runs (20,305) corr +0.05, King −0.03, v3 −0.07, SFT +0.07. **Do not** push the
  teacher to batch more commands per turn.
- **The judge's checklist does not ask for what we assumed.**
  - There is no question about submitting.
  - On the 100 duel tasks, question weight is explore 46%, action 22%, verification 21%, claims 10%.
  - 34 tasks have no action question, and only 46% of action questions ask for a code change.
  - Rollouts that neither edit nor submit can score 0.97–0.999.
  - So the loss is not "SFT forgot to submit". It is that SFT reaches the action and verification
    milestones less often, where a task has them.

### 7.3 Why the teacher looked strong but its student is not

1. **Selection, not a judge bias.**
   - Batch 300 was chosen where the King was weak (King 0.49) and a GLM reference was strong.
   - Only GLM-winning pairs were kept, on one noisy rollout each.
   - Both steps pick GLM's lucky tail.
2. **GLM reference scores cannot be compared with King scores.**
   - Production extracts milestones **from the reference runs** and prunes every question no reference run
     earned (`judge_api.py`, the reference-prune step). So the best-of-3 reference scores ~0.98 by
     construction.
   - Earlier "GLM ≈ King" or "GLM ≫ King" statements based on the `glm_5_2` split, ours included, are void.
   - Our own GLM regenerations were judged like candidates, and they are only comparable on the tasks we
     ran.
3. **Style transfers, skill does not.**
   - A 35B-A3B student learns the teacher's pacing and verbosity in 70 steps on about 160 tasks. It does
     not learn which file to open.
   - The result is slow and less targeted: the known failure of imitating a much stronger model on little
     data.

Consequence: **imitating GLM-5.2 is capped near the King.** v3 sits there, and more GLM data would amplify
the style drift before it adds skill. We stopped the GLM scaling ladder (§5.1).

### 7.4 New direction 1: King self-improvement from its own best rollouts (no API)

The two production rollouts of the King on the same task, scored on the same checklist by the same judge,
differ a lot. Picking the better one is a policy-improvement step (expert iteration / rejection-sampling
fine-tuning). The student learns from its own distribution, so there is no capability gap in style.

**Pool** (`scripts/analyze_king_pair_pool.py`, graded crawl of 34 runs, King 127). This leaves out the
420 tasks already used for training data, pilots and the duel.

| On 2,910 unused tasks | |
|---|---|
| King mean | 0.812 |
| King best-of-2 | **0.875** (+0.063) |
| Best of the 4 rollouts in the run (King + challenger, same judge and checklist) | 0.903 |
| Judge re-read noise on the same trajectory | 0.028 mean (0.012 median), so about 80% of the 0.14 rollout gap is real |

| Group | Share | King mean → best | Pool-mean headroom if closed to the King's best |
|---|---|---|---|
| A: King capable (best ≥ 0.85) | 67% | 0.888 → 0.951 | +0.043 |
| B: King inconsistent (gap ≥ 0.2) | 7% | 0.555 → 0.731 | +0.012 |
| C1: King weak, the run's challenger did it (≥ +0.2) | 2.4% | 0.570 → 0.897 (challenger) | +0.008 via the challenger |
| C2: King weak, nobody in the run did it | 24% | 0.690 → 0.725 | needs outside knowledge (§7.5) |

**Dataset:** [`vinhable/vinhable_king_selfpairs`](https://huggingface.co/datasets/vinhable/vinhable_king_selfpairs),
public, built by `scripts/build_king_selfpairs.py`.
- **Pairs:** 599 tasks with the King's two rollouts ≥ 0.2 apart. Chosen 0.877, rejected 0.506, median
  margin 0.31.
- **Rows:** the same behaviour-group rows and loss rules as `vinhable_single`, giving 3,530 rows (3,146
  train / 384 dev, task-level split). Phases: cold 427, pre_edit 74, at_edit 98.
- **Lab format:** converted and checked token by token, 0 mismatches. 8 rows were dropped where one side had
  no trained turn.
- **Why rows and not whole trajectories:** turn-group rows only matter for DPO. For SFT they give exactly
  the gradient of whole-trajectory rows, at a higher cost. SFT on this set is not recommended anyway,
  because the chosen text is the King's own.

**What differs between chosen and rejected** (whole trajectories, 571 pairs):
- On the surface, almost nothing: turns 12.7 vs 12.9, edits in 92% vs 92%, first edit turn 5.3 vs 5.4,
  command breadth 7.7 vs 7.5, reasoning length +4%.
- Only two differences are visible: the chosen side runs a test or reproduction more often (55% vs 48%) and
  submits more often (54% vs 47%).
- The gap is in *what* was read, concluded and changed. The DPO signal is therefore fine-grained, and
  generalisation is uncertain.

**The main risk is the dedup gate.**
- Both sides are King text, so the NLL of the chosen side under the King is near its sampling entropy, and
  the pull is weak. This is how `gen_vinhable` DPO failed (TRIVIAL/NOISE-COPY).
- The 0.2 margin and the graded judge make the labels much less luck-driven than then, but it has to be
  measured before a duel:
  1. **King NLL proxy (minutes of GPU):**

     ```
     STAGE=lab-nll DATASET=kingself bash scripts/run_vinhable_dpo_8xh200.sh
     STAGE=lab-nll DATASET=single   bash scripts/run_vinhable_dpo_8xh200.sh
     ```

     Each run does a reference pass on a 300-row subset, then reports per-token NLL (all / thinking /
     reply) for each side. The yardstick is `single`: the GLM set whose training passed the gate at only
     1.1×. Run `lab-prep` first; `ops/vinhable-dpo/machine.sh push-data` puts the rows on the box.
  2. **Short train plus local dedup** (`check_dedup_gate_local.py`) before any duel.
- If the pull is too weak, the levers are:
  - more directed steps: more pairs at margin ≥ 0.15 (841 available), or a second epoch;
  - a higher LR or NLL weight on the chosen side;
  - mixing in direction-2 data, which is off-policy by construction.

**Expected gain (a guess, not measured).** Distillation of best-of-N usually recovers 20–40% of the gap,
so about +0.013 to +0.025 per round against a win margin of +0.025. Plan for 2–3 rounds: sample new
rollouts from the improved model, rescore, rebuild the pairs.

### 7.5 New direction 2: hint-guided King regeneration for systematic weaknesses (group C2)

Self-pairs cannot fix what the King never does: C2 is 24% of the pool. GLM text fixes it only by importing
GLM's style (§7.3). The proposal is to let **the King regenerate the task with a hint**, then train on its
trajectory **without the hint**.

1. **Hint source.** Use the task's reference runs (HF `dendriteholdings/albedo`, split `glm_5_2`, free).
   The `milestones` column is production's own summary of what the reference did, with a category for
   each milestone (explore / action / verification).
2. **Rewrite the milestones into "where to look and how to check it", never the fix.** One cheap LLM call
   per task, under $0.005. Worked example from the duel set: `lingui`, King 0.70 on both rollouts, reference
   1.0.
   - The reference found in `git diff HEAD` that `extractors/typescript.ts` had been overwritten with
     `babel.ts` content, and restored it.
   - Hint: *"Before editing, check the repository's git state against HEAD and read extract.ts,
     extractors/index.ts, extractors/typescript.ts and extract.test.ts; establish which file differs from
     HEAD and why. Do not mention this note."*
   - The hint does not say "the file is corrupted" or "restore it".
3. **Placement.** Append the hint once, in a marked block, to the end of the **last observation of the
   prefix**, just before the King's first generated turn. For a cold task with no prefix turns, append it
   to the task message.
   - This is the slot our GLM reasoning nudge used.
   - Observations are role `user` and the production template keeps them, so the hint stays visible for the
     whole rollout.
   - The King gets no reasoning nudge, because it already reasons.
4. **Filter.** Keep trajectories that:
   - score well (≥ the unhinted King + 0.15);
   - never mention the hint (drop such turns, as the nudge echoes were dropped);
   - reach each conclusion **after** the evidence for it. For example, the git diff must be read before the
     restore; an edit to a file never read before is dropped. Otherwise the student learns to jump to
     answers it has no evidence for.
   - The judge's checklist comes from the same milestones as the hint, so a high score here proves little.
     **The evidence filter is the quality gate, not the score.**
5. **Train without the hint.** Strip the block, so the context is token-identical to the eval prompt, and
   check that in code. The pair is chosen = hinted King and rejected = unhinted King on the same task: same
   style, different direction, a clean contrast. Unlike self-pairs, the chosen tokens are off-policy
   (the King would not produce them without the hint), which gives the directed pull the gate needs.
6. **Open questions.**
   - Does a hint raise the King's score at all? If not, the weakness is capability, not direction.
   - What share of trajectories passes the evidence filter?
   - Does the fix generalise? Only if the weakness is a *behaviour*, such as checking git state or writing
     a reproduction; not if it is repo-specific knowledge. Before writing hints, classify the C2 tasks by
     which question tags the King loses (explore / action / verification).

**Cost.** A King rollout needs a vLLM server of the King, the simulator and the judge: about $0.05 per
rollout plus GPU time (200 rollouts take about 15 minutes on 8×H200). A pilot of 20 C2 tasks is about
$2–3 and about 30 minutes; 500 tasks × 2 rollouts is about $50.

### 7.6 Proposed next steps (replacing §5)

1. **Gate proxy for direction 1 (GPU minutes, no API).** Run `lab-nll` for `kingself` and `single`
   (§7.4). If the King's NLL on the self-pairs is far below the GLM set's, expect a gate problem and plan
   the levers before training.
2. **Direction 1 training.**
   - Recipe: v3 on `vinhable_king_selfpairs` (DPO + NLL, lab trainer). There are about 197 steps per epoch
     at 16 rows.
   - Snapshots at several steps, each with the local dedup and the thinking check.
   - Then duel the best snapshot on the same held-out 100 tasks: `duel_checkpoints_vs_king.py run` on the
     existing `duel-input.jsonl`. The self-pairs already exclude those tasks.
3. **Direction 2 pilot.**
   - First, no API: classify C2 tasks by lost question tags (behaviour vs knowledge).
   - Then 20 C2 tasks, measuring: hinted vs unhinted King score, pass rate of the evidence filter, and the
     King's NLL on the hinted trajectories with the hint stripped.
4. **Combine** only if both hold up: self-pairs for groups A/B (plus the 70 C1 challenger trajectories as
   chosen), hinted pairs for C2. Iterate: re-sample from the new model and re-pair.
5. **Carry over from this round:**
   - Save `history.jsonl` and the snapshots off the box before shutting it down. The lab-run curves of
     §7.1 were lost with the box.
   - Keep the NaN guard.
   - Mind the duel noise: about +0.04 is needed for a clear win on 100 tasks.
