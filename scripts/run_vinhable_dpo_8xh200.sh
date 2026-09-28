#!/usr/bin/env bash
# Signal test of GLM-with-thinking DPO (vinhable_single / vinhable_double) on King 127, 8xH200.
#
#   STAGE=setup      bash run_vinhable_dpo_8xh200.sh   # venvs, King weights, both datasets (~1 h)
#   STAGE=prep       ...   # contract check + sizes of the chosen dataset (CPU, minutes)
#   STAGE=reference  ...   # reference log-probs of every row (no grad)
#   STAGE=smoke      ...   # 2 steps on the longest rows: gate, peak VRAM, tokens/s, export, reassemble
#   STAGE=train      ...   # the run; metrics in $RUN/metrics.jsonl, progress in $RUN/progress.json
#   STAGE=reassemble ...   # full checkpoint from the last export (STEP=<n> to pick another)
#   STAGE=think      ...   # vLLM thinking check of the King and of the reassembled checkpoint
#
# DATASET=single|double (default single). Every stage appends to $WORK/logs/<stage>-<dataset>.log.
# Everything runs under nohup-friendly bash; start a stage with `setsid` so a dropped ssh does not
# kill it. Nothing here needs nvcc (Shadeform images have none).
set -euo pipefail

STAGE="${STAGE:?set STAGE}"
DATASET="${DATASET:-single}"
WORK="${WORK:-/workspace/vinhable}"
CODE="${CODE:-$WORK/code}"
KING_DIR="${KING_DIR:-$WORK/models/king127}"
KING_REPO="dendriteholdings/albedo-qwen3.6-35b-king-CXXVII"
KING_REV="e920362b460ae6b2a33c9cb298aa7f14a38d5584"
DATA="$WORK/data/$DATASET/data"
RUN="${RUN:-$WORK/runs/$DATASET}"
TRAIN_ENV="$WORK/venv-train"
VLLM_ENV="$WORK/venv-vllm"
GPUS="${GPUS:-8}"
mkdir -p "$WORK/logs" "$RUN"
# NCCL >= 2.2x turns NVLink SHARP on by default; VMs without Fabric Manager multicast fail with
# "Failed to bind NVLS Multicast memory" (training) or "unhandled cuda error" (vLLM TP), both seen
# on Shadeform. Plain NVLink rings are fine here. Applies to every stage.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
LOG="$WORK/logs/$STAGE-$DATASET.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -u +%FT%TZ) STAGE=$STAGE DATASET=$DATASET"

# objective and optimisation (override from the environment)
TRAIN_ARGS=(
  --beta "${BETA:-0.1}" --agg "${AGG:-mean}" --norm-tokens "${N0:-512}"
  --label-smoothing "${LS:-0.1}" --nll-weight "${NLL:-0.1}"
  --lr "${LR:-5e-6}" --min-lr-ratio "${MIN_LR_RATIO:-0.1}" --warmup-steps "${WARMUP:-5}" --epochs "${EPOCHS:-2}"
  --rows-per-rank "${ROWS_PER_RANK:-2}" --max-seq-tokens "${MAX_SEQ:-65536}"
  --eval-every "${EVAL_EVERY:-10}" --drift-every "${DRIFT_EVERY:-10}"
  --tokens-per-forward "${TOKENS_PER_FORWARD:-196608}" --targets-per-forward "${TARGETS_PER_FORWARD:-12288}"
)
# smoke on the longest rows peaked at 72 GiB/GPU with 131072 tokens per forward; 196608 uses more
# of the 141 GiB and needs fewer ZeRO-3 gathers per step
PACK_ARGS=(--tokens-per-forward "${TOKENS_PER_FORWARD:-196608}" --targets-per-forward "${TARGETS_PER_FORWARD:-12288}")

launch() {  # torchrun, one process per GPU; @record gives every rank a traceback
  export TORCHELASTIC_ERROR_FILE="$RUN/torchrun-error.json"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  "$TRAIN_ENV/bin/torchrun" --standalone --nproc_per_node "$GPUS" "$CODE/scripts/train_vinhable_dpo.py" "$@"
}

case "$STAGE" in
setup)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  nvidia-smi --query-gpu=name,memory.total --format=csv
  df -h "$WORK" | tail -1
  if [ ! -x "$TRAIN_ENV/bin/python" ]; then
    uv venv --python 3.12 "$TRAIN_ENV"
    # torch 2.14 brings Triton 3.8: flash-linear-attention refuses Triton < 3.7.1 on Hopper, whose
    # gated chunk_bwd_dqkwg kernel gives wrong gradients (fla #640). cu130 needs driver >= 580.
    uv pip install --python "$TRAIN_ENV/bin/python" "torch==2.14.0" --index-url https://download.pytorch.org/whl/cu130
    # deepspeed ops are JIT and never built here: the optimizer is torch AdamW
    # accelerate: transformers' DeepSpeed integration (HfDeepSpeedConfig) requires it
    uv pip install --python "$TRAIN_ENV/bin/python" "transformers>=5.17" "deepspeed==0.19.7" "accelerate>=1.1.0" \
      flash-linear-attention safetensors tokenizers jinja2 "huggingface_hub[hf_transfer]" numpy pytest
  fi
  if [ ! -x "$VLLM_ENV/bin/python" ]; then  # vLLM pins its own torch: separate env
    uv venv --python 3.12 "$VLLM_ENV"
    uv pip install --python "$VLLM_ENV/bin/python" vllm tokenizers jinja2
  fi
  "$TRAIN_ENV/bin/python" - <<'EOF'
import torch, transformers, deepspeed
from transformers.utils.import_utils import is_flash_linear_attention_available
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "deepspeed", deepspeed.__version__,
      "fla", is_flash_linear_attention_available())
EOF
  export HF_HUB_ENABLE_HF_TRANSFER=1
  [ -f "$KING_DIR/model.safetensors.index.json" ] || \
    "$TRAIN_ENV/bin/hf" download "$KING_REPO" --revision "$KING_REV" --local-dir "$KING_DIR"
  for d in single double; do
    "$TRAIN_ENV/bin/hf" download "vinhable/vinhable_$d" --repo-type dataset --local-dir "$WORK/data/$d"
  done
  du -sh "$KING_DIR" "$WORK/data"/*
  ;;
prep)
  "$TRAIN_ENV/bin/python" "$CODE/scripts/prepare_vinhable_dpo.py" --data "$DATA" --out "$RUN/prep" --check-every 4
  ;;
reference)
  launch --phase reference --data "$DATA" --output "$RUN" --base-model "$KING_DIR" --max-seq-tokens "${MAX_SEQ:-65536}" \
    "${PACK_ARGS[@]}"
  ;;
sweep)
  # one run, then a thinking check of every export, to pick the stopping point by generation
  # health as well as dev accuracy (the first run's final step degenerated: 19% of turns cut).
  REF="${REF:-$WORK/runs/$DATASET/reference.jsonl}"
  SWEEP="${SWEEP:-$WORK/runs/$DATASET-${TAG:-v2}}"
  # SKIP_TRAIN=1 only runs the thinking checks of an existing sweep's exports
  mkdir -p "$SWEEP/think"
  [ -n "${SKIP_TRAIN:-}" ] || RUN="$SWEEP" launch --phase train --data "$DATA" --output "$SWEEP" \
    --reference "$REF" --base-model "$KING_DIR" "${TRAIN_ARGS[@]}" --export-steps "${EXPORT_STEPS:-}"
  # the King's check does not depend on the run: reuse any earlier one
  for k in "$WORK"/runs/*/think/king127.summary.json; do
    if [ -f "$k" ] && [ ! -f "$SWEEP/think/king127.summary.json" ]; then cp "$(dirname "$k")"/king127.* "$SWEEP/think/"; fi
  done
  [ -f "$SWEEP/think/king127.summary.json" ] || \
    "$VLLM_ENV/bin/python" "$CODE/scripts/thinking_check_vllm.py" --model "$KING_DIR" --label king127 \
      --data "$DATA" --out "$SWEEP/think" || echo "thinking check failed for king127"
  for d in $(ls -d "$SWEEP"/export-step* | sort -V); do
    step="${d##*export-step}"
    [ -f "$SWEEP/think/step$step.summary.json" ] && continue
    model="$WORK/models/$DATASET-${TAG:-v2}-step$step"
    rm -rf "$model"
    "$TRAIN_ENV/bin/python" "$CODE/scripts/reassemble_trained_checkpoint.py" --king "$KING_DIR" \
      --trained "$d/trained.safetensors" --out "$model"
    "$VLLM_ENV/bin/python" "$CODE/scripts/thinking_check_vllm.py" --model "$model" --label "step$step" \
      --data "$DATA" --out "$SWEEP/think" || echo "thinking check failed for step$step"
    rm -rf "$model"  # 67 GB each; the slim export stays
  done
  "$TRAIN_ENV/bin/python" - "$SWEEP" <<'EOF'
import json, sys
from pathlib import Path
sweep = Path(sys.argv[1])
dev = {r["step"]: r for r in map(json.loads, (sweep / "metrics.jsonl").open()) if "dev_accuracy" in r}
keys = ("closed_rate", "empty_think_rate", "cut_at_limit_rate", "degenerate_rate", "one_bash_block_rate",
        "reasoning_tokens_mean", "action_chars_p90")
print("label          dev_acc  " + "  ".join(keys))
for f in sorted((sweep / "think").glob("*.summary.json"), key=lambda p: (p.stem != "king127.summary", p.stem)):
    s = json.loads(f.read_text())
    step = int(s["label"][4:]) if s["label"].startswith("step") else None
    acc = dev.get(step, {}).get("dev_accuracy")
    print(f"{s['label']:14s} {acc if acc is None else round(acc, 3)!s:7s}  " + "  ".join(
        f"{s.get(k) if s.get(k) is None else round(s[k], 3)!s:>{len(k)}}" for k in keys))
EOF
  ;;
dedup)
  # local replica of the validator's dedup gate on every export, on CPU at low priority so it can run
  # beside training; keeps polling for new exports while a sweep/full session is alive.
  GENESIS_DIR="${GENESIS_DIR:-$WORK/models/genesis}"
  [ -f "$GENESIS_DIR/model.safetensors.index.json" ] || \
    "$TRAIN_ENV/bin/hf" download dendriteholdings/albedo-qwen3.6-35b-king-genesis --local-dir "$GENESIS_DIR"
  OUTD="$WORK/runs/dedup"
  mkdir -p "$OUTD"
  export OMP_NUM_THREADS="${DEDUP_THREADS:-64}" MKL_NUM_THREADS="${DEDUP_THREADS:-64}"
  while true; do
    todo=0
    for d in $(ls -d $WORK/runs/*/export-step* 2>/dev/null | sort -V); do
      run="$(basename "$(dirname "$d")")"; step="${d##*export-step}"; tag="$run-step$step"
      [ -f "$OUTD/$tag.json" ] && continue
      [ -f "$d/trained.safetensors" ] || continue
      [ "$run" = "smoke" ] && continue
      todo=1
      model="$WORK/models/dd-$tag"
      rm -rf "$model"
      nice -n 19 "$TRAIN_ENV/bin/python" "$CODE/scripts/reassemble_trained_checkpoint.py" --king "$KING_DIR" \
        --trained "$d/trained.safetensors" --out "$model"
      (cd "$CODE" && PYTHONPATH=src nice -n 19 "$TRAIN_ENV/bin/python" scripts/check_dedup_gate_local.py \
        --candidate "$model" --king "$KING_DIR" --seed-model "$GENESIS_DIR" --device cpu --report "$OUTD/$tag.json") \
        || echo "dedup check failed for $tag"
      rm -rf "$model"
      echo "$tag: $(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['verdict'], 'rel_struct', r['rel_struct'], 'density', r['density'])" "$OUTD/$tag.json" 2>/dev/null)"
    done
    if [ "$todo" = 0 ] && ! tmux ls 2>/dev/null | grep -q -E "^(sweep|full)-"; then break; fi
    [ "$todo" = 0 ] && sleep 300
  done
  ;;
full)
  # reference -> train -> reassemble -> thinking check, unattended; each step only if the last succeeded
  STAGE=reference bash "$0"
  STAGE=train bash "$0"
  STAGE=reassemble bash "$0"
  STAGE=think bash "$0"
  ;;
smoke)
  # longest rows first: peak memory and throughput show up in the first step
  SMOKE="$RUN/smoke"
  mkdir -p "$SMOKE"
  SMOKE_ROWS=(--limit-rows 32 --limit-dev-rows 8 --longest-first)
  launch --phase reference --data "$DATA" --output "$SMOKE" --base-model "$KING_DIR" \
    --max-seq-tokens "${MAX_SEQ:-65536}" "${SMOKE_ROWS[@]}"
  launch --phase train --data "$DATA" --output "$SMOKE" --base-model "$KING_DIR" "${TRAIN_ARGS[@]}" \
    "${SMOKE_ROWS[@]}" --max-steps 2 --eval-every 2 --drift-every 2 --export-steps 2
  "$TRAIN_ENV/bin/python" "$CODE/scripts/reassemble_trained_checkpoint.py" --king "$KING_DIR" \
    --trained "$SMOKE/export-step2/trained.safetensors" --out "$WORK/models/smoke-$DATASET"
  tail -3 "$SMOKE/metrics.jsonl"
  ;;
train)
  launch --phase train --data "$DATA" --output "$RUN" --base-model "$KING_DIR" "${TRAIN_ARGS[@]}" \
    --export-steps "${EXPORT_STEPS:-}"
  ;;
reassemble)
  STEP="${STEP:-$(ls -d "$RUN"/export-step* | sed 's/.*export-step//' | sort -n | tail -1)}"
  "$TRAIN_ENV/bin/python" "$CODE/scripts/reassemble_trained_checkpoint.py" --king "$KING_DIR" \
    --trained "$RUN/export-step$STEP/trained.safetensors" --out "$WORK/models/$DATASET-step$STEP"
  ;;
think)
  STEP="${STEP:-$(ls -d "$WORK/models/$DATASET"-step* | sed 's/.*-step//' | sort -n | tail -1)}"
  [ -f "$RUN/think/king127.summary.json" ] || \
    "$VLLM_ENV/bin/python" "$CODE/scripts/thinking_check_vllm.py" --model "$KING_DIR" --label king127 \
      --data "$DATA" --out "$RUN/think"
  "$VLLM_ENV/bin/python" "$CODE/scripts/thinking_check_vllm.py" --model "$WORK/models/$DATASET-step$STEP" \
    --label "$DATASET-step$STEP" --data "$DATA" --out "$RUN/think"
  ;;
*)
  echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
echo "=== $(date -u +%FT%TZ) STAGE=$STAGE done"
