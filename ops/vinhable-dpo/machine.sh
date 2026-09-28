#!/usr/bin/env bash
# Local helper for the rented 8xH200 box (Git Bash on Windows).
#
#   HOST=1.2.3.4 [PORT=22] [SSH_USER=shadeform] bash ops/vinhable-dpo/machine.sh push   # code -> box
#   HOST=... bash ops/vinhable-dpo/machine.sh stage setup ["LR=1e-5 ..."]             # stage in a detached tmux session
#   HOST=... bash ops/vinhable-dpo/machine.sh attach setup-single                     # watch it live (Ctrl-b d to leave)
#   HOST=... bash ops/vinhable-dpo/machine.sh status                                  # sessions, GPU use, disk
#   HOST=... bash ops/vinhable-dpo/machine.sh tail setup                              # follow a stage log
#   HOST=... bash ops/vinhable-dpo/machine.sh pull                                    # results -> E:
#
# The key never leaves this machine: ssh uses it from ~/.ssh. Nothing secret is copied to the box.
set -euo pipefail

HOST="${HOST:?set HOST}"
PORT="${PORT:-22}"
SSH_USER="${SSH_USER:-shadeform}"
KEY="${KEY:-$HOME/.ssh/albedo_vast_ed25519}"
WORK="${WORK:-/workspace/vinhable}"
DATASET="${DATASET:-single}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_OUT="${LOCAL_OUT:-/e/albedo-storage-temp/vinhable-dpo-runs}"
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$SSH_USER@$HOST")

case "${1:?push|stage|attach|status|tail|pull|ssh}" in
push)
  bundle="$(mktemp -d)/code.tgz"
  tar -C "$ROOT" -czf "$bundle" \
    scripts/vinhable_dpo_data.py scripts/prepare_vinhable_dpo.py scripts/train_vinhable_dpo.py \
    scripts/reassemble_trained_checkpoint.py scripts/thinking_check_vllm.py scripts/run_vinhable_dpo_8xh200.sh \
    scripts/make_tiny_qwen35moe.py tests/test_vinhable_dpo_trainer.py \
    assets/tokenizers/Qwen3.6-35B-A3B
  "${SSH[@]}" "sudo mkdir -p $WORK && sudo chown \$(id -u):\$(id -g) $WORK && mkdir -p $WORK/code"
  scp -i "$KEY" -P "$PORT" "$bundle" "$SSH_USER@$HOST:$WORK/code.tgz"
  "${SSH[@]}" "tar -C $WORK/code -xzf $WORK/code.tgz && ls $WORK/code/scripts"
  ;;
stage)
  stage="${2:?stage name}"
  shift 2
  extra="$*"
  # a detached tmux session per stage: it runs on the box whatever happens to this connection,
  # and `attach` shows it live; the shell stays open afterwards so the last output can be read
  session="$stage-$DATASET"
  "${SSH[@]}" "tmux has-session -t $session 2>/dev/null && { echo 'session $session exists (attach or kill it first)'; exit 1; }; \
    tmux new-session -d -s $session \"cd $WORK && env $extra STAGE=$stage DATASET=$DATASET WORK=$WORK bash $WORK/code/scripts/run_vinhable_dpo_8xh200.sh; echo; echo '[stage finished, exit '\\\$?']'; exec bash\" \
    && echo started tmux session $session"
  ;;
attach)
  # ssh -t so tmux gets a terminal; detach again with Ctrl-b d (the stage keeps running)
  ssh -t -i "$KEY" -p "$PORT" "$SSH_USER@$HOST" "tmux attach -t ${2:?session, e.g. train-single}"
  ;;
status)
  "${SSH[@]}" "tmux ls 2>/dev/null || echo 'no tmux sessions'; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1"
  ;;
tail)
  stage="${2:?stage name}"
  "${SSH[@]}" "tail -n ${LINES:-40} -f $WORK/logs/$stage-$DATASET.log"
  ;;
pull)
  mkdir -p "$LOCAL_OUT"
  # small results only; exports are ~3 GB each and come separately (PULL_EXPORTS=1)
  # GNU tar applies --exclude only to names after it: the patterns must precede logs/runs
  "${SSH[@]}" "cd $WORK && tar -czf - --exclude='*.safetensors' --exclude='reference.rank*' logs runs" \
    | tar -C "$LOCAL_OUT" -xzf -
  if [ "${PULL_EXPORTS:-0}" = 1 ]; then
    "${SSH[@]}" "cd $WORK && tar -cf - runs/*/export-step*" | tar -C "$LOCAL_OUT" -xf -
  fi
  ls -R "$LOCAL_OUT" | head -40
  ;;
ssh)
  "${SSH[@]}"
  ;;
esac
