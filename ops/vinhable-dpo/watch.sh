#!/usr/bin/env bash
# Low-frequency watcher for a detached stage on the box (default: the `full` chain).
# Polls every INTERVAL seconds (default 600), appends one line per poll to a local log, and exits
# only on an event, so whoever launched it in the background is woken once:
#   done      the tmux session printed "[stage finished"
#   error     a Traceback in the stage logs
#   nan       NaN/inf in the latest metrics
#   stalled   progress.json / rank0.log unchanged for STALL_MIN minutes (default 40)
#
#   HOST=... DATASET=single bash ops/vinhable-dpo/watch.sh
set -uo pipefail

HOST="${HOST:?set HOST}"
PORT="${PORT:-22}"
SSH_USER="${SSH_USER:-shadeform}"
KEY="${KEY:-$HOME/.ssh/albedo_vast_ed25519}"
WORK="${WORK:-/workspace/vinhable}"
DATASET="${DATASET:-single}"
STAGE="${STAGE:-full}"
INTERVAL="${INTERVAL:-600}"
STALL_MIN="${STALL_MIN:-40}"
LOCAL_LOG="${LOCAL_LOG:-/e/albedo-storage-temp/vinhable-dpo-runs/watch-$STAGE-$DATASET.log}"
mkdir -p "$(dirname "$LOCAL_LOG")"
SSH=(ssh -i "$KEY" -p "$PORT" -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 "$SSH_USER@$HOST")

probe() {
  "${SSH[@]}" "
    R=$WORK/runs/${RUN_SUBDIR:-$DATASET}; L=$WORK/logs
    last=\$(tail -n 1 \$R/metrics.jsonl 2>/dev/null | cut -c1-400)
    rank0=\$(tail -n 1 \$R/rank0.log 2>/dev/null | cut -c1-200)
    age=\$(( (\$(date +%s) - \$(stat -c %Y \$R/rank0.log 2>/dev/null || date +%s)) / 60 ))
    finished=\$(tmux capture-pane -p -t $STAGE-$DATASET 2>/dev/null | grep -c '\[stage finished')
    errors=\$(cat \$L/*-$DATASET.log 2>/dev/null | grep -c 'Traceback')
    gpu=\$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | awk -F, '{u+=\$1; m+=\$2} END {printf \"%d%%/%dGiB\", u/NR, m/NR/1024}')
    echo \"FIN=\$finished ERR=\$errors AGE=\$age GPU=\$gpu\"
    echo \"RANK0 \$rank0\"
    echo \"LAST \$last\"
  " 2>&1
}

errors_at_start=$(probe | sed -n 's/.*ERR=\([0-9]*\).*/\1/p' | head -1)
errors_at_start=${errors_at_start:-0}
echo "$(date '+%F %T') watching $STAGE-$DATASET every ${INTERVAL}s (tracebacks already in logs: $errors_at_start)" >> "$LOCAL_LOG"
while true; do
  out="$(probe)"
  status="$(echo "$out" | head -1)"
  echo "$(date '+%F %T') $status | $(echo "$out" | sed -n 's/^RANK0 //p' | cut -c1-160)" >> "$LOCAL_LOG"
  fin=$(echo "$status" | sed -n 's/.*FIN=\([0-9]*\).*/\1/p'); err=$(echo "$status" | sed -n 's/.*ERR=\([0-9]*\).*/\1/p')
  age=$(echo "$status" | sed -n 's/.*AGE=\([0-9]*\).*/\1/p')
  event=""
  [ "${fin:-0}" -gt 0 ] && event="done"
  [ "${err:-0}" -gt "$errors_at_start" ] && event="error"
  # only the training signals: the final drift record legitimately holds a NaN increment ratio
  echo "$out" | grep '^LAST' | grep -q -E '"(loss|grad_norm)": (NaN|-?Infinity)' && event="nan"
  [ "${age:-0}" -ge "$STALL_MIN" ] && event="stalled"
  if [ -n "$event" ]; then
    echo "$(date '+%F %T') EVENT=$event" >> "$LOCAL_LOG"
    echo "EVENT=$event"; echo "$out"
    exit 0
  fi
  sleep "$INTERVAL"
done
