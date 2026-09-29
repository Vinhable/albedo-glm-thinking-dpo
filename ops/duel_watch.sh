#!/usr/bin/env bash
# Poll a duel generation run every 10s, write progress-live.txt, exit on done / crash / stall.
# usage: bash ops/duel_watch.sh <label> <target_rows> <log_file>
OUT=/e/albedo-storage-temp/duel-sft-dpo-vs-king-20260928
LABEL=$1; TARGET=$2; LOG=$3
LIVE=$OUT/progress-live.txt
while true; do
  gen=$(grep -cE "\"label\": ?\"$LABEL\".*\"status\": ?\"generated\"" "$OUT/generated.jsonl" 2>/dev/null)
  err=$(grep -c '"status": "error"' "$LOG" 2>/dev/null)
  age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  {
    echo "updated: $(date '+%F %T')   label: $LABEL"
    echo "generated: $gen / $TARGET   errors: $err   log idle: ${age}s"
    echo "--- last rollouts ---"
    grep '"status"' "$LOG" | tail -5
  } > "$LIVE"
  if grep -q '"generated_rows"' "$LOG"; then echo "DONE $LABEL gen=$gen err=$err"; tail -1 "$LOG"; exit 0; fi
  if grep -qE '^Traceback|SystemExit' "$LOG"; then echo "CRASH $LABEL gen=$gen"; grep -A3 -E '^Traceback|SystemExit' "$LOG" | tail -6; exit 1; fi
  if [ "$age" -gt 600 ]; then echo "STALL $LABEL gen=$gen idle=${age}s"; exit 2; fi
  sleep 10
done
