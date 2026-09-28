#!/usr/bin/env bash
# Start production's repo-context (grounding) service on this machine, for local simulators.
#
# Needs, before the first run:
#   - the public production manifest at $STORE/manifest.json (sha256 e3cff617..., see
#     docs/LOCAL_GROUNDING.md);
#   - a GitHub token in the repo's .env as ALBEDO_REPO_CONTEXT_GITHUB_TOKEN=... (never on the
#     command line, never in chat). Without it GitHub allows 60 requests/hour and open-swe-traces /
#     swe-hero samples cannot be grounded.
#
# Usage (Git Bash):  bash scripts/run_repo_context_local.sh
# Then:              curl http://127.0.0.1:8093/healthz
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STORE="${STORE:-E:/albedo-storage-temp/repo-context}"

export ALBEDO_REPO_CONTEXT_CACHE_DIR="$STORE/cache"
export ALBEDO_REPO_CONTEXT_DATASET_MANIFEST_PATH="$STORE/manifest.json"
export ALBEDO_REPO_CONTEXT_DATASET_MANIFEST_HASH="e3cff61772b0096811d4c5d8bbc8dee8dacbd9a069bc4557608adf1c1c2ddf40"
export ALBEDO_REPO_CONTEXT_MAX_CACHE_GB="${MAX_CACHE_GB:-15}"
export ALBEDO_REPO_CONTEXT_API_HOST=127.0.0.1
export ALBEDO_REPO_CONTEXT_API_PORT="${PORT:-8093}"
export PYTHONPATH="$ROOT/src"

cd "$ROOT"  # pydantic-settings reads .env from the working directory
if ! grep -q '^ALBEDO_REPO_CONTEXT_GITHUB_TOKEN=..' .env 2>/dev/null; then
  echo "missing ALBEDO_REPO_CONTEXT_GITHUB_TOKEN in $ROOT/.env" >&2
  exit 1
fi
if [ ! -f "$ALBEDO_REPO_CONTEXT_DATASET_MANIFEST_PATH" ]; then
  echo "missing manifest: $ALBEDO_REPO_CONTEXT_DATASET_MANIFEST_PATH" >&2
  exit 1
fi
mkdir -p "$ALBEDO_REPO_CONTEXT_CACHE_DIR"
# the wrapper only swaps a temp-file helper Windows cannot use; the service itself is upstream's
exec py -3 "$ROOT/scripts/repo_context_windows.py"
