#!/usr/bin/env bash
# Push latest repo from Mac to AutoDL (no GitHub). Run on your Mac:
#
#   bash scripts/linux/sync_from_mac.sh
#   REMOTE_HOST=autodl-mempatch REMOTE_DIR=/root/autodl-tmp/MemPatch bash scripts/linux/sync_from_mac.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-autodl-mempatch}"
REMOTE_DIR="${REMOTE_DIR:-/root/autodl-tmp/MemPatch}"

echo "Sync $ROOT -> ${REMOTE_HOST}:${REMOTE_DIR}"
echo "(excludes local/, .venv/, caches, gitignored artifacts)"

rsync -avz --progress \
  --exclude '.git/' \
  --exclude 'local/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.pycache_compile/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$ROOT/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Done. On server: cd ${REMOTE_DIR} && git log -1 --oneline 2>/dev/null || true"
