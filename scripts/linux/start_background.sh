#!/usr/bin/env bash
# Start 3-model paper pipeline in a detached screen session.
#
#   export LOCAL_ROOT=/root/autodl-tmp/mempatch_local
#   bash scripts/linux/start_background.sh
# Optional: export HF_TOKEN=... only when downloading gated weights (local models skip this).
set -euo pipefail
LINUX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$LINUX_DIR/../.." && pwd)"

export LOCAL_ROOT="${LOCAL_ROOT:-/root/autodl-tmp/mempatch_local}"
export HF_HOME="${HF_HOME:-$LOCAL_ROOT/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
export PIPELINE_LOG="${PIPELINE_LOG:-$LOCAL_ROOT/logs/pipeline.log}"

if [[ -f "$ROOT/.env" && -z "${HF_TOKEN:-}" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

mkdir -p "$LOCAL_ROOT/logs" "$LOCAL_ROOT/models" "$LOCAL_ROOT/hf_cache"
chmod +x "$LINUX_DIR"/*.sh 2>/dev/null || true

RUNNER="$LOCAL_ROOT/run_paper_campaign.sh"
{
  cat <<EOF
#!/usr/bin/env bash
set -eo pipefail
export LOCAL_ROOT="$LOCAL_ROOT"
export HF_HOME="$HF_HOME"
export HF_ENDPOINT="$HF_ENDPOINT"
export HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET"
EOF
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf 'export HF_TOKEN=%q\n' "$HF_TOKEN"
  fi
  cat <<EOF
export PYTHON="$PYTHON"
export PIPELINE_LOG="$PIPELINE_LOG"
cd "$ROOT"
exec bash "$LINUX_DIR/run_paper_campaign.sh" >>"\$PIPELINE_LOG" 2>&1
EOF
} >"$RUNNER"
chmod +x "$RUNNER"

screen -X -S mempatch quit 2>/dev/null || true
screen -dmS mempatch bash "$RUNNER"
sleep 2
screen -ls
echo "Log: $PIPELINE_LOG"
echo "Attach: screen -r mempatch"
