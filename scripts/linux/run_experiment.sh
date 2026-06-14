#!/usr/bin/env bash
# Unified experiment entrypoint: smoke or formal.
set -euo pipefail
mode="${1:?usage: run_experiment.sh smoke|formal}"
case "$mode" in
  smoke) exec bash "$(dirname "$0")/run_smoke_no_lora.sh" ;;
  formal) exec bash "$(dirname "$0")/run_formal.sh" ;;
  *) echo "unknown mode: $mode (expected smoke or formal)" >&2; exit 2 ;;
esac
