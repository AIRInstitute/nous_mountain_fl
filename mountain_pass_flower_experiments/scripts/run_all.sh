#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m mountain_pass_fl.run_all --data-dir data/raw --out-dir outputs --config configs/default.yaml --rebuild-segments "$@"
