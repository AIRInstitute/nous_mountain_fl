#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m mountain_pass_fl.run_extended --data-dir data/raw --out-dir outputs_extended --config configs/extended.yaml --rebuild-segments
