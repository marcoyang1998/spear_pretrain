#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${ICEFALL_ROOT:-../icefall}:${PYTHONPATH:-}"
PYTHON=${PYTHON:-/home/yangxiaoyu/miniconda3/envs/encoder/bin/python}
"$PYTHON" -m unittest discover -s tests -v
