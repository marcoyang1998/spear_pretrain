#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/yangxiaoyu/miniconda3/envs/encoder/bin/python}
"$PYTHON" -m unittest discover -s mvq/tests -v
