#!/usr/bin/env bash
# One-shot FND-CLIP training for macOS / Linux.
#
#   bash scripts/run_train.sh                # defaults: 5 epochs, batch 32
#   bash scripts/run_train.sh --epochs 3     # any fnd.train flag can be appended
#
# Needs data/processed/balanced_5group.csv from scripts/run_dataset.sh.
# Downloads bert-base-uncased, openai/clip-vit-base-patch32 and the ResNet-50
# ImageNet weights on first run (about 1 GB).  Output in outputs/fnd_clip_v1/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec > >(tee -a run_train.log) 2>&1

PY=python3; command -v python3 >/dev/null || PY=python
$PY --version

echo "===== install ====="
$PY -m pip install -r requirements.txt

echo "===== unit tests ====="
$PY -m pytest fnd/tests -q

echo "===== train FND-CLIP ====="
test -f data/processed/balanced_5group.csv || { echo "run scripts/run_dataset.sh first"; exit 1; }
$PY -m fnd.train --csv data/processed/balanced_5group.csv --out outputs/fnd_clip_v1 "$@"

echo "===== test metrics ====="
cat outputs/fnd_clip_v1/test_metrics.json
