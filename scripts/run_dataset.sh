#!/usr/bin/env bash
# One-shot dataset build for macOS / Linux.
#
#   bash scripts/run_dataset.sh
#
# Steps: install deps -> unit tests -> download MMFakeBench -> extract every
# zip -> inspect (all images must exist) -> build balanced dataset -> verify.
# Everything is also written to run_dataset.log in the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec > >(tee -a run_dataset.log) 2>&1

step() { printf '\n===== %s =====\n' "$1"; }

step "0. Python"
PY=python3; command -v python3 >/dev/null || PY=python
$PY --version

step "1. HuggingFace token (read-only; owner revokes it after the download)"
export HF_TOKEN="${HF_TOKEN:-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}"

step "2. Install dependencies"
$PY -m pip install --upgrade pip
$PY -m pip install -r requirements.txt huggingface_hub

step "3. Unit tests"
$PY -m pytest fnd/tests -q

step "4. Download MMFakeBench (resumes if interrupted)"
$PY -c "from huggingface_hub import snapshot_download; import os; snapshot_download('liuxuannan/MMFakeBench', repo_type='dataset', local_dir='data/raw/MMFakeBench', token=os.environ['HF_TOKEN'])"
ls -la data/raw/MMFakeBench

step "5. Extract every zip archive"
find data/raw/MMFakeBench -name '*.zip' | while read -r z; do
  echo "extracting $z"; unzip -q -o "$z" -d data/raw/MMFakeBench
done
find data/raw/MMFakeBench -maxdepth 3 -type d | sort | head -60

step "6. Inspect: map every record, require every image file"
$PY -m fnd.data.mmfakebench --root data/raw/MMFakeBench --require-images

step "7. Build the balanced dataset and verify"
$PY -m fnd.data.build --mmfakebench data/raw/MMFakeBench --out data/processed --images required

step "DONE  ->  data/processed/balanced_5group.csv  (log: run_dataset.log)"
