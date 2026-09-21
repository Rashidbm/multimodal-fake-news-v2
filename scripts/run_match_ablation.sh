#!/usr/bin/env bash
# Match vs. mismatch: extract both backbones, probe each, compare head to head.
#
#   bash scripts/run_match_ablation.sh                  # the full experiment
#   bash scripts/run_match_ablation.sh --limit 200      # any fnd.extract_match flag
#   SMOKE=1 bash scripts/run_match_ablation.sh          # CLIP only, 32 rows, CPU
#   VLM_BATCH=2 bash scripts/run_match_ablation.sh      # smaller VLM batches
#
# Needs data/processed/balanced_5group.csv from scripts/run_dataset.sh and the
# images it points at. First run downloads CLIP-L (~1.7 GB) and Qwen2.5-VL-3B
# (~7 GB) into ~/.cache/huggingface. CLIP takes minutes; the VLM takes hours,
# which is exactly why both are cached and never re-extracted.
# Output in features/ (gitignored) and outputs/match_*/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec > >(tee -a run_match.log) 2>&1

PY=python3; command -v python3 >/dev/null || PY=python
$PY --version

CSV=data/processed/balanced_5group.csv
VLM_BATCH="${VLM_BATCH:-4}"

echo "===== install ====="
$PY -m pip install -r requirements.txt

echo "===== unit tests (no downloads) ====="
$PY -m pytest fnd/tests/test_match_mismatch.py fnd/tests/test_compare_match.py -q

test -f "$CSV" || { echo "run scripts/run_dataset.sh first"; exit 1; }

if [ "${SMOKE:-0}" = "1" ]; then
  echo "===== smoke test: plumbing only, CLIP on CPU, 32 rows ====="
  $PY -m fnd.extract_match --csv "$CSV" --backbone clip --device cpu \
      --limit 32 --batch-size 4 --out features/v_match_clip_smoke.pt "$@"
  exit 0
fi

# CLIP first: it is cheap, and if anything is wrong with the CSV, the image
# paths or the splits, it fails in minutes rather than hours into the VLM.
echo "===== extract: clip ====="
$PY -m fnd.extract_match --csv "$CSV" --backbone clip \
    --out features/v_match_clip.pt "$@"

echo "===== extract: qwen-vl ====="
$PY -m fnd.extract_match --csv "$CSV" --backbone qwenvl --batch-size "$VLM_BATCH" \
    --out features/v_match_qwenvl.pt "$@"

# Each backbone on its own first: the comparison table is deliberately narrow,
# and the per-scenario breakdown here is what says WHAT the stream detects.
echo "===== probe: clip alone ====="
$PY -m fnd.probe_match --features features/v_match_clip.pt --csv "$CSV" \
    --out outputs/match_probe_clip

echo "===== probe: qwen-vl alone ====="
$PY -m fnd.probe_match --features features/v_match_qwenvl.pt --csv "$CSV" \
    --out outputs/match_probe_qwenvl

echo "===== head to head ====="
$PY -m fnd.compare_match --csv "$CSV" \
    --features clip=features/v_match_clip.pt \
    --features qwenvl=features/v_match_qwenvl.pt \
    --out outputs/match_compare

echo
echo "The table to read: outputs/match_compare/comparison.md"
echo "See docs/MATCH_MISMATCH.md for what each row means and what to do with it."
