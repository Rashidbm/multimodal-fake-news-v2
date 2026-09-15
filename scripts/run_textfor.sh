#!/usr/bin/env bash
# One-shot Text Fluoroscopy feature extraction for macOS / Linux.
#
#   bash scripts/run_textfor.sh                      # Qwen3.5-9B on the GPU box
#   bash scripts/run_textfor.sh --batch-size 2       # any fnd.extract_textfor flag
#   SMOKE=1 bash scripts/run_textfor.sh              # Qwen2-0.5B on CPU, 8 rows
#
# Needs data/processed/balanced_5group.csv from scripts/run_dataset.sh.
# Downloads Qwen3.5-9B (~18 GB) into ~/.cache/huggingface on first run; needs a
# recent transformers. If the repo is gated, run `huggingface-cli login` once.
# Output in features/v_textfor.pt (gitignored) and outputs/textfor_probe/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec > >(tee -a run_textfor.log) 2>&1

PY=python3; command -v python3 >/dev/null || PY=python
$PY --version

CSV=data/processed/balanced_5group.csv

echo "===== install ====="
$PY -m pip install -r requirements.txt

echo "===== unit tests ====="
$PY -m pytest fnd/tests/test_text_fluoroscopy.py -q

test -f "$CSV" || { echo "run scripts/run_dataset.sh first"; exit 1; }

if [ "${SMOKE:-0}" = "1" ]; then
  echo "===== smoke test: plumbing only, small model on CPU ====="
  $PY -m fnd.extract_textfor --csv "$CSV" --out features/v_textfor_smoke.pt \
      --model Qwen/Qwen2-0.5B-Instruct --device cpu --limit 8 --batch-size 2 "$@"
  exit 0
fi

# 200 rows first: this file is what stages 4 and 5 build against while the
# full extraction runs. Catching a shape or alignment problem here costs a
# minute; catching it after the full run costs two hours.
echo "===== subset (200 rows) ====="
$PY -m fnd.extract_textfor --csv "$CSV" --out features/v_textfor_subset.pt --limit 200 "$@"

echo "===== full extraction ====="
$PY -m fnd.extract_textfor --csv "$CSV" --out features/v_textfor.pt "$@"

echo "===== meta ====="
cat features/v_textfor.json

# The extractor produces vectors, not predictions. The probe trains a small
# head on them and reports what this stream scores on its own, beside the
# baselines, so the number can be read as good or bad.
echo "===== probe (scores for this stream alone) ====="
$PY -m fnd.probe_textfor --features features/v_textfor.pt --csv "$CSV" \
    --out outputs/textfor_probe

# The domain-matched control (same topic, different authorship) reuses the
# cached vectors, so it costs minutes rather than another extraction:
#   python scripts/enrich_provenance.py --csv "$CSV" --raw data/raw/MMFakeBench \
#       --out data/processed/enriched.csv
#   $PY -m fnd.probe_textfor --features features/v_textfor.pt \
#       --csv data/processed/enriched.csv --where domain=gossip \
#       --out outputs/textfor_probe_gossip
