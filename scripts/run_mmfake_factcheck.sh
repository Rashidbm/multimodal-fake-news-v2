#!/usr/bin/env bash
# One command: start the backends, wait for the model, check them, build the
# MMFakeBench manifest and run the four comparisons.
#
#   bash scripts/run_mmfake_factcheck.sh MMFAKEBENCH_ROOT [val-smoke|test-pilot]
#
# MMFAKEBENCH_ROOT is the folder with MMFakeBench_val.json / MMFakeBench_test.json and
# the extracted MMFakeBench_val/ and MMFakeBench_test/ image folders.
#   val-smoke   (default) 20 balanced validation examples: fix execution failures here
#   test-pilot  200 balanced test examples: run once, after the configuration is frozen
# Safe to re-run: existing manifests are reused and finished examples are skipped.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
[ $# -ge 1 ] || { echo "usage: $0 MMFAKEBENCH_ROOT [val-smoke|test-pilot]" >&2; exit 2; }
DATA="$1"
STAGE="${2:-val-smoke}"
case "$STAGE" in
  val-smoke)  SPLIT=val;  LIMIT=20;  NAME=gemma4-val-smoke ;;
  test-pilot) SPLIT=test; LIMIT=200; NAME=gemma4-test-pilot ;;
  *) echo "stage must be val-smoke or test-pilot" >&2; exit 2 ;;
esac
if [ ! -f "$DATA/MMFakeBench_${SPLIT}.json" ] && [ -z "$(find "$DATA" -name '*.arrow' -print -quit 2>/dev/null)" ]; then
  echo "error: $DATA has no MMFakeBench_${SPLIT}.json (or Arrow files)" >&2
  exit 1
fi

# 1. Python environment.
if [ ! -d .venv-factcheck ]; then
  python3 -m venv .venv-factcheck
fi
# shellcheck disable=SC1091
source .venv-factcheck/bin/activate
pip install -q -r requirements-factcheck.txt

# 2. Backends.
bash scripts/start_factcheck_local.sh
# shellcheck disable=SC1091
source deploy/factcheck/local.env.example

# 3. Wait for the model (first start downloads the weights) and SearXNG.
COMPOSE=(docker compose -f deploy/factcheck/compose.yaml)
echo "waiting for the model server at $FACTCHECK_VLM_URL (first start can take a long time) ..."
for i in $(seq 1 480); do
  if curl -sf "$FACTCHECK_VLM_URL/models" >/dev/null 2>&1; then break; fi
  state="$("${COMPOSE[@]}" ps --format '{{.State}}' vlm 2>/dev/null || true)"
  if [ "$state" != "running" ] && [ "$state" != "restarting" ] && [ "$i" -gt 2 ]; then
    echo "error: the vlm container is '$state'. Last log lines:" >&2
    "${COMPOSE[@]}" logs --tail 40 vlm >&2
    exit 1
  fi
  if [ $((i % 10)) -eq 0 ]; then
    echo "  still loading ($((i / 2)) min). Latest log line:"
    "${COMPOSE[@]}" logs --tail 1 vlm 2>/dev/null | sed 's/^/    /'
  fi
  sleep 30
done
curl -sf "$FACTCHECK_VLM_URL/models" >/dev/null || { echo "error: model server not ready after 4 hours" >&2; exit 1; }
for i in $(seq 1 30); do curl -sf "$FACTCHECK_SEARCH_URL/" >/dev/null 2>&1 && break; sleep 2; done

# 4. Manifest (reused if it already exists) and the four runs + comparison.
MANIFEST="outputs/factcheck/mmfake-${SPLIT}-${LIMIT}.jsonl"
python -m src.factcheck prepare-mmfake --root "$DATA" --split "$SPLIT" --balanced --limit "$LIMIT" --output "$MANIFEST"
bash scripts/run_factcheck_pilot.sh "$MANIFEST" "outputs/factcheck/$NAME"

echo
echo "Done. Results: outputs/factcheck/$NAME/comparison.md (and one report.md per mode)."
