#!/usr/bin/env bash
# Run the four fact-check comparisons on one manifest, then write comparison.md.
#   bash scripts/run_factcheck_pilot.sh MANIFEST OUTPUT_DIR [extra evaluate options]
# Extra options (e.g. --exclude-domain example.org --cutoff-date 2024-01-01) are passed
# to every run. Re-running the same command resumes; recorded errors are kept. To retry
# failures after changing code or configuration, use a fresh OUTPUT_DIR.
set -euo pipefail

[ $# -ge 2 ] || { echo "usage: $0 MANIFEST OUTPUT_DIR [evaluate options]" >&2; exit 2; }
MANIFEST="$1"; OUT="$2"; shift 2
PY="${PYTHON:-python}"
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }
mkdir -p "$OUT"
[ -f outputs/factcheck/container-images.json ] && cp outputs/factcheck/container-images.json "$OUT/"
git rev-parse HEAD > "$OUT/git-commit.txt" 2>/dev/null || true
git status --porcelain > "$OUT/git-status.txt" 2>/dev/null || true

"$PY" -m src.factcheck doctor "$@"

"$PY" -m src.factcheck evaluate --manifest "$MANIFEST" --output "$OUT" --mode closed_book "$@"
"$PY" -m src.factcheck evaluate --manifest "$MANIFEST" --output "$OUT" --mode search "$@"
"$PY" -m src.factcheck evaluate --manifest "$MANIFEST" --output "$OUT" --mode direct --evidence-from "$OUT/search" "$@"
"$PY" -m src.factcheck evaluate --manifest "$MANIFEST" --output "$OUT" --mode assessed --evidence-from "$OUT/search" "$@"
"$PY" -m src.factcheck compare "$OUT"
