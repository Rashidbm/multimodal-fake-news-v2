# Qwen3-VL-Embedding semantic experiment: final result files

Compact outputs copied from the Training PC (`D:\MultiGuard\semantic_arms`, produced 2026-09-28).
Narrative and caveats: [docs/SEMANTIC_QWEN_EXPERIMENT.md](../../docs/SEMANTIC_QWEN_EXPERIMENT.md).

| File | Contents |
|---|---|
| `comparison.json` | Test metrics for the released CLIP-L model and the Qwen arm, same scorer |
| `evaluation.json` | Qwen arm test metrics, selected candidate and threshold |
| `selection.json` | Validation-selected candidate (`extra1_ooc0.125_c0.01`), hashes, feature_dim 5634 |
| `results.json` | All fitted candidates with validation metrics |
| `qwen_embedding_predictions.csv` | Per-row test probabilities (1,235 rows) |

Not included: embedding caches (`*.pt`), fitted classifiers (`*.joblib`) and model weights.
