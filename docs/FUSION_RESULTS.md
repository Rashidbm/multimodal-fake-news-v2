# Fusion results

These are results for the **V3 fusion module**, not for the Qwen3-VL-Embedding semantic experiment
(see [SEMANTIC_QWEN_EXPERIMENT.md](SEMANTIC_QWEN_EXPERIMENT.md)). The semantic input is the CLIP-L
semantic model.

`fnd.train_fusion` trains `V3FusionModule` (`fnd.models.pipeline_v3`, tokenized pairwise
cross-attention, commit 607d9f5) on cached branch vectors. Only the fusion module trains; the
branches are frozen. Data: the team dataset `fnd_team_dataset_2026-09-14` (17,794 train, 2,375
validation, 1,235 test with 247 per scenario).

## Fused features

| Input | Shape | Producer |
|---|---|---|
| `semantic` | `[N, 768]` | `models/semantic` bundle (CLIP-L semantic model) |
| `image` | `[N, 768]` | `models/image/news` bundle |
| `text` | `[N, 4096]` | Qwen3.5-9B layer 30, masked mean, max length 64 ([TEXT_FLUOROSCOPY.md](TEXT_FLUOROSCOPY.md)) |

(The text branch's Qwen3.5-9B model is unrelated to the rejected Qwen3-VL-Embedding semantic arm.)

Train and validation vectors are the release asset `fusion-training-features.npz` (20,162 rows; 7
train/validation rows whose image cache was unavailable are excluded). Recomputing 16 of its rows
with the current bundles reproduced its `semantic` column (cosine >= 0.9997) and its `image` column
with the **news** variant (cosine >= 0.9999). The release has no test rows; the 1,235 test vectors
were built with the same three producers and the text vectors joined by `text_asset_id`.

```text
python -m fnd.train_fusion --features fusion-training-features.npz --test-features test_features.npz --out outputs/fusion_v3
```

Both files use the arrays `sample_ids`, `split`, `scenario` (original 1..5), `semantic`, `image`,
`text`. Recipe (as `fnd.train_team_fusion`): AdamW 1e-4, batch 64, class-balanced sampler,
cross-entropy plus 0.1 x image auxiliary loss, checkpoint selection on validation macro-F1,
patience 10. Test rows are scored once, by the selected checkpoint, after training.
Scenario-to-class mapping: S4 genuine, S1 out-of-context, S3 real text/fake image,
S2 fake or edited text/real image, S5 fake or edited text/fake image.

## Result (seed 42)

Best validation macro-F1 0.8428 at epoch 8 of 18.

| Test class (247 each) | Recall | F1 |
|---|---:|---:|
| Genuine | 0.8623 | 0.7802 |
| Out-of-context | 0.8097 | 0.8351 |
| Real text, fake image | 0.8623 | 0.8386 |
| Fake/edited text, real image | 0.8381 | 0.8734 |
| Fake/edited text, fake image | 0.8016 | 0.8553 |
| **Five-class accuracy / macro-F1** | **0.8348** | **0.8365** |

Collapsed to real vs fake, test accuracy is **0.9028** (macro-F1 0.8589), level with the semantic
model alone (0.9045), while the fusion also names the kind of manipulation. The largest confusions
are fake-text/fake-image predicted as real-text/fake-image (46), out-of-context predicted as genuine
(39) and fake text predicted as genuine (31).

## Limitations

- **Single seed (42).** Treat this as the current experiment result, not a final robustness
  estimate. Differences of about 0.01 between runs should not be read as meaningful without repeated
  seeds.
- The trained fusion checkpoint is not yet connected to runtime inference or the dashboard
  ([DASHBOARD.md](DASHBOARD.md)).
- `fusion-training-features.npz` and the test feature file are not in this repository.
