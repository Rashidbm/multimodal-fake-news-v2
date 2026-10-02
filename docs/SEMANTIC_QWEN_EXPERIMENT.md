# Semantic branch experiment: Qwen3-VL-Embedding vs CLIP-L

**Decision: CLIP-L remains the selected and default semantic model.** Qwen3-VL-Embedding-8B was
evaluated as an alternative third feature block and scored lower overall, so it was not adopted.
Qwen3-VL-Embedding is *not* required for training, inference or the dashboard on `main`.

This document preserves the experiment for reproducibility. The implementation is **not on `main`**;
it lives in the historical branches `experiment/semantic-qwen3vl-embedding` (commit `4e63fec`) and
`Qwen3VL_FUSION` (commit `78724a8`). Compact final result files are in
[`reports/semantic_qwen/`](../reports/semantic_qwen/).

## Why it was tested

The semantic arm uses CLIP-L for image and caption embeddings. The question was whether a newer
joint vision-language embedding model, Qwen3-VL-Embedding-8B, would give a stronger image-caption
match/mismatch signal (especially out-of-context pairs) while keeping the downstream `[768]`
`v_semantic` interface unchanged.

## Setup

All results are on the team dataset `fnd_team_dataset_2026-09-14` (21,404 image-caption pairs:
17,794 train, 2,375 validation, 1,235 test with 247 per scenario). The test split is the
five-scenario set behind the semantic branch's recorded 90.45%.

The Qwen arm keeps the semantic model's FND-CLIP and BLIP blocks and replaces only the CLIP-L block
with one joint Qwen3-VL-Embedding-8B vector of the image and caption (feature kind
`blip_v1_qwen_embedding`, 5,634 features, versus `blip_v1_clip_large`, 4,611). Qwen is frozen and runs
through the official `Qwen3VLEmbedder` shipped with the checkpoint. The Logistic Regression grid,
`protected_point` selection and `fit_projection` are unchanged. The Qwen arm was fitted on train,
selected on validation only (candidate `extra1_ooc0.125_c0.01`, threshold 0.2411) and scored once on
test. The CLIP-L numbers are the released semantic model's recorded test predictions
(`reports/semantic`), scored with the same `fnd.evaluate.score_rows`.

## Results (test, 1,235 rows)

| Metric | CLIP-L (released) | Qwen3-VL-Embedding |
|---|---:|---:|
| Accuracy | **0.9045** | 0.8818 |
| Balanced accuracy | **0.8871** | 0.8456 |
| Macro-F1 | **0.8605** | 0.8256 |
| Recall fake / genuine | **0.9160 / 0.8583** | 0.9059 / 0.7854 |
| AUC / AUC OOC vs genuine | **0.9638 / 0.9389** | 0.9430 / 0.8702 |
| Recall S1 out-of-context | **0.8623** | 0.7814 |
| Recall S2 fake text | 0.8543 | **0.9069** |
| Recall S3 fake image | **0.9474** | 0.9433 |
| Recall S4 genuine | **0.8583** | 0.7854 |
| Recall S5 fake both | **1.0000** | 0.9919 |

The raw numbers are in `reports/semantic_qwen/comparison.json` (both models) and
`evaluation.json` (Qwen arm). They match the table above.

## Conclusion

CLIP-L is better overall and on every metric except one scenario: Qwen3-VL-Embedding was better on
the **fake-text scenario (S2: 0.9069 vs 0.8543)**. The largest loss for Qwen is on out-of-context vs
genuine, the match/mismatch case itself (S1 0.7814 vs 0.8623; OOC-vs-genuine AUC 0.8702 vs 0.9389).
CLIP-L is therefore kept as the default semantic model.

## Caveats (important)

This experiment does **not** show that Qwen3-VL-Embedding is inherently worse than CLIP-L.

- **Feature design.** CLIP-L supplies separate image and text embeddings plus their product,
  difference and cosine; the Qwen arm supplies a single joint embedding, so the classifier never sees
  an explicit image-text comparison. Separate Qwen image and caption embeddings with the same comparison
  features were **not tested**.
- **Selection procedure (Case B).** The original selection inputs (recall floors, extra-pair
  schedule) were not available, so the Qwen arm was selected without floors. The two models share
  test rows and metric code, not an identical selection procedure. No CLIP-L arm was refitted under
  the Case B rules.
- **Precision.** Embeddings were extracted one pair at a time (batch size 1) in BF16 on an RTX 4090.
  The extractor's smoke check fails its strict norm tolerance (±1e-3) under BF16 (norms
  0.9962-1.0042), which the classifier's standardisation absorbs.
- **transformers 5.x.** Running the extractor needs two runtime workarounds: the checkpoint's
  `scripts/qwen3_vl_embedding.py` imports `check_model_inputs`, which transformers 5 removed (the
  script never uses it), and `load_embedder` must register the loaded script in `sys.modules`.
- **Single run.** One fit and one test scoring; no repeated seeds.

## Reproducing (from the experiment branches)

Check out the experiment branch first; none of these modules exist on `main`:

```text
git checkout Qwen3VL_FUSION      # or experiment/semantic-qwen3vl-embedding
```

The text below is the experiment section of `docs/SEMANTIC_BRANCH.md` as it was on that branch.


Branch `experiment/semantic-qwen3vl-embedding`. Only the third feature block changes:

| | Baseline | Experiment |
|---|---|---|
| Third block | CLIP-L `[img, txt, img*txt, abs(img-txt), cos]`, 3,073 | one joint image+caption embedding, D (4,096 for the 8B model) |
| Feature kind | `blip_v1_clip_large`, 4,611 | `blip_v1_qwen_embedding`, 1,538 + D |

FND-CLIP, BLIP, labels, the Logistic Regression grid, `protected_point` selection and
`fit_projection` are unchanged. Qwen is frozen and runs through the official
`Qwen3VLEmbedder` shipped in the checkpoint (`<model>/scripts/qwen3_vl_embedding.py`):
default instruction, dynamic-resolution images, built-in pooling and L2 normalization.
The fitted scaler, classifier, threshold and projection are new for the Qwen arm;
`models/semantic/` is never read for fitted artifacts and never written.

Setup: `pip install -r requirements-semantic.txt` (adds `qwen-vl-utils`) and point
`QWEN_MODEL_PATH` at the local model directory (`export QWEN_MODEL_PATH=...`, or
`$env:QWEN_MODEL_PATH = "..."` in PowerShell). Loading is offline (`local_files_only`).
The GPU is chosen with `CUDA_VISIBLE_DEVICES`; one BF16 copy per RTX 4090.

```text
# 1. Smoke: images, local load, CUDA, dimension, norms, batch invariance, cache write
python -m fnd.cache_qwen_embedding --csv TRAIN_DEV.csv --out features/qwen/smoke.pt --smoke 4 --batch-size 4

# 2. Extract once, one process per GPU, then merge in CSV order
CUDA_VISIBLE_DEVICES=0 python -m fnd.cache_qwen_embedding --csv TRAIN_DEV.csv --out features/qwen/train_dev.0.pt --shard 0/2 --batch-size 4
CUDA_VISIBLE_DEVICES=1 python -m fnd.cache_qwen_embedding --csv TRAIN_DEV.csv --out features/qwen/train_dev.1.pt --shard 1/2 --batch-size 4
python -m fnd.cache_qwen_embedding --csv TRAIN_DEV.csv --out features/qwen/train_dev.pt --merge features/qwen/train_dev.0.pt features/qwen/train_dev.1.pt
# Repeat for each evaluation CSV with --splits listing every split value in that CSV.

# 3. Fit each arm on identical rows and FND/BLIP caches (validation-only selection)
python -m fnd.fit_semantic_arm fit --third clip_large     --csv TRAIN_DEV.csv --blip BLIP.pt --v1 V1.pt --third-cache CLIP.pt              --out outputs/semantic_arms/clip_large
python -m fnd.fit_semantic_arm fit --third qwen_embedding --csv TRAIN_DEV.csv --blip BLIP.pt --v1 V1.pt --third-cache features/qwen/train_dev.pt --out outputs/semantic_arms/qwen_embedding

# 4. Evaluate both locked arms once
python -m fnd.fit_semantic_arm evaluate --eval-csv EVAL.csv --blip EVAL_BLIP.pt --v1 EVAL_V1.pt --out outputs/semantic_arms/evaluation \
    --arm clip_large outputs/semantic_arms/clip_large EVAL_CLIP.pt --arm qwen_embedding outputs/semantic_arms/qwen_embedding features/qwen/eval.pt

# 5. Export the Qwen bundle (768-d projection) and v_semantic for Fusion
python -m fnd.fit_semantic_arm export --fit-dir outputs/semantic_arms/qwen_embedding --csv TRAIN_DEV.csv --blip BLIP.pt --v1 V1.pt \
    --third-cache features/qwen/train_dev.pt --out outputs/semantic_arms/qwen_bundle
python -m fnd.cache_semantic_bundle --bundle outputs/semantic_arms/qwen_bundle/bundle.json --csv FUSION.csv --out artifacts/semantic_qwen_features.pt --device cuda
```

**Case A, original artifacts present:** use `data/processed/semantic_resolution_train_dev.csv`,
the `outputs/semantic_resolution/expanded_{blip,v1,clip}.pt` caches, and add to both `fit` calls
`--extra-csv data/processed/semantic_resolution_extra_train.csv --reference-diagnostic
outputs/semantic_resolution/incumbent_diagnostic.json --reference-manifest
outputs/semantic_resolution/data_volume/manifest.json`. Evaluate `data/processed/recall_evaluation.csv`
with the `outputs/recall_improvement/eval_{blip,old_v1,clip_large}.pt` caches, `--evaluation-group
v1_five_scenarios` and `--expect-predictions reports/semantic/five_scenario_predictions.csv`: the
CLIP arm must reproduce the recorded predictions (error < 1e-10) before the Qwen result is accepted.
Run the evaluation again on `data/processed/semantic_resolution_confirmation.csv` with the
`outputs/semantic_resolution/confirmation_{blip,v1,clip}.pt` caches for the paired genuine/OOC comparison. If image paths in the CSVs point elsewhere, add
`--path-prefix OLD=NEW` to the Qwen extraction; the CSV and its hash stay unchanged.

**Case B, artifacts missing:** freeze one CSV with `sample_id, split, text, image_path, scenario,
label_binary, evaluation_group, caption_id`, extract FND/BLIP/CLIP-L/Qwen caches from it, and fit both
arms without the reference options. The historical 90.45% is then not a direct benchmark.

Outputs: `fit` writes every candidate, `results.json` and `selection.json`; `evaluate` writes
`evaluation.json` and per-arm predictions; `export` writes `bundle.json` (`feature_kind`,
`qwen_embedding` settings and hashes), the classifier, `semantic_projection.npz` and `ARTIFACTS.json`.
The bundle is rejected if its classifier kind and encoders disagree. `cache_semantic_bundle`
fails unless the exported features are `[N, 768]`.
