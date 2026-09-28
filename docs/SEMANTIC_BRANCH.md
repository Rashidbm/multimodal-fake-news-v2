# Semantic branch

## What is implemented

This is the selected FND-CLIP + BLIP + CLIP-L/14 semantic extension, not an exact
reproduction of the FND-CLIP authors' implementation. It accepts image pixels and
caption tokens. Ground-truth labels, scenario numbers and source names are not
model inputs.

The frozen encoders produce:

- FND-CLIP: 768 semantic features and one classification logit. Its internal
  encoders are ResNet-50, BERT and CLIP-B/32.
- BLIP (`Salesforce/blip-itm-base-coco`): 768 joint image/text features and one
  image-text mismatch logit.
- Additional CLIP-L/14: image and text embeddings (768 each), their elementwise
  product (768), absolute difference (768), and cosine similarity (1).

These form 4,611 raw features. A standardized, regularized logistic classifier
predicts three internal categories: genuine, OOC, and other manipulated pairs.
The reported binary fake score is the sum of the two non-genuine probabilities.
The fixed selected threshold is **0.3552747289549618**.

For downstream integration, a training-fitted projection maps the standardized
features to **768 dimensions**. It preserves the classifier directions and adds
principal-component directions. Fusion consumes this vector; the binary decision
and fake score are not separate fusion inputs. The vector does retain learned
classification information, including information derived from the branch logits.
Changing this representation requires new downstream caches and training.

## Setup and trained files

Use a dedicated environment. The verified Mac environment was Python 3.13.11,
PyTorch 2.11.0, torchvision 0.26.0 and transformers 5.2.0. For NVIDIA, install the
matching CUDA-enabled PyTorch/torchvision build before the remaining dependencies.
This release is CPU-tested on macOS; Windows/CUDA execution is not newly verified
by this branch.

From the repository root:

```text
python -m pip install -r requirements-semantic.txt
```

The repository includes `models/semantic/bundle.json` with relative artifact paths.
Place these three **separately transferred** files in that same directory:

| File | Approximate size | Purpose |
|---|---:|---|
| `epoch_5.pt` | 1,086.67 MiB | Trained FND-CLIP checkpoint |
| `extra1_ooc0.25_c0.001.joblib` | 0.21 MiB | Selected classifier and scaler |
| `semantic_projection.npz` | 27.04 MiB | Selected 768-dimensional projection |

Exact hashes are in [ARTIFACTS.json](../models/semantic/ARTIFACTS.json); inference
checks them. These files exist in the previously assembled full Windows demo
under `outputs/semantic_resolution/selected/`. The small fusion-only package does
not contain them. Alternatively, download and extract `semantic-weights.zip`
from the [model release](https://github.com/Rashidbm/multimodal-fake-news-v2/releases/tag/semantic-image-models-2026-09-26)
into the repository root; it supplies these files under `models/semantic/`.

BLIP and CLIP-L download from their pinned Hugging Face revisions on first use.
The FND checkpoint records its own configuration; its pretrained encoder/tokenizer
dependencies may also need downloading or an existing cache. Do not replace
missing trained files with random or freshly initialized models.

## Run one prediction

```text
python -m fnd.predict_semantic_fusion --bundle models/semantic/bundle.json --image /path/to/image.jpg --text "The actual caption" --device cuda
```

Use `--device cpu` without an NVIDIA GPU or `--device mps` on the Mac. On Windows,
use a quoted Windows image path. Output contains the binary prediction, fake
score, fixed threshold, and BLIP mismatch diagnostic. These scores are model
outputs, not externally calibrated factual certainty.

## Return the vector for integration

```python
from PIL import Image
from fnd.predict_semantic_fusion import SemanticFusionPredictor

model = SemanticFusionPredictor("models/semantic/bundle.json", device="cuda")
with Image.open("example.jpg") as image:
    result = model.predict([image], ["The actual caption"], return_semantic=True)
semantic_vector = result["semantic"]  # CPU float32 tensor, shape [1, 768]
```

The caller moves that tensor to its fusion model's device as needed. No image-
forensics or text-forensics model is required to run this semantic branch.

To export a table of vectors:

```text
python -m fnd.cache_semantic_bundle --bundle models/semantic/bundle.json --csv /path/to/samples.csv --out artifacts/semantic_features.pt --device cuda --batch-size 4
```

Required CSV columns: `sample_id`, `image_path`, `text`. Optional `image_sha1` is
checked against image bytes. IDs must be unique. Image paths must be absolute or
relative to the command's working directory, not automatically relative to the
CSV. The command exports every supplied row, so provide the intended split(s).
Join the resulting `sample_ids` and `[N,768]` features by ID; never assume another
branch has identical row order. Output also records bundle/CSV hashes.

## Training history and reproducibility boundaries

FND-CLIP was trained before this extension; BLIP and CLIP-L remained pretrained.
For the final update, all encoders stayed fixed and the classifier was refitted on
17,794 pairs: 8,897 genuine, 5,435 OOC, and 1,154 each in S2/S3/S5. Training loss
mass was 50% genuine, 25% OOC, and 8.33% each for the remaining scenarios.

Reusable entry points:

| Module | Role |
|---|---|
| `fnd.train` | Base FND-CLIP training |
| `fnd.cache_blip`, `fnd.cache_v1_features`, `fnd.cache_clip_large` | Cache encoder features |
| `fnd.fit_semantic_fusion`, `fnd.fit_clip_large_fusion` | Fit alternative semantic classifiers |
| `fnd.semantic_resolution`, `fnd.semantic_more_data` | Recorded development experiments and final data expansion |
| `fnd.export_semantic_projection` | Training-fitted classifier-preserving projection |
| `fnd.evaluate_semantic_resolution` | Locked comparison against the earlier bundle |

Use `--help` on argument-based entry points. The final experiment modules and
historical verification scripts require the original dataset manifests, caches,
previous bundle, and selection records in their recorded project-relative layout.
They are not one-command retraining from a Git clone. This release contains code,
the inference bundle template and evaluation evidence, not the original training
data/caches or all previous experimental checkpoints. Do not use the published
test predictions or labels to tune a new model.

## Shared labels

The corrected folder mapping is [FOLDER_TO_GROUP.json](../configs/dataset/FOLDER_TO_GROUP.json).
The full modality/scenario contract is [LABEL_MAPPING.json](../configs/dataset/LABEL_MAPPING.json).
Keep the team's existing split and sample IDs.

| Scenario | Meaning | Binary pair label |
|---|---|---:|
| S1 | Authentic text/image paired out of context | 1 |
| S2 | Fake/edited text, authentic image | 1 |
| S3 | Authentic text, fake image | 1 |
| S4 | Genuine image/text pair | 0 |
| S5 | Fake/edited text and fake image | 1 |

Do not use the pair label as an image-only label or an AI-authorship label.

## Recorded five-scenario evaluation

The selected standalone binary classifier achieved **90.45% accuracy**
(1,117 / 1,235), 85.83% genuine recall, 91.60% fake recall, and 86.05% macro-F1.
There are 247 examples per scenario: 988 fake and 247 genuine pairs. Always
predicting fake scores 80% on this composition. This set had been evaluated in
earlier experiments and was reused as a regression check; this is not an
independent new benchmark or a five-class multimodal fusion result.

Actual/predicted counts:

| Actual | Predicted genuine | Predicted fake |
|---|---:|---:|
| Genuine | 212 | 35 |
| Fake | 83 | 905 |

[Metrics and individual predictions](../reports/semantic) are included. Existing
evaluation samples must remain excluded from training and model selection.

## Checks

```text
python -m pytest fnd/tests -q
```

Tests use synthetic inputs and do not download model weights. They verify model
shapes, label/input isolation, probability handling, threshold selection, feature
projection equivalence and dataset logic. Real-checkpoint inference additionally
requires the three model files and pretrained caches described above.

## Experiment: Qwen3-VL-Embedding-8B in place of CLIP-L

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
