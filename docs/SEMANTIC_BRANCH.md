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
