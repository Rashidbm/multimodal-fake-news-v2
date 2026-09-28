# Fusion, match/mismatch comparison and dashboard

This branch adds three things on top of the semantic, image and text branches:

1. The result of the match/mismatch comparison: the released CLIP-L semantic model against the
   Qwen3-VL-Embedding-8B arm of `fnd.fit_semantic_arm`.
2. `fnd/train_fusion.py`: training and evaluation of the V3 fusion module (`fnd.models.pipeline_v3`)
   on the three branches' features.
3. `dashboard/`: a web backend that runs the branches live behind the semester-1 interface.

All results below are on the team dataset `fnd_team_dataset_2026-09-14` (21,404 image-caption
pairs: 17,794 train, 2,375 validation, 1,235 test with 247 per scenario). The test split is the
five-scenario set behind the semantic branch's recorded 90.45%.

## 1. Match/mismatch: CLIP-L vs Qwen3-VL-Embedding

The Qwen arm keeps the semantic model's FND-CLIP and BLIP blocks and replaces only the CLIP-L block
with one joint Qwen3-VL-Embedding-8B vector of the image and caption (feature kind
`blip_v1_qwen_embedding`, 5,634 features). It was fitted with `fnd.fit_semantic_arm fit` on train,
selected on validation only (candidate `extra1_ooc0.125_c0.01`, threshold 0.2411), and scored once
on test. The CLIP-L numbers are the released semantic model's recorded test predictions
(`reports/semantic`), scored with the same `fnd.evaluate.score_rows`.

| Test metric (1,235 rows) | CLIP-L (released) | Qwen3-VL-Embedding |
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

**CLIP-L is kept.** The largest loss is on out-of-context vs genuine, the match/mismatch case
itself. A likely reason is the feature design rather than the encoder: CLIP-L supplies separate
image and text embeddings plus their product, difference and cosine, while the Qwen arm supplies one
joint embedding, so the classifier never sees an explicit image-text comparison. Separate Qwen image
and caption embeddings with the same comparison features were not tested.

Scope of the comparison:

- This is "Case B" in [SEMANTIC_BRANCH.md](SEMANTIC_BRANCH.md): the original selection inputs
  (recall floors, extra-pair schedule) were not available, so the Qwen arm was selected without
  floors. The two models share test rows and metric code, not an identical selection procedure.
- No CLIP-L arm was refitted under the Case B rules.
- Qwen embeddings were extracted one pair at a time (batch size 1) in BF16 on an RTX 4090. The
  extractor's smoke check fails its strict norm tolerance (±1e-3) under BF16: norms were
  0.9962-1.0042, which the classifier's standardisation absorbs.

Running `fnd.cache_qwen_embedding` with transformers 5.x needs two workarounds, applied at runtime
without editing any file: the checkpoint's `scripts/qwen3_vl_embedding.py` imports
`check_model_inputs`, which transformers 5 removed (the script never uses it), and
`load_embedder` must register the loaded script in `sys.modules`, which transformers 5 looks up
when the model class is instantiated.

## 2. Full prediction system (fusion)

`fnd.train_fusion` trains `V3FusionModule` (tokenized pairwise cross-attention, commit 607d9f5) on
cached vectors. Only the fusion module trains; the branches are frozen.

| Input | Shape | Producer |
|---|---|---|
| `semantic` | `[N, 768]` | `models/semantic` bundle (CLIP-L semantic model) |
| `image` | `[N, 768]` | `models/image/news` bundle |
| `text` | `[N, 4096]` | Qwen3.5-9B layer 30, masked mean, max length 64 ([TEXT_FLUOROSCOPY.md](TEXT_FLUOROSCOPY.md)) |

Train and validation vectors are the release asset `fusion-training-features.npz` (20,162 rows; 7
train/validation rows whose image cache was unavailable are excluded). Recomputing 16 of its rows
with the current bundles reproduced its `semantic` column (cosine ≥ 0.9997) and its `image` column
with the **news** variant (cosine ≥ 0.9999). The release has no test rows; the 1,235 test vectors
were built with the same three producers and the text vectors joined by `text_asset_id`.

```text
python -m fnd.train_fusion --features fusion-training-features.npz --test-features test_features.npz --out outputs/fusion_v3
```

Both files use the arrays `sample_ids`, `split`, `scenario` (original 1..5), `semantic`, `image`,
`text`. The recipe follows `fnd.train_team_fusion`: AdamW 1e-4, batch 64, class-balanced sampler,
cross-entropy plus 0.1 × image auxiliary loss, checkpoint selection on validation macro-F1,
patience 10. Test rows are scored once, by the selected checkpoint, after training.
Scenario-to-class mapping: S4 genuine, S1 out-of-context, S3 real text/fake image,
S2 fake or edited text/real image, S5 fake or edited text/fake image.

Result (seed 42; best validation macro-F1 0.8428 at epoch 8 of 18):

| Test class (247 each) | Recall | F1 |
|---|---:|---:|
| Genuine | 0.8623 | 0.7802 |
| Out-of-context | 0.8097 | 0.8351 |
| Real text, fake image | 0.8623 | 0.8386 |
| Fake/edited text, real image | 0.8381 | 0.8734 |
| Fake/edited text, fake image | 0.8016 | 0.8553 |
| **Five-class accuracy / macro-F1** | **0.8348** | **0.8365** |

Collapsed to real vs fake, test accuracy is 0.9028 (macro-F1 0.8589), level with the semantic model
alone (0.9045), while the fusion also names the kind of manipulation. The largest confusions are
fake-text/fake-image predicted as real-text/fake-image (46), out-of-context predicted as genuine
(39) and fake text predicted as genuine (31). This is a single seed; differences of about 0.01
between runs should not be read as meaningful without repeated seeds.

## 3. Dashboard

```text
python -m dashboard.server
python -m dashboard.server --fusion package.module:factory
```

Open http://127.0.0.1:8000. Each request runs the semantic branch and the image branch (news
bundle by default) on the first GPU and Qwen3.5-9B on the second, using the text settings above
(pass `--text-model` with the local model directory). The page is the semester-1 interface with
null-safe rendering only; its layout and labels are unchanged.

Without `--fusion` the verdict reads "Final prediction pending" and the class probabilities stay
empty; the module rows show the image branch score, the BLIP mismatch score and, as overall risk,
the semantic branch's pair fake score. The text rows stay empty because no model scores AI
authorship or text patterns. A fusion factory is described in the `dashboard/server.py` docstring;
classes are matched by name. The trained fusion checkpoint above is not yet connected.

The server runs offline and never downloads models: set `HF_HOME` to the folder that holds the
pretrained caches before starting it.
