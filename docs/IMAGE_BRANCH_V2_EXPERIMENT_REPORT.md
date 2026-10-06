# Image Branch v2: experiment report

Complete technical record of the redesigned MultiGuard Image Branch: the data, the old and new pipelines, the
fair comparison, the results and the production recommendation. It is written to be read without the conversation
history. Branch `experiment/image-branch-v2`. All numbers come from `python -m fnd.imagev2.aggregate` over the run
directories listed in [Artifacts and Paths](#artifacts-and-paths).

> **DINOv3 note.** DINOv3 ViT-L/16 was the originally intended candidate. Access to `facebook/dinov3-vitl16-pretrain-lvd1689m`
> stayed gated (the access request was "awaiting review from the repo authors"), so **DINOv2-L (`facebook/dinov2-large`)
> was used as an approved fallback**. Every result labelled B1 below is a **DINOv2-L result, not a DINOv3 result**. DINOv3 is
> still untested.

## 1. Executive summary

- **Why redesign.** The deployed branch (CLIP + RGB ResNet + DCT ResNet, fixed PCA projection) scored 97% on our news data but
  76.8% on GenImage and 50.7% on DGM4. It leans heavily on CLIP (duplicating the Semantic Branch), its two CNNs were
  trained on GenImage (external data), its 768 vector is a fixed unsupervised projection, and our data has strong
  resolution/format/source shortcuts.
- **What was compared** (identical data, split, preprocessing, sampler, bottleneck, heads, losses, seeds; only the frozen
  feature extractor differs):
  **A0** = CLIP-L only, **A1** = the current CLIP + RGB + DCT encoders, **B1** = DINOv2-L (frozen, 4 layers × CLS + mean patch
  = 8,192 features).
- **Main result.** On the primary metrics A1 and B1 are **statistically indistinguishable**: test balanced accuracy
  0.971 vs 0.966 (Δ +0.005, 95% CI −0.009…+0.018); worst-domain balanced accuracy 0.942 vs 0.902 (Δ +0.033, CI −0.013…+0.079).
  A1 is numerically ahead on the weakest domain (Fakeddit photo edits: fake recall 0.90 vs 0.81). B1 is ahead on AUROC
  (0.993 vs 0.987) and on the held-out Midjourney test (0.986 vs 0.978), and is behind on the held-out COCO test
  (balanced accuracy 0.74–0.76 vs 0.78–0.80, though its AUROC is higher).
- **Fairness.** A1's RGB/DCT encoders carry GenImage and external-forensic knowledge; B1 is a generic foundation model.
  A1 is therefore **not eligible** for a branch that must use only our team data, so the production candidate is chosen among
  the eligible arms (A0, B1).
- **Winner.** Unconstrained comparison: **inconclusive (A1 ≈ B1)**. Under the project constraints: **B1 (DINOv2-L, frozen,
  shared bottleneck, binary loss)**. Its main weakness is manipulation (Photoshop-style edits): recall about 0.80.
- **Auxiliary 3-class loss.** Did not help: test balanced accuracy changed by −0.001 to −0.005 for every arm, which is within seed noise (no paired
  test across λ was run). Recommended λ = 0.
- **Recommendation.** Use B1 λ=0 (checkpoint `B1_main_lam0_s47`) as the Image Branch, threshold 0.5, and later test DINOv3
  with the same harness. Fusion must be re-extracted and retrained on the new `v_imgfor`; the dashboard is not switched yet.

## 2. Role of the Image Branch in MultiGuard

```
image ─► Image Branch ─► v_imgfor [768] ─┐
caption ─► Semantic Branch ─► v_semantic [768] ─┼─► fusion (V3FusionModule) ─► 5 classes
caption ─► Text Branch (Qwen) ─► v_textfor [4096] ─┘
```

Fusion predicts five classes: genuine, out-of-context, real text + fake image, fake/edited text + real image, fake/edited
text + fake image. The Image Branch answers one question about the **image itself**: is it authentic or fake
(AI-generated or manipulated)? It never sees the caption. Fusion also trains an auxiliary real/fake image head on
`v_imgfor.detach()` (`fnd/models/pipeline_v3.py`).

## 3. Input/output contract

- **Input:** a PIL image (pixels only; file names, metadata, captions and labels are not inputs).
- **Output:** `features` = `v_imgfor` [B, 768]; `probability` = P(fake) [B]; optional `type_probabilities` (REAL / AI_GENERATED /
  MANIPULATED, diagnostic, only when the aux loss was used). Decision: `probability >= threshold` (0.5).
- Target: 0 = authentic image, 1 = AI-generated or manipulated image.

## 4. Image truth definitions

| Truth | Binary | Joint label | Meaning |
|---|---|---:|---|
| REAL | 0 | 0 | Authentic image by dataset construction, including images under fake text or out-of-context captions |
| AI_GENERATED | 1 | 1 | Fully generated image |
| MANIPULATED | 1 | 2 | Real photo edited (Photoshop-style composites) |
| EXCLUDE | | | Unresolved; never used |

Relation to the five project scenarios: S1 (OOC), S2 (fake text) and S4 (genuine) have REAL images; S3 (fake image) and S5
(fake text + fake image) have AI_GENERATED or MANIPULATED images. The old `image_fake` column is exactly "binary fake" but
mixes AI and manipulation; scenario and `label_binary` must never be used as image truth.

## 5. Team dataset provenance

`fnd_team_dataset_2026-09-14` (MMFakeBench plus NewsCLIPpings material): `master.csv` 21,404 pair rows, 21,169 unique
images (21,169 files under `images\`), split train 17,794 / val 2,375 / test 1,235 rows. Image bytes are preserved
(no resize or recompression). Files date from 14 September 2026 and were re-checked unchanged (390 Fakeddit edits) before the
B1 run.

## 6. Why only original project train is used

The project val and test splits are reserved for the multimodal fusion evaluation (the semantic team already used the test
rows). The Image Branch therefore trains, selects and tests **only inside the original project train split**, by carving a
new internal 70/15/15 split from it. No GenImage, DGM4 or other external data is used for training anywhere in the new
comparison. Protected external-evaluation fingerprints (6,000 rows) are checked and none is included.

## 7. Mapping logic

`fnd/data/build_image_branch_v2.py` assigns truth from explicit rules per (source, subcategory) plus pixel-size checks, never
from `scenario`/`label_binary`/`image_fake` alone. Full table: `docs/IMAGE_DATA_MAPPING.md`. Summary:

| Truth | Source groups | Evidence |
|---|---|---|
| REAL | NewsCLIPpings (matched and OOC), MMFB genuine sources, OOC, fake-text groups, `DGM4_text_edit_senti` | image untouched; only captions are mismatched or edited |
| AI_GENERATED | `fever_AI`, `antifact_image_generation` (1024×1024), `llm_*`, `gossipcop_midjourney`, `coco_image_edit`, `coco_text_edit` (512×512) | 1024² or 512² generated images, visual inspection; 15 `fever_AI` PNGs carry `sd_xl_base_1.0` metadata |
| MANIPULATED | `Fakeddit_photo_edit` | Photoshop-style composites at native sizes |
| EXCLUDE | 11 `antifact` images that are not 1024×1024 | likely AI-edited real photos |

Misleading labels: `fake_cls=mismatch` covers real OOC images and synthetic COCO images; `coco_image_edit` is not an edit of a
real photo; `source_split` is not our split. Generator names are style inferences except Midjourney V6 and the 15 SDXL images.

## 8. Exact class and source counts (original train, unique images)

| Truth | Images |
|---|---:|
| REAL | 15,317 |
| AI_GENERATED | 1,907 |
| MANIPULATED | 390 |
| EXCLUDE | 11 |
| Total (17,794 rows, 169 duplicate rows collapsed) | 17,625 |

| Truth | Sub-source (domain) | Images |
|---|---|---:|
| REAL | NewsCLIPpings JPEG (news) | 10,422 |
| REAL | MMFB VisualNews PNG (news) | 3,989 |
| REAL | MMFB COCO (coco) | 450 |
| REAL | MMFB Fakeddit (fakeddit) | 456 |
| AI_GENERATED | DALL-E-3-style `fever_AI`, 15 verified SDXL (news) | 514 |
| AI_GENERATED | Midjourney-style `antifact` (news) | 375 |
| AI_GENERATED | `llm_*` (news) | 258 |
| AI_GENERATED | Midjourney V6 `gossipcop_midjourney` (news) | 136 |
| AI_GENERATED | SD-512 COCO counterfactuals (coco) | 624 |
| MANIPULATED | `Fakeddit_photo_edit` (fakeddit) | 390 |

## 9. Internal train/val/test split

70/15/15 by group, stratified by (domain, truth, sub-source), seed `image_branch_v2_20261006`; one split per image across all tasks.

| Truth | train | val | test |
|---|---:|---:|---:|
| REAL | 10,726 | 2,296 | 2,295 |
| AI_GENERATED | 1,337 | 286 | 284 |
| MANIPULATED | 273 | 59 | 58 |

Val and test files contain the balanced **core** rows only (REAL 319 / 316, AI 260 / 258, MANIPULATED 59 / 58; domain mix
news 191+191, COCO 67+67, Fakeddit 58+58 positives/negatives in the test set). The train file also holds reserve real news
images, which the sampler draws on.

## 10. Leakage prevention

Groups join images that are identical, near-duplicates (dHash distance ≤ 4) or share a caption of at least four words:
13,295 groups, **0 cross-split groups**; 0 images appear in two splits; 0 overlaps with the protected fingerprints; every image
is original-train. `validate()` asserts all of this on every build and the build is deterministic. Not covered: near-duplicates
beyond a 64-bit dHash.

## 11. Shortcut audit

A gradient-boosted classifier on file metadata alone separated classes with AUC about 1.0 (AI vs real) and 0.99 (manipulation),
driven by size and format: real MMFB images are mostly PNG at 256 px height or 320 px width, AI images are 1024² or 512², Fakeddit
edits keep native resolution, MMFB real images are always RGBA, 15 SDXL PNGs carry generation metadata. After conversion to RGB
and a 224×224 resize, cheap pixel statistics still reach AUC 0.61–0.76. Mitigations used: one canonical image for every class,
alpha dropped, metadata ignored, random JPEG/resize augmentation, a sampler that balances classes inside every domain and
alternates real-news JPEG/PNG sources. Residual leakage is measured in section 36.

## 12. The old/current pipeline

```
Raw image
 ├─ CLIP-L/14 image tower (shorter side 224 bicubic + center crop; L2-normalised) ── 768   frozen
 ├─ RGB ResNet-50 (squash 224×224, ImageNet norm)  rgb_robust.pt ───────────────── 2,048  frozen
 └─ DCT ResNet-50 (Y, 8×8 and 16×16 block DCT, log, z-score)  dct_guide.pt ──────── 2,048  frozen
        concat 4,864 ─► logistic head ─► P(fake)
                    └─► fixed projection (center, 768×4,864 orthonormal: classifier direction + 767 PCA) ─► v_imgfor [768]
```

The input is pixels only; there is no text, semantic score or semantic flag in the Image Branch
(`fnd/forensic/inference.py`). The CLIP text tower is loaded but never called. The Semantic Branch loads the same CLIP-L checkpoint
(`openai/clip-vit-large-patch14`, revision `32bd642`) and uses both towers, so the CLIP computation is duplicated at runtime and the
image embedding overlaps with `v_semantic`. Known weaknesses: about 99.8% of the `news` head's raw weight sits on the CLIP block,
the projection is fixed and unsupervised, and the old head's results were 97.4% (project news test), 76.8% (GenImage) and 50.7% (DGM4).

## 13. Current pipeline training history

| Component | Initialisation | Training |
|---|---|---|
| CLIP-L/14 | OpenAI image-text pretraining | none |
| `rgb_robust` | upstream author's frequency-mask ResNet-50 checkpoint (external forensic training) | fine-tuned by us on **GenImage** (32,000 train, 4,000 val; conv1/bn1/layer1/layer2 frozen; JPEG augmentation; default recipe AdamW lr 1e-4; best epoch 14; the exact command-line flags were not recoverable) |
| `dct_guide` | ImageNet ResNet-50, new 1-channel `conv1` | trained on **GenImage**; z-score constants fit on GenImage pixels; best epoch 5 |
| `news` head + projection | none | team data: logistic regression and PCA on the 17,620 project-train news images |

## 14. GenImage involvement

GenImage (8 generators, 2,000/250/500 images per class per generator) trained both CNNs of the old branch. It is **not** used in
any new run, and the new manifests contain no GenImage rows.

## 15. Why the old deployed head could not be used fairly

The deployed `news` head and projection were fitted on all 17,620 original-train news images. The Image Branch v2 val/test images are
subsets of those same images, so evaluating the deployed bundle on them would be contaminated. A fresh common head must be trained
on v2-train only, with the old encoders frozen.

## 16. Experiment A0

CLIP-L image tower only, frozen: raw feature 768 (the first 768 columns of A1; same canonical pixels, same weights). A0 is a diagnostic
baseline (UnivFD-style) and shows what the semantic-overlapping CLIP path alone provides.

## 17. Experiment A1

The deployed frozen encoders CLIP + `rgb_robust` + `dct_guide`, raw feature 4,864, extracted from the canonical image
(`fnd.imagev2.extract --arm A1`, which verifies equality with `ImageBranch.encode_images`). The deployed head/projection are not used.
**Fairness caveat:** `rgb_robust` and `dct_guide` carry GenImage and external-forensic knowledge, which A0 and B1 do not.

Geometry check (RGB/DCT were trained on squashed images, the canonical input is a center crop; both trained on clean views only):

| Arm (train views: clean only) | test bAcc | test AUROC | worst-domain bAcc |
|---|---|---|---|
| A1 (canonical crop for RGB/DCT) | 0.968 ± 0.007 [0.956, 0.978] | 0.990 ± 0.001 [0.981, 0.996] | 0.938 ± 0.016 [0.904, 0.960] |
| A1s (legacy squash for RGB/DCT) | 0.965 ± 0.007 [0.951, 0.977] | 0.992 ± 0.002 [0.987, 0.997] | 0.930 ± 0.013 [0.889, 0.953] |

The difference is not material (A1 − A1s: Δ bAcc +0.003, CI −0.006…+0.011), so A1 is not disadvantaged by the shared geometry.

## 18. Experiment B1 (DINOv2-L)

`facebook/dinov2-large`, revision `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`, Apache-2.0, `Dinov2Model`, 304.4M parameters, hidden size 1,024,
24 transformer blocks, 16 heads, patch 14, no register tokens. At 224 px there are 257 tokens (1 CLS + 256 patches). Frozen; fp32.
Taps: blocks **[12, 16, 20, 24]** (outputs of those blocks from `hidden_states`; the last is before the final LayerNorm). For each tap:
CLS token and mean of the patch tokens. Raw feature = 4 × (1,024 + 1,024) = **8,192** (verified empirically). Normalisation: ImageNet mean/std.

## 19. Why DINOv2 instead of DINOv3

DINOv3 ViT-L/16 was the original candidate. The Hugging Face repo is gated and our request was "awaiting a review from the repo authors"
(HTTP 403 `GatedRepoError`), so DINOv2-L was approved as a fallback. The harness is model-agnostic: the extraction path was validated against a tiny
random DINOv3 model, so a DINOv3 run needs only access plus `--model facebook/dinov3-vitl16-pretrain-lvd1689m` (section 47).

## 20. Shared preprocessing

Canonical image (`fnd/imagev2/canonical.py`): decode, RGB (alpha dropped, EXIF/PNG metadata ignored), view corruption at native
resolution, shorter side resized to 224 (bicubic), center crop 224×224. Views: `clean`; `jpeg75` (JPEG quality 75) and `blur1`
(Gaussian radius 1) for core val/test rows; `aug1`, `aug2` for train rows (seeded per image: with p=0.5 a random 50–100% downscale with a
random interpolation, then with p=0.5 JPEG quality 60–95). The same canonical pixels feed every encoder; only each model's normalisation
differs (A: CLIP/ImageNet constants and the DCT transform; B1: ImageNet).

## 21. Feature extraction

`fnd/imagev2/extract.py`: frozen encoders, fp32, batch 64, 10 data-loader workers, resumable. Per image id: clean for all 17,614 rows; aug1/aug2
for the 12,336 train rows; jpeg75/blur1 for the 1,270 core val/test rows. A1: 4,864 dims; B1: 8,192 dims. `validate_cache.py` checks row
counts, ids, split membership, shapes and NaN/Inf; clean views re-extract bit-identically on the samples tested (A1: 128 images, B1: 3 images; max difference 0.0).

## 22. Shared bottleneck and heads

```
raw features ─► standardise (train mean/std per dimension) ─► Linear(D→1024) ─► GELU ─► Dropout(0.2) ─► Linear(1024→768) ─► LayerNorm = v_imgfor [768]
                                                                                       ├─ binary head Linear(768→1)
                                                                                       └─ auxiliary head Linear(768→3)
```
D = 768 (A0), 4,864 (A1), 8,192 (B1). Both heads back-propagate into the bottleneck. Backbones are frozen.

## 23. Binary vs auxiliary supervision

Binary target: REAL→0, AI_GENERATED→1, MANIPULATED→1. Auxiliary target: 0/1/2. Run A: λ_aux = 0 (binary only) and λ_aux = 0.25.

## 24. Sampler

Natural train counts: REAL 10,726 / AI 1,337 / MANIPULATED 273. Per epoch 4,096 draws: REAL 50% / AI 35% / MANIPULATED 15%, balanced
1:1 inside every domain, real news alternating JPEG/PNG sources:

| Cell | Pool | Share | Draws/epoch | Repeats/epoch |
|---|---:|---:|---:|---:|
| news AI | 900 | 25.9% | 1,062 | 1.18 |
| news real (JPEG) | 7,297 | 13.0% | 531 | 0.07 |
| news real (PNG) | 2,794 | 13.0% | 531 | 0.19 |
| coco AI | 437 | 9.1% | 372 | 0.85 |
| coco real | 315 | 9.1% | 372 | 1.18 |
| fakeddit MANIPULATED | 273 | 15.0% | 614 | 2.25 |
| fakeddit real | 320 | 15.0% | 614 | 1.92 |

MANIPULATED stayed at 390 images, so the 15% policy was kept (about 2.25 repeats per epoch, each draw using a random one of 3 train views).

## 25. Loss

`BCEWithLogits(binary) + λ_aux · CrossEntropy(3-class, label_smoothing=0.1)`; no class weights (the sampler balances).

## 26. Seeds

13, 29, 47 for every configuration (three seeds each for λ = 0 and 0.25, for the main protocol and the two held-out protocols).

## 27. Model selection

AdamW (lr 5e-4, weight decay 1e-2, gradient clip 1.0), batch 128, 4,096 draws per epoch, at most 60 epochs, patience 8. Selection:
the best **worst-domain balanced accuracy on the clean validation set** at threshold 0.5, ties broken by lower validation BCE. The test
set is never used for selection. Validation-tuned thresholds were unstable across seeds (0.06–0.90 on a 638-image validation set; the tuned
threshold often lowered the worst-domain accuracy), so the production threshold stays at 0.5.

## 28. Hardware and GPU setup

2 × RTX 4090 (24 GB), 63.8 GB RAM, 32 logical CPUs, system Python 3.13.7, torch 2.11.0+cu128, transformers 5.2.0. Each extraction ran as a separate process with
`CUDA_VISIBLE_DEVICES` set (the process sees its GPU as `cuda:0`): GPU 0 for A1 (and the A heads), GPU 1 for A1s (and the geometry-check heads) and later for B1 and its heads. Observed VRAM: A1 extraction
about 5.0 GB, A1s about 3.9 GB, B1 peak allocated 3.5 GB (batch 64). Both GPUs were used independently; a single GPU would also have sufficed.

## 29. Extraction and training times

| Step | Time |
|---|---|
| A1 features (5 views, 44,826 images) | 365 s (clean view 136 s) |
| A1s features (clean, RGB+DCT) | 61 s |
| B1 features (5 views, 44,826 images) | 293 s (clean view 104 s) |
| Head training | 3–13 s per run (A: 42 runs incl. the geometry check, B1: 18 runs) |

Throughput on the clean view: about 130 images/s (A1, includes CPU DCT) and 169 images/s (B1). Inference needs one backbone for B1 (304M parameters) versus CLIP-L plus two ResNet-50s plus CPU DCT for A1.

## 30. Results: main protocol

Joint-forensic **test** (632 core images: 316 real, 258 AI, 58 manipulated), clean view, threshold 0.5. Mean ± std over 3 seeds; brackets are 95% group-bootstrap
intervals of the seed mean (1,000 resamples of manifest groups).

| Arm | λ | val worst-dom bAcc | test bAcc | test AUROC | fake recall | real recall | precision | F1 | worst-domain bAcc |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0 | 0.940 ± 0.011 | 0.960 ± 0.002 [0.946, 0.973] | 0.989 ± 0.002 [0.982, 0.995] | 0.963 ± 0.010 | 0.957 ± 0.014 | 0.957 ± 0.013 | 0.960 ± 0.001 | 0.922 ± 0.007 [0.875, 0.945] |
| A0 | 0.25 | 0.935 ± 0.004 | 0.959 ± 0.003 [0.944, 0.972] | 0.988 ± 0.002 [0.980, 0.995] | 0.948 ± 0.003 | 0.970 ± 0.005 | 0.970 ± 0.005 | 0.959 ± 0.003 | 0.914 ± 0.007 [0.863, 0.946] |
| A1 | 0 | 0.935 ± 0.008 | 0.971 ± 0.001 [0.958, 0.983] | 0.987 ± 0.001 [0.977, 0.995] | 0.976 ± 0.007 | 0.966 ± 0.004 | 0.967 ± 0.004 | 0.971 ± 0.002 | 0.942 ± 0.004 [0.901, 0.962] |
| A1 | 0.25 | 0.935 ± 0.004 | 0.966 ± 0.003 [0.953, 0.978] | 0.988 ± 0.001 [0.979, 0.995] | 0.967 ± 0.010 | 0.965 ± 0.014 | 0.966 ± 0.014 | 0.966 ± 0.003 | 0.937 ± 0.003 [0.895, 0.957] |
| B1 | 0 | 0.912 ± 0.004 | 0.966 ± 0.002 [0.953, 0.977] | 0.993 ± 0.001 [0.986, 0.997] | 0.958 ± 0.010 | 0.975 ± 0.008 | 0.974 ± 0.007 | 0.966 ± 0.002 | 0.902 ± 0.020 [0.855, 0.938] |
| B1 | 0.25 | 0.907 ± 0.007 | 0.961 ± 0.004 [0.949, 0.973] | 0.992 ± 0.001 [0.986, 0.997] | 0.955 ± 0.015 | 0.968 ± 0.012 | 0.968 ± 0.011 | 0.961 ± 0.004 | 0.895 ± 0.029 [0.854, 0.926] |

## 31. Confidence intervals and paired differences

Paired group-bootstrap differences (same resamples for both arms, seed-mean):

| Comparison | λ | Δ test bAcc [95% CI] | Δ test AUROC [95% CI] | Δ worst-domain bAcc [95% CI] |
|---|---:|---|---|---|
| A0 − A1 | 0 | -0.011 [-0.020, -0.002] | +0.002 [-0.004, +0.009] | -0.018 [-0.048, +0.007] |
| A0 − A1 | 0.25 | -0.007 [-0.019, +0.005] | -0.001 [-0.007, +0.006] | -0.019 [-0.059, +0.021] |
| A0 − B1 | 0 | -0.006 [-0.021, +0.008] | -0.003 [-0.009, +0.002] | +0.015 [-0.033, +0.062] |
| A0 − B1 | 0.25 | -0.002 [-0.017, +0.013] | -0.005 [-0.012, +0.002] | +0.018 [-0.034, +0.066] |
| A1 − B1 | 0 | +0.005 [-0.009, +0.018] | -0.005 [-0.012, +0.001] | +0.033 [-0.013, +0.079] |
| A1 − B1 | 0.25 | +0.005 [-0.009, +0.018] | -0.004 [-0.012, +0.002] | +0.037 [-0.007, +0.076] |
| A1 − A1s | 0 | +0.003 [-0.006, +0.011] | -0.003 [-0.007, +0.002] | +0.010 [-0.017, +0.036] |

Intervals for every metric are wide (the manipulated test class has 58 images), so small differences should not be over-read.

## 32. Confusion matrices

Binary (clean test, threshold 0.5, mean per seed, λ = 0; rows are counts out of 632):

| Arm | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| A0 | 304.3 | 302.3 | 13.7 | 11.7 |
| A1 | 308.3 | 305.3 | 10.7 | 7.7 |
| B1 | 302.7 | 308.0 | 8.0 | 13.3 |

3-class (λ = 0.25, summed over 3 seeds; rows true REAL / AI / MANIPULATED, columns predicted):

| Arm | REAL | AI_GENERATED | MANIPULATED |
|---|---|---|---|
| A0 | 918, 26, 4 | 22, 751, 1 | 24, 11, 139 |
| A1 | 918, 24, 6 | 8, 760, 6 | 17, 3, 154 |
| B1 | 926, 6, 16 | 4, 769, 1 | 36, 4, 134 |

Most common errors: B1 mislabels manipulated images as real (36 of 174, 21%), A1 does so less often (17 of 174, 10%) but confuses real with AI more (24 vs 6). All arms
separate AI from real well; the Fakeddit photo edits are the hard class.

## 33. Per-domain results

Clean test, threshold 0.5 (domains: news = REAL vs AI news images; coco = real COCO vs SD-512 counterfactuals; fakeddit = real Fakeddit vs Photoshop edits):

| Arm | λ | domain | n | bAcc | AUROC | fake recall | real recall |
|---|---|---|---|---|---|---|---|
| A0 | 0 | coco | 134 | 0.938 ± 0.004 | 0.969 ± 0.009 | 0.975 ± 0.019 | 0.900 ± 0.019 |
| A0 | 0 | fakeddit | 116 | 0.922 ± 0.007 | 0.988 ± 0.003 | 0.891 ± 0.022 | 0.954 ± 0.029 |
| A0 | 0 | news | 382 | 0.979 ± 0.002 | 0.996 ± 0.000 | 0.981 ± 0.007 | 0.977 ± 0.010 |
| A0 | 0.25 | coco | 134 | 0.943 ± 0.004 | 0.969 ± 0.002 | 0.970 ± 0.012 | 0.915 ± 0.007 |
| A0 | 0.25 | fakeddit | 116 | 0.914 ± 0.007 | 0.983 ± 0.001 | 0.851 ± 0.008 | 0.977 ± 0.008 |
| A0 | 0.25 | news | 382 | 0.979 ± 0.007 | 0.996 ± 0.002 | 0.970 ± 0.009 | 0.988 ± 0.007 |
| A1 | 0 | coco | 134 | 0.943 ± 0.004 | 0.971 ± 0.005 | 0.980 ± 0.007 | 0.905 ± 0.007 |
| A1 | 0 | fakeddit | 116 | 0.948 ± 0.012 | 0.969 ± 0.004 | 0.902 ± 0.033 | 0.994 ± 0.008 |
| A1 | 0 | news | 382 | 0.988 ± 0.002 | 0.998 ± 0.000 | 0.997 ± 0.002 | 0.979 ± 0.004 |
| A1 | 0.25 | coco | 134 | 0.940 ± 0.006 | 0.970 ± 0.003 | 0.970 ± 0.012 | 0.910 ± 0.021 |
| A1 | 0.25 | fakeddit | 116 | 0.943 ± 0.004 | 0.978 ± 0.005 | 0.885 ± 0.008 | 1.000 ± 0.000 |
| A1 | 0.25 | news | 382 | 0.983 ± 0.004 | 0.998 ± 0.001 | 0.991 ± 0.012 | 0.974 ± 0.017 |
| B1 | 0 | coco | 134 | 0.955 ± 0.006 | 0.990 ± 0.000 | 0.970 ± 0.012 | 0.940 ± 0.021 |
| B1 | 0 | fakeddit | 116 | 0.902 ± 0.020 | 0.988 ± 0.006 | 0.805 ± 0.041 | 1.000 ± 0.000 |
| B1 | 0 | news | 382 | 0.990 ± 0.004 | 0.999 ± 0.000 | 1.000 ± 0.000 | 0.979 ± 0.007 |
| B1 | 0.25 | coco | 134 | 0.943 ± 0.019 | 0.991 ± 0.001 | 0.960 ± 0.007 | 0.925 ± 0.037 |
| B1 | 0.25 | fakeddit | 116 | 0.902 ± 0.036 | 0.987 ± 0.003 | 0.805 ± 0.072 | 1.000 ± 0.000 |
| B1 | 0.25 | news | 382 | 0.986 ± 0.003 | 0.999 ± 0.000 | 0.998 ± 0.002 | 0.974 ± 0.007 |

Fake recall by AI family (λ = 0, mean over seeds): B1 detects 1.00 of the `dalle3_style`, `llm`, `midjourney_style` and `midjourney_v6` images and 0.97 of SD-512 counterfactuals; A1 0.99/1.00/1.00/1.00/0.98; the Fakeddit edits are 0.90 (A1), 0.89 (A0) and 0.81 (B1).

## 34. Held-out results

Source-held-out protocols (separate head per protocol; test is never seen in training):
`h_mj` holds out every Midjourney-family image (Midjourney-style `antifact` plus Midjourney V6) with matched real test images; `h_cf` trains and selects on the news domain
only and tests on SD-512 COCO counterfactuals vs real COCO (a new generator **and** a new content domain).

| Arm | protocol | λ | n test | bAcc | AUROC | fake recall | real recall |
|---|---|---|---|---|---|---|---|
| A0 | h_cf | 0 | 900 | 0.678 ± 0.036 | 0.875 ± 0.026 | 0.400 ± 0.075 | 0.956 ± 0.014 |
| A0 | h_cf | 0.25 | 900 | 0.693 ± 0.025 | 0.876 ± 0.021 | 0.421 ± 0.054 | 0.964 ± 0.007 |
| A0 | h_mj | 0 | 1022 | 0.962 ± 0.003 | 0.995 ± 0.001 | 0.935 ± 0.009 | 0.989 ± 0.002 |
| A0 | h_mj | 0.25 | 1022 | 0.964 ± 0.006 | 0.995 ± 0.001 | 0.936 ± 0.016 | 0.992 ± 0.004 |
| A1 | h_cf | 0 | 900 | 0.784 ± 0.020 | 0.885 ± 0.018 | 0.658 ± 0.038 | 0.910 ± 0.019 |
| A1 | h_cf | 0.25 | 900 | 0.805 ± 0.004 | 0.911 ± 0.009 | 0.684 ± 0.009 | 0.926 ± 0.009 |
| A1 | h_mj | 0 | 1022 | 0.978 ± 0.004 | 0.996 ± 0.000 | 0.971 ± 0.006 | 0.986 ± 0.002 |
| A1 | h_mj | 0.25 | 1022 | 0.973 ± 0.001 | 0.997 ± 0.000 | 0.955 ± 0.002 | 0.991 ± 0.002 |
| B1 | h_cf | 0 | 900 | 0.740 ± 0.053 | 0.928 ± 0.013 | 0.497 ± 0.110 | 0.983 ± 0.003 |
| B1 | h_cf | 0.25 | 900 | 0.763 ± 0.032 | 0.949 ± 0.004 | 0.551 ± 0.066 | 0.976 ± 0.005 |
| B1 | h_mj | 0 | 1022 | 0.986 ± 0.001 | 0.999 ± 0.000 | 0.981 ± 0.006 | 0.991 ± 0.005 |
| B1 | h_mj | 0.25 | 1022 | 0.985 ± 0.006 | 0.999 ± 0.000 | 0.977 ± 0.012 | 0.992 ± 0.003 |

Held-out tables report seed mean ± std only (no bootstrap). On Midjourney all arms are strong and B1 is best. On the COCO counterfactuals every arm drops: B1 keeps the best AUROC but misses many fakes at threshold 0.5 (fake recall 0.50–0.55), A1 detects more (0.66–0.68).

## 35. JPEG and blur robustness

Clean-trained heads evaluated on JPEG quality 75 and Gaussian blur radius 1 test views (core val/test only):

| Arm | λ | JPEG-75 bAcc | blur bAcc | JPEG-75 AUROC | blur AUROC |
|---|---|---|---|---|---|
| A0 | 0 | 0.959 ± 0.001 | 0.946 ± 0.005 | 0.989 ± 0.002 | 0.984 ± 0.002 |
| A0 | 0.25 | 0.958 ± 0.001 | 0.941 ± 0.007 | 0.987 ± 0.001 | 0.983 ± 0.002 |
| A1 | 0 | 0.969 ± 0.004 | 0.963 ± 0.004 | 0.986 ± 0.002 | 0.986 ± 0.003 |
| A1 | 0.25 | 0.965 ± 0.002 | 0.960 ± 0.002 | 0.987 ± 0.001 | 0.987 ± 0.003 |
| B1 | 0 | 0.969 ± 0.004 | 0.949 ± 0.005 | 0.993 ± 0.001 | 0.990 ± 0.001 |
| B1 | 0.25 | 0.963 ± 0.003 | 0.953 ± 0.002 | 0.992 ± 0.001 | 0.990 ± 0.001 |

JPEG-75 hardly moves any arm; blur lowers every arm slightly (B1 the most in balanced accuracy, though its AUROC stays 0.990).

## 36. Source leakage

Linear probes on the frozen `v_imgfor` of real images (fit on train reals, scored on test reals; domain probe has chance 0.33, news JPEG/PNG probe chance 0.50):

| Arm | λ | participation ratio | rank@95% | domain probe bAcc (chance 0.33) | news JPEG/PNG probe bAcc (chance 0.50) |
|---|---|---|---|---|---|
| A0 | 0 | 2.08 ± 0.43 | 162.0 ± 51.8 | 0.785 ± 0.022 | 0.754 ± 0.023 |
| A0 | 0.25 | 1.82 ± 0.25 | 132.7 ± 38.1 | 0.786 ± 0.019 | 0.746 ± 0.024 |
| A1 | 0 | 1.52 ± 0.02 | 6.7 ± 0.5 | 0.740 ± 0.006 | 0.723 ± 0.028 |
| A1 | 0.25 | 1.84 ± 0.29 | 10.0 ± 1.4 | 0.736 ± 0.003 | 0.747 ± 0.026 |
| B1 | 0 | 1.60 ± 0.05 | 15.3 ± 0.5 | 0.765 ± 0.007 | 0.796 ± 0.009 |
| B1 | 0.25 | 1.59 ± 0.09 | 17.0 ± 1.4 | 0.756 ± 0.030 | 0.798 ± 0.014 |

Every arm encodes source clearly above chance (domain probe 0.72–0.79, format probe 0.72–0.80). The v2 sampler and augmentation reduce but do not remove it. A1 leaks least,
B1 slightly more (format probe 0.80), A0 the most on the domain probe. Leakage is a risk for every candidate.

## 37. Effective rank

Participation ratio of the `v_imgfor` covariance spectrum is only about 1.5–2.1 for all arms, i.e. the 768-d vector is dominated by one or two directions (the real/fake axis). The number of
components for 95% of the variance: B1 15–17, A1 7–10, A0 about 130–160 (A0's linear CLIP input is much more spread out). So B1's vector is somewhat richer than A1's but neither is a high-dimensional representation.

## 38. Binary-only vs multi-task

Auxiliary 3-class results (clean test, λ = 0.25):

| Arm | λ | macro-F1 | accuracy | MANIPULATED recall | MANIPULATED precision | AI recall | REAL recall |
|---|---|---|---|---|---|---|---|
| A0 | 0.25 | 0.932 ± 0.011 | 0.954 ± 0.005 | 0.799 ± 0.043 | 0.966 ± 0.025 | 0.970 ± 0.002 | 0.968 ± 0.000 |
| A1 | 0.25 | 0.950 ± 0.004 | 0.966 ± 0.003 | 0.885 ± 0.008 | 0.929 ± 0.027 | 0.982 ± 0.005 | 0.968 ± 0.007 |
| B1 | 0.25 | 0.927 ± 0.008 | 0.965 ± 0.003 | 0.770 ± 0.043 | 0.888 ± 0.011 | 0.994 ± 0.004 | 0.977 ± 0.005 |

| Effect of λ = 0.25 vs 0 | A0 | A1 | B1 |
|---|---|---|---|
| test balanced accuracy | −0.001 | −0.005 | −0.005 |
| worst-domain balanced accuracy | −0.008 | −0.005 | −0.007 |
| held-out COCO balanced accuracy | +0.015 | +0.021 | +0.023 |
| domain-probe accuracy | 0.786 vs 0.785 | 0.736 vs 0.740 | 0.756 vs 0.765 |
| effective rank (participation ratio) | 1.82 vs 2.08 | 1.84 vs 1.52 | 1.59 vs 1.60 |

The auxiliary head did not improve binary accuracy, did not help manipulated recall in B1 (3-class manipulated recall 0.77 vs binary fake recall 0.805 on Fakeddit) and did not change source leakage.
Small gains on the held-out COCO test (about +2 points) are within seed noise (std 0.004–0.05). The 3-class head is only informative as a diagnostic.

## 39. A1 vs B1 direct comparison

| Item | A1 (current encoders + shared head) | B1 (DINOv2-L + shared head) |
|---|---|---|
| Test balanced accuracy (λ=0) | 0.971 ± 0.001 | 0.966 ± 0.002 |
| Test AUROC | 0.987 | 0.993 |
| Fake recall / real recall | 0.976 / 0.966 | 0.958 / 0.975 |
| Worst-domain balanced accuracy | 0.942 | 0.902 |
| Fakeddit (manipulation) fake recall | 0.902 | 0.805 |
| Held-out Midjourney bAcc | 0.978 | 0.986 |
| Held-out COCO bAcc / AUROC | 0.784 / 0.885 | 0.740 / 0.928 |
| JPEG-75 / blur bAcc | 0.969 / 0.963 | 0.969 / 0.949 |
| Domain / format probe | 0.740 / 0.723 | 0.765 / 0.796 |
| Effective rank (95% components) | 6.7 | 15.3 |
| Raw / output feature dimension | 4,864 / 768 | 8,192 / 768 |
| External forensic pretraining | **yes** (GenImage, upstream forensic checkpoint) | **no** (generic DINOv2) |
| Backbones at inference | CLIP-L + 2 ResNet-50 + CPU DCT | one ViT-L |
| Semantic-branch overlap | CLIP-L (same checkpoint) | none |
| Extraction time (clean view) | 136 s | 104 s |
| Maintainability | three checkpoints, vendored upstream code | one Hugging Face model |

## 40. Winner

- **Unconstrained (A0 vs A1 vs B1):** inconclusive between A1 and B1. A1 is numerically ahead on the primary selection metric (worst-domain balanced accuracy on both validation, 0.935 vs 0.912, and test, 0.942 vs 0.902), driven entirely by manipulation; the differences are inside the confidence intervals. B1 is ahead on AUROC and Midjourney.
- **Under the project constraints (team data only, no GenImage-derived weights):** A1 is not eligible. Between the eligible arms, B1 matches A0 on balanced accuracy (Δ −0.006, CI −0.021…+0.008), has a numerically higher AUROC (not significant) and better held-out Midjourney and COCO AUROC (seed std only; no bootstrap for held-out tests), no overlap with the Semantic Branch's CLIP, and a lower worst-domain accuracy (0.902 vs 0.922, not significant). **B1 λ=0 is the production choice.**
- **Remaining weaknesses:** manipulation detection (recall about 0.80), source leakage, a one-dimensional embedding, wide intervals from a small test set, and unstable thresholds.

## 41. Final recommended production architecture

```
Image (PIL)
  └─ RGB, drop alpha/metadata, shorter side 224 (bicubic), center crop 224×224
      └─ DINOv2-L/14 (facebook/dinov2-large, frozen, fp32)
          └─ hidden_states of blocks 12, 16, 20, 24: CLS + mean patch each → 8,192 raw features
              └─ standardise (train mean/std stored in the checkpoint)
                  └─ Linear(8192→1024) → GELU → Dropout(0.2, train only) → Linear(1024→768) → LayerNorm
                      └─ v_imgfor [768] ──► multimodal fusion
                          └─ binary head Linear(768→1) → P(fake); decision at threshold 0.5
```

Trainable parts: the bottleneck and heads only (36.8 MB checkpoint). Loss: binary BCE (λ_aux = 0). Sampler: 50/35/15 with in-domain balance. Checkpoint selection: best worst-domain validation balanced accuracy, ties by validation BCE.
Chosen checkpoint: `B1_main_lam0_s47` (validation worst-domain balanced accuracy 0.9153, tied with seed 13, lower validation BCE 0.1195). Runtime outputs: `features` [B,768], `probability` [B], optional none.

## 42. Integration with multimodal fusion

- `v_imgfor` is [B, 768]; `V3FusionModule` consumes it unchanged (shape verified: `[1,768]` → `main_logits [1,5]`).
- **Fusion must be re-extracted and retrained.** The new vector is in a different feature space, so the old cached `image` column (news bundle) and the old fusion checkpoint are incompatible.
- `fnd/imagev2/infer.py` exposes the same interface as the old `ImageBranch` (`features`, `probability`, `config['threshold']`), but `dashboard/server.py` still imports the old class; switching it is a remaining step.
- Stacking caveat: the Image Branch is trained on 70% of the project-train images and fusion trains on the same images, so image features on fusion-train rows are overconfident compared with unseen data; check the gap on the project val split.

## 43. Reproducibility

See [How to run the final Image Branch](#how-to-run-the-final-image-branch). Tests: `python -m pytest fnd/tests/test_image_branch_v2.py fnd/tests/test_imagev2.py -q`. Mapping is deterministic; extraction is bit-reproducible for clean views; training is seeded.

## 44. Git-tracked files

Code: `fnd/data/build_image_branch_v2.py`, `fnd/imagev2/*.py` (canonical, manifest, extract, sampler, model, metrics, train_head, aggregate, validate_cache, infer), tests `fnd/tests/test_image_branch_v2.py`, `fnd/tests/test_imagev2.py`; docs `docs/IMAGE_DATA_MAPPING.md`, this report; compact manifests `data/image_branch_v2/{mapping_rules.csv,summary.json,excluded_ambiguous.csv,*_val.csv,*_test.csv}`.

## 45. Generated artifacts (not in Git)

Feature caches, run directories (config, metrics, predictions, curves, checkpoints), aggregate tables, logs, the runtime bundle and the Hugging Face cache; all on `D:` (see the table below).

## 46. Limitations

- Small data: 390 manipulated images (273 train, 58 test) from one source (Fakeddit photo edits); the test set is 632 images, so intervals are wide and many differences are not significant.
- The manipulation class is limited to Photoshop-style composites; nothing here measures face swaps, inpainting or other edits. Do not call this a general manipulation detector.
- A1 contains external forensic knowledge; B1 and A0 are generic foundation models. Generators other than Midjourney V6 and the 15 SDXL images are style inferences.
- Source and format leakage remains (probes 0.72–0.80 vs chance); the v2 data has resolution, format and domain shortcuts that augmentation only partly removes.
- The COCO counterfactual label (AI_GENERATED, medium-high confidence) is the least certain mapping rule; the held-out COCO test mixes a new generator and a new domain.
- No external benchmark was evaluated; no claim is made beyond the tested domains. DINOv3 and LoRA adaptation were not tested.
- Validation-tuned thresholds are unstable; fusion integration is not done; no plots were generated (per-run `curves.json` exist).

## 47. Future DINOv3 experiment

After access is granted: `python -m fnd.imagev2.extract --arm B1 --model facebook/dinov3-vitl16-pretrain-lvd1689m --out D:\MultiGuard\image_branch_v2\features\B1_dinov3 ...`, check the printed layer taps and token layout (DINOv3 ViT-L/16: 4 register tokens, hidden size 1,024, patch 16), then
`python -m fnd.imagev2.train_head --arm B1_dinov3 ...` with the same seeds and λ values, and compare against B1 with `fnd.imagev2.aggregate`. LoRA adaptation is **not** recommended yet: frozen B1 is competitive but not clearly better than the alternatives, its weak spot is manipulation (273 training images from one source), and adapting a 300M model on that
risks memorising the source; run it only after the DINOv3 frozen comparison and only with a held-out manipulation source.

## 48. Final next steps

1. Keep B1 λ=0 (`B1_main_lam0_s47`) as the Image Branch candidate; keep A1 only as a reference ablation.
2. Re-extract `v_imgfor` for fusion with the bundle and retrain the fusion head; compare on the project val split.
3. Switch `dashboard/server.py` to `fnd.imagev2.infer.ImageBranchV2` (same interface) and verify end-to-end.
4. Test DINOv3 when access is approved; consider more manipulation data to address the weakest class.
5. Back up the bundle and DINOv2 files (they are machine-local).

## Artifacts and Paths

All paths are on the Training PC; the `D:\...` paths are **machine-specific**.

### Repository / code

| Artifact | Purpose | Exact path | Git tracked? | Required for inference/training? |
|---|---|---|---|---|
| Experiment worktree | working copy of the branch | `D:\MultiGuard\wt\image-branch-v2` | n/a | training: yes |
| Branch | experiment branch (based on local main) | `experiment/image-branch-v2` | n/a | |
| Mapping builder | rebuilds manifests | `D:\MultiGuard\wt\image-branch-v2\fnd\data\build_image_branch_v2.py` | yes | training |
| Image Branch v2 code | extract / train / aggregate / infer | `D:\MultiGuard\wt\image-branch-v2\fnd\imagev2\` | yes | inference: `canonical.py, extract.py, model.py, infer.py` |
| Tests | unit tests | `fnd\tests\test_image_branch_v2.py`, `fnd\tests\test_imagev2.py` | yes | no |
| Docs | mapping + this report | `docs\IMAGE_DATA_MAPPING.md`, `docs\IMAGE_BRANCH_V2_EXPERIMENT_REPORT.md` | yes | no |
| Configs | hyper-parameters are CLI defaults (`DEFAULTS` in `train_head.py`) and stored per run (`config.json`) | n/a | yes | |

### Dataset

| Artifact | Purpose | Exact path | Git tracked? | Required for inference/training? |
|---|---|---|---|---|
| Team dataset root | source of images and metadata | `D:\MultiGuard\data\fnd_team_dataset_2026-09-14` | no | training: yes; runtime: no |
| Images | 21,169 files | `D:\MultiGuard\data\fnd_team_dataset_2026-09-14\images\` | no | training: yes |
| `master.csv`, `image_samples.csv`, `protected_external_evaluations.csv` | metadata | `...\data\` | no | training: yes |
| Metadata-only copy (no images) | convenience | `C:\Users\497-MultiGuard\Desktop\Multiguard_windows_demo\multiguard_dataset\fnd_team_dataset_2026-09-14` | no | no |
| Committed manifests | audit | `data\image_branch_v2\{mapping_rules.csv,summary.json,excluded_ambiguous.csv,*_val.csv,*_test.csv}` | **yes** | no |
| Generated manifests | train/all | `data\image_branch_v2\*_train.csv`, `*_all.csv` (git-ignored) | no | training: yes |

### Current (old) model assets

| Artifact | Purpose | Exact path | Git tracked? | Required for inference/training? |
|---|---|---|---|---|
| CLIP-L/14 | A0/A1 CLIP path; also used by the Semantic Branch | `D:\hf_cache\hub\models--openai--clip-vit-large-patch14\snapshots\32bd64288804d66eefd0ccbe215aa642df71cc41` (copy in `C:\Users\497-MultiGuard\.cache\huggingface`) | no | A arms only |
| RGB ResNet-50 | A1 | `C:\Users\497-MultiGuard\Desktop\Multiguard_windows_demo\multimodal-fake-news-v2\models\image\news\rgb_robust.pt` (SHA-256 `30e54db3…`) | no | A1 only |
| DCT ResNet-50 | A1 | `...\models\image\news\dct_guide.pt` (SHA-256 `23e3ba53…`) | no | A1 only |
| Old bundle, head, projection | the deployed `news` variant | `...\models\image\news\bundle.json`, `head.pt` | bundle.json yes; weights no | old pipeline only; **not used** in the new comparison |
| Upstream code | vendored for RGB ResNet | `external\FakeImageDetection` (submodule, commit `55b7142`) | submodule | A1 only |

### DINOv2

| Artifact | Purpose | Exact path | Git tracked? | Required for inference/training? |
|---|---|---|---|---|
| Model ID | backbone | `facebook/dinov2-large` | no | **yes** |
| Snapshot / revision | pinned version | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` | stored in `bundle.json` | **yes** |
| Local cache | weights | `D:\hf_cache\hub\models--facebook--dinov2-large\snapshots\47b73eefe95e8d44ec3623f8890bd894b6ea2d6c\` (`model.safetensors`, 1.22 GB) | no | **yes** (set `HF_HOME=D:\hf_cache`) |

### Feature caches (training-only)

| Artifact | Purpose | Exact path | Git tracked? | Required for inference/training? |
|---|---|---|---|---|
| A1 cache (also A0 = first 768 columns) | 4,864-d features, 5 views | `D:\MultiGuard\image_branch_v2\features\A1\` | no | training only |
| A1s cache | RGB+DCT with squash geometry | `D:\MultiGuard\image_branch_v2\features\A1s\` | no | training only |
| B1 cache | DINOv2-L 8,192-d features, 5 views | `D:\MultiGuard\image_branch_v2\features\B1_dinov2\` | no | training only |
| Robustness views | `jpeg75.npy`, `blur1.npy` inside each cache | same folders | no | evaluation only |

### Checkpoints (all training-only unless marked)

Every run directory is `D:\MultiGuard\image_branch_v2\runs\<arm>_<protocol>_lam<λ>_s<seed>\` with `checkpoint.pt` (main protocol only), `config.json`, `metrics.json`, `curves.json`, `predictions.npz`, `env.json`.

| Model / config | λ | Seed | Checkpoint path | Recommended for production? |
|---|---|---|---|---|
| A0 best | 0 / 0.25 | 13, 29, 47 | `runs\A0_main_lam{0,0.25}_s{13,29,47}\checkpoint.pt` | no |
| A1 best | 0 / 0.25 | 13, 29, 47 | `runs\A1_main_lam{0,0.25}_s{13,29,47}\checkpoint.pt` | no (reference) |
| A1 / A1s geometry check | 0 | 13, 29, 47 | `runs\A1_tvclean_main_lam0_s*`, `runs\A1s_tvclean_main_lam0_s*` | no |
| B1 best | 0 / 0.25 | 13, 29, 47 | `runs\B1_main_lam{0,0.25}_s{13,29,47}\checkpoint.pt` | **`runs\B1_main_lam0_s47\checkpoint.pt`** |
| Held-out protocol runs | 0 / 0.25 | 13, 29, 47 | `runs\*_h_mj_*`, `runs\*_h_cf_*` (no checkpoint saved) | no |

### Results

| Artifact | Purpose | Exact path | Git tracked? |
|---|---|---|---|
| Aggregate tables | all tables in this report | `D:\MultiGuard\image_branch_v2\results\tables.md` | no |
| Aggregate JSON (means, CIs, deltas) | machine-readable | `D:\MultiGuard\image_branch_v2\results\summary.json` | no |
| Per-run metrics / confusion matrices | per seed | `...\runs\<run>\metrics.json` (`three_class.confusion`, `binary.tp/tn/fp/fn`) | no |
| Bootstrap | computed on the fly in `aggregate.py` from `predictions.npz` (1,000 group resamples) | `...\runs\<run>\predictions.npz` | no |
| Experiment-A-only snapshot | earlier table | `D:\MultiGuard\image_branch_v2\results_A\tables.md` | no |
| Plots | none generated | n/a | n/a |
| Logs | extraction, training, aggregation | `D:\MultiGuard\image_branch_v2\logs\` | no |

### Final production assets

**REQUIRED FOR FINAL IMAGE BRANCH INFERENCE**

| Asset | Exact path |
|---|---|
| Backbone weights | `D:\hf_cache\hub\models--facebook--dinov2-large\snapshots\47b73eefe95e8d44ec3623f8890bd894b6ea2d6c\model.safetensors` |
| Bundle (preprocessing, taps, threshold, target, head config, checksums) | `D:\MultiGuard\image_branch_v2\production\B1_dinov2_lam0_s47\bundle.json` |
| Image-branch head checkpoint (includes the train mean/std normalisation buffers) | `D:\MultiGuard\image_branch_v2\production\B1_dinov2_lam0_s47\head.pt` (SHA-256 in the bundle) |
| Runtime code | `fnd/imagev2/infer.py`, `extract.py` (DinoEncoders), `canonical.py`, `model.py` |
| Example output | `D:\MultiGuard\image_branch_v2\production\example_v_imgfor.pt` ([1,768]) |

**TRAINING-ONLY / NOT REQUIRED AT RUNTIME**

Train/all manifests, feature caches, every other run directory and optimizer state, `predictions.npz` (bootstrap inputs), logs, `results\`, the old CLIP/RGB/DCT weights, the old head/projection and the team dataset images.

## How to run the final Image Branch

Environment: system Python 3.13 (torch 2.11+cu128, transformers 5.2.0), working directory `D:\MultiGuard\wt\image-branch-v2`.

```powershell
$env:HF_HOME = "D:\hf_cache"
$DATA = "D:\MultiGuard\data\fnd_team_dataset_2026-09-14"; $OUT = "D:\MultiGuard\image_branch_v2"
```

### Rebuild dataset mapping
```powershell
python -m fnd.data.build_image_branch_v2 --dataset-root $DATA --output-dir data\image_branch_v2 --check-determinism
```

### Extract features
```powershell
$env:CUDA_VISIBLE_DEVICES = "1"     # B1 on GPU 1 (the process sees it as cuda:0)
python -m fnd.imagev2.extract --arm B1 --model facebook/dinov2-large --dataset-root $DATA --manifest-dir data\image_branch_v2 --out $OUT\features\B1_dinov2 --device cuda:0 --workers 10
python -m fnd.imagev2.validate_cache --features $OUT\features\B1_dinov2 --manifest-dir data\image_branch_v2
# A1 (GPU 0): python -m fnd.imagev2.extract --arm A1 --bundle <repo>\models\image\news\bundle.json --out $OUT\features\A1 --device cuda:0 ...
```

### Train the selected Image Branch
```powershell
python -m fnd.imagev2.train_head --arm B1 --features $OUT\features --runs $OUT\runs --manifest-dir data\image_branch_v2 --lambdas 0 0.25 --seeds 13 29 47 --protocols main h_mj h_cf
```

### Evaluate
```powershell
python -m fnd.imagev2.aggregate --runs $OUT\runs --manifest-dir data\image_branch_v2 --out $OUT\results --n-boot 1000
```

### Export the runtime bundle (selected run)
```powershell
python -c "from fnd.imagev2.infer import export_bundle; export_bundle(r'$OUT\runs\B1_main_lam0_s47', r'$OUT\production\B1_dinov2_lam0_s47')"
```

### Run one-image inference and generate `v_imgfor` [768]
```powershell
python -m fnd.imagev2.infer --bundle $OUT\production\B1_dinov2_lam0_s47\bundle.json --image photo.jpg --features-out v_imgfor.pt
```
```python
from PIL import Image
from fnd.imagev2.infer import ImageBranchV2
branch = ImageBranchV2(r"D:\MultiGuard\image_branch_v2\production\B1_dinov2_lam0_s47\bundle.json", device="cuda:0")
out = branch([Image.open("photo.jpg")])      # out["features"]: [1, 768]; out["probability"]: [1]
```
(Runtime and cached-feature paths were checked to agree exactly: maximum difference 0.0 on 24 validation images.)

### Integrate / run with the multimodal fusion
**Not implemented yet.** Remaining steps: (1) recompute `v_imgfor` for every fusion row with `ImageBranchV2` (same `sample_id` joins), (2) retrain with `python -m fnd.train_fusion --features <new npz> ...`, (3) change the import and `--image-bundle` handling in `dashboard/server.py` to `ImageBranchV2` and its `bundle.json`, (4) verify end to end. Only the shape contract (`[1,768]` into `V3FusionModule`) has been verified so far.

## Final Deployment Checklist

- [x] final backbone available locally (`D:\hf_cache`, DINOv2-L pinned revision)
- [x] final Image Branch checkpoint available (`production\B1_dinov2_lam0_s47\head.pt`)
- [x] preprocessing config available (`bundle.json`)
- [x] normalization statistics available (stored in `head.pt`)
- [x] threshold stored (0.5, in `bundle.json`)
- [x] `v_imgfor` output verified as [1,768]
- [ ] fusion compatibility verified end to end (shape only; fusion not retrained)
- [ ] dashboard integration verified (not switched over)
- [ ] required files backed up (everything is on one machine; copy the bundle and the DINOv2 snapshot elsewhere)
- [x] generated training artifacts excluded from Git
- [ ] DINOv3 comparison run (access still pending)
- [ ] more manipulation data/sources evaluated
