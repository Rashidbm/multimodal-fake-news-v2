# Image data mapping (Image Branch v2)

Image-only truth labels and task manifests for the next Image Branch, built from our team
dataset. No model is trained or changed by this mapping.

## 1. Purpose

The team dataset labels *pairs* (image + caption) for news classification. Image forensics needs a
label for the *image itself*, and the existing columns cannot provide it (section 3). This
mapping assigns each image one of three truths, from explicit per-source rules, and builds
group-safe train/val/test manifests for three tasks.

## 2. Dataset scope

- **Team dataset only** (`fnd_team_dataset_2026-09-14`: MMFakeBench + NewsCLIPpings material).
- **Original project `split == train` only.** Project `val` and `test` rows are never loaded into a manifest.
- **No GenImage, no DGM4, no external data.** The package contains no face-swap or face-attribute
  manipulations, so the manipulation class covers only the manipulation data we have (section 9).
- Protected external-evaluation fingerprints are checked and none is included.
- Images are not copied or modified; manifests hold paths relative to the dataset root.

## 3. Why the existing labels are not enough

| Column | Why it cannot be the image truth |
|---|---|
| `scenario`, `label_binary` | Scenario 1 (out of context) and 2 (fake text) are "fake news" but their images are authentic. |
| `image_fake` | 1 for scenarios 3 and 5, but it mixes AI generation (e.g. `fever_AI`), Photoshop edits (`Fakeddit_photo_edit`) and synthetic counterfactuals (`coco_*`). |
| MMFakeBench `fake_cls` | `mismatch` covers real OOC images, a DGM4 caption edit on a real image, and fully synthetic COCO images. `textual_veracity_distortion` covers both real and AI images. (Not present in the package; it is documented in the dataset paper.) |
| Folder names | `coco_image_edit` is named an edit, but its images are uniform 512x512 synthetic images, not edited real photos. `coco_text_edit` has the same image type. |
| `source_split` | Original source split, not our project split. |

## 4. Final image truth classes

| Truth | Label (joint) | Meaning |
|---|---:|---|
| `REAL` | 0 | Authentic image by dataset construction, including images under fake text or OOC captions |
| `AI_GENERATED` | 1 | Fully generated image |
| `MANIPULATED` | 2 | Real photo that was edited (Photoshop-style composites) |
| `EXCLUDE` | | Unresolved or low confidence; never in a task manifest |

## 5. Mapping rules

Encoded in `RULES` in `fnd/data/build_image_branch_v2.py` and written to `mapping_rules.csv`.

| Source | Subcategory | Original meaning | Truth | Confidence | Reason |
|---|---|---|---|---|---|
| NewsCLIPpings | (paired) | matched and mismatched (OOC) | REAL | high | Mismatch is the caption; the image is authentic |
| newsclippings | bbc, guardian, usa_today, washington_post | matched supplement | REAL | high | Genuine news images |
| MMFB | bbc, guardian, usa_today, wash, coco | genuine | REAL | high | Paper section 3.2 |
| MMFB | fakeddit | genuine | REAL | medium-high | Paper section 3.2 (user photos, memes) |
| MMFB | Newsclipings_person / scene / semantic | OOC | REAL | high | Image untouched |
| MMFB | rumor_match, politicat_match, gossipcop_match, chatgpt_match | fake text | REAL | high | Repurposed real photo |
| MMFB | DGM4_text_edit_senti | fake text | REAL | high | Only the caption is edited |
| MMFB | Fakeddit_photo_edit | fake image | MANIPULATED | medium-high | Photoshop-style composites; native sizes 280-5285 px |
| MMFB | antifact_image_generation (1024x1024) | fake image | AI_GENERATED | high | Fully generated, Midjourney-like style |
| MMFB | fever_AI | fake text + fake image | AI_GENERATED | high | 1024x1024; 15 PNGs carry `sd_xl_base_1.0` metadata; the rest look DALL-E-3-like |
| MMFB | llm_rewrite, llm_gossip_md_generation, llm_science_md_generation | fake text + fake image | AI_GENERATED | high | 1024x1024 generated |
| MMFB | gossipcop_midjourney | fake text + fake image | AI_GENERATED | high | Named Midjourney V6 |
| MMFB | coco_image_edit, coco_text_edit (512x512) | image edit / fake text + fake image | AI_GENERATED | medium-high | Uniform 512x512 synthetic COCO-Counterfactuals images, visually diffusion-generated |
| MMFB | antifact_image_generation (not 1024x1024, 11 images) | fake image | EXCLUDE | low | Likely AI-edited real photos; ambiguous |

A group that fails its pixel-size check is excluded rather than force-labelled. A subcategory
with no rule stops the build with an error.

Generator names are inferred from image style and size, except Midjourney V6 (named) and the 15
SDXL images (metadata). Treat `ai_family` as a grouping aid, not a verified generator label.

## 6. Final counts (original train, unique images)

| | Images |
|---|---:|
| `master.csv` rows with split train | 17,794 |
| Unique images (duplicates collapsed) | **17,625** |
| REAL | 15,317 |
| AI_GENERATED | 1,907 |
| MANIPULATED | 390 |
| EXCLUDE | 11 |

169 rows were duplicates (104 real images that appear under several captions).

## 7. Internal split

- 70 / 15 / 15 by **group**, using seed `image_branch_v2_20261006`, stratified by (domain, truth, sub-source).
- One split per image across all three tasks, so an image never changes split between tasks.
- Groups link images that are the same image, near-duplicates (dHash Hamming distance <= 4), or share a
  caption of at least four words. No group crosses a split.
- Original project val/test images are never used.

## 8. AI detector (`ai_detector_*.csv`)

REAL (0) vs AI_GENERATED (1); MANIPULATED and EXCLUDE are left out. The Fakeddit domain is not in this pool.

Each row has a `balance_role`:

- `core`: the balanced set, 1:1 within each (split, domain): news (AI 1,283 vs real news, half
  NewsCLIPpings JPEG and half MMFB PNG) and COCO (counterfactuals vs 450 real COCO photos).
- `reserve_real`: extra real news images, to be resampled each epoch.
- `reserve_positive`: extra counterfactual images (COCO real is the limit).

| Split | Real core | AI core | Reserve real | Reserve AI |
|---|---:|---:|---:|---:|
| train | 1,215 | 1,215 | 9,191 | 122 |
| val | 260 | 260 | | |
| test | 258 | 258 | | |

`*_val.csv` and `*_test.csv` contain core rows only; the full set (including reserves) is in `*_all.csv`.
`split_h_mj` and `split_h_cf` are two source-held-out protocols: all Midjourney-family images as test (511 AI, 511 real), and all COCO counterfactuals vs real COCO as test (450 each, train on news only).

## 9. Manipulation detector (`manip_detector_*.csv`)

REAL (0) vs MANIPULATED (1), using real negatives from the same Fakeddit domain.

| Split | Real core | Manipulated | Reserve real |
|---|---:|---:|---:|
| train | 273 | 273 | 47 |
| val | 59 | 59 | |
| test | 58 | 58 | |

**Limitation.** The only manipulation source in our data is `Fakeddit_photo_edit` (Photoshop-style
composites on one platform). This detector covers that domain; it is **not a universal manipulation
detector**. It has no face-swap, face-attribute or inpainting examples, no held-out manipulation
type is possible, and the test set has 58 positives.

## 10. Joint 3-class (`joint_forensic_*.csv`)

Labels: 0 REAL, 1 AI_GENERATED, 2 MANIPULATED. Natural counts, no duplicated or oversampled rows.

| Split | Real (core + reserve) | AI (core + reserve) | Manipulated |
|---|---:|---:|---:|
| train | 1,488 + 9,238 | 1,215 + 122 | 273 |
| val | 319 | 260 | 59 |
| test | 316 | 258 | 58 |

Balance in the training sampler, not in the files: class-balanced sampling with equal thirds per
epoch, drawing reals from the core and reserve; MANIPULATED is oversampled about 4x relative to
the AI class in the core, so use strong augmentation and treat its metrics cautiously.

## 11. Shortcut risk and mandatory standardization

A one-off audit (gradient boosting on file metadata, trained on core-train, tested on core-test):

| Task | Metadata-only AUC |
|---|---:|
| AI vs real (news + COCO) | 1.000 |
| COCO counterfactual vs real COCO | 1.000 |
| Manipulation (Fakeddit core) | 0.990 |
| AI vs real, format/mode/EXIF only (no sizes) | 0.81 |

Drivers: real MMFB images are mostly PNG at 256 px height or 320 px width; AI images are 1024x1024 or
512x512; Fakeddit real images are all 320 px wide while the edits keep their native high resolution;
MMFB real images are always RGBA while AI images are RGB or RGBA; 15 SDXL PNGs carry generation
metadata. After converting to RGB and resizing to 224x224, cheap pixel statistics reach AUC 0.61-0.76
(partly genuine signal). The audit script is not part of the repo.

Training must use one pipeline for every image of every class:

1. Decode, convert to RGB, drop the alpha channel.
2. Ignore EXIF and PNG text metadata.
3. Resize with one policy and interpolation to a fixed size.
4. Random JPEG re-encoding (quality 60-95) and random resize augmentation.
5. For Fakeddit: downscale the edited images to the real images' width (320 px) before the common resize.

Original images must not be overwritten. This is a documented requirement; nothing is preprocessed here.

## 12. Leakage checks

`validate()` in the build script asserts: every image exists and belongs to original train; no
protected fingerprint (SHA-1 or dHash) is included; no image appears in two splits; no group crosses a
split; each task contains only its allowed truths; no excluded image is in a task manifest; val/test
files contain core rows only; paths are relative; a rebuild with the same seed gives identical manifests
(`--check-determinism`).

Not covered: near-duplicates beyond a 64-bit dHash and caption overlap.

## 13. Reproduction

```powershell
python -m fnd.data.build_image_branch_v2 `
  --dataset-root "D:\MultiGuard\data\fnd_team_dataset_2026-09-14" `
  --output-dir "data\image_branch_v2" --check-determinism
python -m pytest fnd/tests/test_image_branch_v2.py -q
```

Committed in Git: `mapping_rules.csv`, `summary.json`, `excluded_ambiguous.csv` and the compact
`*_val.csv` / `*_test.csv` manifests (about 1 MB). The larger `*_train.csv` and `*_all.csv`
manifests are git-ignored; regenerate them with the command above (the build is deterministic).

Options: `--seed` (default `image_branch_v2_20261006`), `--skip-hash-check`, `--workers`. The
build writes the CSVs, `mapping_rules.csv`, `excluded_ambiguous.csv` and `summary.json`.

Manifest columns: `image_id`, `image_path`, `source`, `subcategory`, `all_subcategories`,
`original_project_split`, `image_truth`, `label`, `confidence`, `mapping_rule`, `ai_family`,
`domain`, `subsource`, `group_id`, `split`, `balance_role`, `width`, `height`, `format`, `mode`,
`file_size`, `sha256`, `n_caption_rows`. `original_scenarios` and `original_image_fake` are original
metadata, **not ground truth**. Join `image_id` to `master.csv` for the caption `sample_id`s.
