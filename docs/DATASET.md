# Dataset: `fnd_team_dataset_2026-09-14`, MMFakeBench usage and the unseen evaluation subset

All numbers below were recomputed from the real files (`data/master.csv` of the team package and the
Hugging Face `liuxuannan/MMFakeBench` `MMFakeBench_val.json` / `MMFakeBench_test.json`), not estimated.

## Project dataset

`fnd_team_dataset_2026-09-14` has **21,404** image-caption pairs with a fixed `split` column (never
re-split per branch). Sources: **NewsCLIPpings 11,562** and **MMFakeBench 9,842**. DGM4 images are not a
separate source here (one MMFakeBench subcategory is named `DGM4_text_edit_senti`).

| Split | Total | NewsCLIPpings | MMFakeBench |
|---|---:|---:|---:|
| Train | 17,794 | 10,422 | 7,372 |
| Validation | 2,375 | 1,140 | 1,235 |
| Test | 1,235 | 0 | 1,235 |

Five-class mapping (scenario in `master.csv`, corrected mapping in [`configs/dataset`](../configs/dataset);
see [MAPPING_JUSTIFICATION.md](MAPPING_JUSTIFICATION.md)):

| Split | Real (S4) | Out-of-Context (S1) | Fake image (S3) | Fake/edited text (S2) | Fake text + fake image (S5) |
|---|---:|---:|---:|---:|---:|
| Train | 8,897 | 5,435 | 1,154 | 1,154 | 1,154 |
| Validation | 817 | 817 | 247 | 247 | 247 |
| Test | 247 | 247 | 247 | 247 | 247 |

The fusion results in [FUSION_RESULTS.md](FUSION_RESULTS.md) (0.8348 accuracy, 0.8365 macro-F1) were
measured on the **internal** held-out Test row above. This is **not** an external MMFakeBench benchmark.

## How much of MMFakeBench is already inside the project

Local IDs are `mmfb_<file>_<index>`, where the index is the record's position in the original JSON.
Every one of the 900 + 8,942 present rows matches the Hugging Face record at that index by exact
caption and image-folder name, and the 8,942 test-file images are byte-identical to the project copies.

| Original MMFakeBench source | Records | Train | Validation | Test | Unseen (absent) |
|---|---:|---:|---:|---:|---:|
| `MMFakeBench_val.json` | 1,000 | 672 | 120 | 108 | 100 |
| `MMFakeBench_test.json` | 10,000 | 6,700 | 1,115 | 1,127 | 1,058 |

Consequences:
- **The full MMFakeBench val or test set is NOT external** for our model: about 78-79% of each file is in
  train or validation, and about 11% is in the project Test split.
- The absent rows were mostly dropped by the dataset's class cap: MMFakeBench rows are capped at exactly
  1,648 per scenario for S1/S2/S3/S5 (3,250 for S4). The small remainder (a few rows each in S1/S3/S4)
  has no recorded reason (de-duplication or missing images are possible but unproven).

## External / unseen candidate (MMFakeBench test file)

Manifests (small, tracked), in [`data/external_eval/`](../data/external_eval/):

| File | Rows | Definition |
|---|---:|---|
| `mmfakebench_test_unseen.csv` | 1,058 | Test-file rows absent from the project by original index |
| `mmfakebench_test_unseen_caption_clean.csv` | 1,047 | Same, minus 11 rows whose exact caption appears in project train/validation (none appear in Test) |

Columns: `external_id` (`mmfb_test_<index>`), `mmfb_index`, `source_file`, `text`, `image_path` (original,
e.g. `/fake/fever_AI_test_100/x.png`), `relative_image_path` (under the local MMFakeBench root),
`image_exists`, `text_source`, `image_source`, `gt_answers`, `fake_cls`, `subcategory`, `group`,
`scenario`, `scenario_name`, `overlap_status`, `caption_in_project_splits`, `image_bytes_in_project`.

**Image leakage check.** Comparing SHA-256/SHA-1 of the unseen images with every project image, **67 of
the 1,058 unseen rows have image bytes that already exist in the project** (50 train, 11 validation,
6 test; the `image_bytes_in_project` column records the split). Dropping them from the caption-clean set
leaves **985 rows with both an unseen index, a clean caption and an unseen image**: filter on
`image_bytes_in_project == ""`. Near-duplicate (perceptual) images were not checked.

Class mix of the 1,047 caption-clean rows (exact):

| Scenario | Rows |
|---|---:|
| Fake text + fake image (S5) | 753 |
| Fake/edited text, real image (S2) | 244 |
| Real (S4) | 48 |
| Fake image, real text (S3) | 2 |
| Out-of-context (S1) | 0 |

By `fake_cls`: textual_veracity_distortion 736, mismatch 262, original 48, visual_veracity_distortion 1.
Main subcategories: fever_AI 319, coco_text_edit 181, llm_rewrite 102, rumor_match 96,
gossipcop_midjourney 80, DGM4_text_edit_senti 80, chatgpt_match 68.

**What it can and cannot be used for.** It is dominated by fake-text cases and has no out-of-context
and almost no genuine or fake-image rows, so it is **not balanced and not representative of the five
classes**. It is usable only as a **stress test of text-fake and double-fake detection**, and it is not a
general external benchmark. Its rows come from the same benchmark, generators and subcategories as the
training data, so it is "unseen rows", not "independent source". Report results on it separately from the
internal Test result, and say which filter was applied.

## Machine-specific paths (Training PC)

| Item | Path |
|---|---|
| Team dataset (with images, hash-named) | `D:\MultiGuard\data\fnd_team_dataset_2026-09-14\` |
| Full MMFakeBench test archive | `D:\MultiGuard\data\raw\MMFakeBench\MMFakeBench_test.zip` (6,808,483,932 bytes) |
| Extracted images (image root) | `D:\MultiGuard\data\raw\MMFakeBench\` , 10,000 files under `MMFakeBench_test\{real,fake}\<subcategory>\` |
| MMFakeBench test metadata | `D:\MultiGuard\data\raw\MMFakeBench\MMFakeBench_test.json` |
| Manifests | `data/external_eval/*.csv` in this repository |

`relative_image_path` is relative to the MMFakeBench root above; keep the root in an environment
variable or config rather than in the manifest. `MMFakeBench_val` images are not downloaded (only its
JSON was inspected). The archives and images are not tracked in Git (`data/raw/` and `*.zip` are ignored).
MMFakeBench has its own access terms; do not redistribute its data.
