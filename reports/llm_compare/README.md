# Text branch: other LLMs vs Qwen3.5-9B (target `text_fake`)

Same data (`dataset_2026-09-14/data/text_samples.csv`, 16,509 unique captions, joined by
`text_asset_id`), same probe code (`fnd/probe_textfor_layers.py`, which reuses the MLP head,
training loop and val-chosen threshold of `fnd/probe_textfor.py`), seeds 42, 0, 1, 2, 3.
Features: frozen instruct model, bf16, raw caption (no chat template), masked mean, max_len 64.

| Model | Layer | Test AUC | Test F1 | Test bal-acc | Test acc |
|---|---|---|---|---|---|
| Qwen3.5-9B (instruct) | 30 (fixed, fusion layer) | 0.946 ± 0.006 | 0.814 ± 0.028 | 0.845 ± 0.024 | 0.839 ± 0.031 |
| Ministral-3-8B-Instruct-2512 (BF16) | 24 (max val AUC) | **0.951 ± 0.004** | **0.841 ± 0.011** | **0.867 ± 0.010** | **0.876 ± 0.009** |

Per seed (Ministral − Qwen), F1: +0.046, +0.048, +0.058, +0.000, −0.016.
Ministral is ahead on average and more stable, but wins only 3 of 5 seeds: suggestive, not conclusive.

Notes
- Ministral layer chosen on val only (`ministral3_8b_instruct/val_layers.csv`); layers 9–31 all
  reach val AUC 0.954–0.963, so the result does not hinge on picking exactly layer 24.
- Ministral BOS token excluded from the mean (Qwen adds none); offload: 12 GiB on GPU, rest in CPU RAM.
- Qwen L30 through this probe gives the recorded AUC (0.945) but lower F1 than the earlier
  multi-layer script (0.847 ± 0.005, not in this repo). The probe's val-chosen threshold lands at
  0.20–0.25 for Qwen (precision 0.76), so F1 is probe-sensitive; compare models only within one probe.
- Features: `D:/fnd_features/ministral3_8b_instruct/` (35 layers, float16), Qwen: `features/v_textfor.pt`.

## Fusion: Qwen L30 vs Ministral L24 text (VAL, 3 seeds)

`fnd/train_fusion.py` unchanged; semantic/image columns from the release `fusion-training-features.npz`
(sha256 ab1cb0d3…); text column swapped by `scripts/swap_text_features.py` (join check: cosine 1.000000).
Runs in `outputs/fusion_compare/`. Val also selects the checkpoint, so both are slightly optimistic.

| Val, 5-class | Qwen L30 | Ministral L24 | per seed (42, 0, 1) |
|---|---|---|---|
| Macro-F1 | 0.835 ± 0.001 | **0.841 ± 0.005** | +0.009 +0.008 +0.001 |
| Accuracy | 0.817 ± 0.002 | **0.823 ± 0.002** | +0.008 +0.008 +0.001 |
| Binary accuracy | 0.853 ± 0.003 | **0.859 ± 0.001** | +0.008 +0.009 +0.002 |
| F1 fake text + real image | 0.871 | **0.885** | +0.015 +0.020 +0.005 |

Ministral wins every seed, by a small margin, mostly on the text-driven scenario. Test pending
(`test_features.npz` is not in the release).

## Fusion TEST (scored once per checkpoint, `scripts/score_fusion_test.py`)

Test file `features/test_features.npz` (sha256 3cf8542f…, 1,235 rows, 247 per scenario); Ministral copy
built by `scripts/swap_text_features.py` (join check vs `features/v_textfor.pt`: cosine 1.000000 on all rows).
Baseline: retrained qwen_L30_s42 gives acc 0.8356 / macro-F1 0.8385 vs reference 0.8348 / 0.8365.

| Test, 5-class | Qwen L30 | Ministral L24 | per seed (42, 0, 1) | mean gain |
|---|---|---|---|---|
| Accuracy | **0.833 ± 0.005** | 0.831 ± 0.008 | −0.009 +0.003 −0.002 | −0.002 |
| Macro-F1 | **0.836 ± 0.005** | 0.833 ± 0.008 | −0.008 +0.003 −0.002 | −0.002 |
| Binary accuracy | **0.893 ± 0.003** | 0.892 ± 0.009 | −0.005 +0.007 −0.004 | −0.001 |
| F1 fake text + real image | 0.871 ± 0.016 | **0.875 ± 0.015** | −0.015 +0.031 −0.004 | +0.004 |
| F1 fake text + fake image | **0.872 ± 0.011** | 0.862 ± 0.005 | −0.016 −0.012 −0.003 | −0.010 |
| F1 real text + fake image | **0.850 ± 0.014** | 0.844 ± 0.002 | −0.007 −0.018 +0.008 | −0.006 |
| F1 genuine | 0.769 ± 0.006 | **0.770 ± 0.014** | −0.003 +0.010 −0.003 | +0.002 |
| F1 out-of-context | **0.818 ± 0.012** | 0.816 ± 0.014 | −0.001 +0.004 −0.007 | −0.002 |

Conclusion: on test, Ministral L24 is a tie with Qwen L30 (Ministral loses 2 of 3 seeds; all gaps are
within seed noise, and ±0.011 is the sampling error of accuracy on 1,235 rows). The val gain did not
transfer. Keep Qwen L30 in fusion; Ministral is a valid "different family, same result" comparison.
