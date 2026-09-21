# Match vs. mismatch: CLIP or Qwen-VL for the agreement branch

The new pipeline's agreement branch needs one frozen encoder to answer *does
this caption belong to this picture?* Two candidates are plausible and the
choice is not obvious, so this is set up as an experiment rather than an
opinion.

Files: `fnd/models/match_mismatch.py` (the two encoders),
`fnd/extract_match.py` (caches `v_match`), `fnd/probe_match.py` (scores one
backbone in detail), `fnd/compare_match.py` (the head-to-head),
`fnd/tests/test_match_mismatch.py`, `fnd/tests/test_compare_match.py`,
`scripts/run_match_ablation.sh`.

## The two candidates, and what actually separates them

| | `clip` | `qwenvl` |
|---|---|---|
| default | `openai/clip-vit-large-patch14` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| reads the pair | two towers, separately | one decoder, together |
| output | `[img, txt, img*txt, abs(img-txt), cos]` | one pooled hidden state |
| width | `4D + 1` (3073 at D=768) | `hidden_size` (2048 at 3B) |
| cost | minutes for the whole CSV | hours, and needs the GPU |
| zero-shot score | yes, the raw cosine | none |

CLIP summarises each side **before** they meet. That is a real limitation for
this task: an out-of-context pair is built precisely so that both halves are
individually plausible, and a summary of each half separately can miss that
this caption is about *a different* flood. A VLM reads the picture and the
caption in one sequence, so the caption tokens can attend to the image patches
and — in principle — notice that the entity named is not the entity shown.

"In principle" is the part being tested. The VLM is also 20–40x more expensive
per pair, so it has to earn the slot, not merely occupy it.

## What counts as the right answer

The dataset's five scenarios do not split cleanly into "match" and "mismatch",
so the probe reports two targets and quotes both.

**`mismatch_clean` — `ooc` vs `genuine` only.** This is the headline. In both
classes the caption is real prose and the photograph is a real photograph; the
*only* difference is whether they belong together. Nothing else can explain a
gap, which is what makes this the honest measure of an agreement encoder.

**`mismatch_all` — `ooc` vs the other four scenarios.** This is what the branch
meets at inference, and it is harder for a reason that is not the encoder's
fault: a `fake_text_real_image` pair *also* fails to describe its picture, and
this target asks the encoder to put it on the negative side anyway.

The three tampered scenarios are **excluded** from `mismatch_clean` (marked
`-1`), not folded into either side. Guessing a label for them would be
inventing ground truth, which is the failure mode `fnd/data/records.py` was
written to prevent.

The probe also prints the **mean predicted mismatch probability inside each
scenario**. A detector that fires on `ooc` *and* on `fake_text_real_image` is
reading "the caption does not describe this picture", which is the truthful
description of what agreement can see. The fusion stage needs to know that
before it is asked to separate those two scenarios.

## Why the comparison is fair

Everything downstream of the encoder is pinned:

- **Same rows, proven not assumed.** `compare_match` compares the two id lists
  and the split of every row, and stops if they differ. A `--limit` on one
  extraction and not the other would otherwise produce a confident comparison
  of two different test sets.
- **Same head** (`H → 1024 → 512 → 1`, Tanh), same optimiser, same learning
  rate, same early stopping — the head the textual-forensic probe already uses,
  so these numbers are comparable to that stream's too.
- **Standardised per backbone with train statistics only.** The two feature
  spaces differ in scale and width; using test rows for the mean and variance
  would leak.
- **Threshold chosen on val, never on test.**
- **Several seeds.** One head on one seed is a sample of size one. The table
  reports mean ± spread across `--seeds` runs, so a gap smaller than the
  seed-to-seed wobble is visible as one.
- **A paired bootstrap on the test set.** The seed-averaged probabilities of
  the two backbones are resampled on the *same* indices, so the shared
  difficulty of a resample cancels and what is left is the encoder difference.
  A winner is declared only when the 95% interval excludes zero.

Two things are deliberately *not* equalised. The feature widths differ
(3073 vs 2048) because forcing them equal would mean projecting one backbone's
output and comparing an encoder against an encoder-plus-projection. And the
VLM has no cosine, so the zero-shot column is blank for it — CLIP's raw cosine
is reported because if a trained head cannot beat one frozen number, the honest
summary of the stream is that one number.

## Reading the output

`outputs/match_compare/comparison.md` is the table to paste into the report.
Three outcomes, three different conclusions:

- **CLIP wins or ties.** Take CLIP. It is a fraction of the cost and it hands
  the fusion stage a cosine that is interpretable in an error analysis.
- **Qwen-VL wins and the interval excludes zero.** The joint read is buying
  something real. Check the size of the gap against the extraction cost before
  committing: the branch runs once per sample at inference too.
- **Neither is distinguishable.** The script says so instead of crowning the
  higher mean. Then the tie-breaker is cost, latency and memory, and that means
  CLIP.

Look at the raw-cosine line before anything else. If the CLIP head's AUC is
barely above it, the interaction features are not adding much and the whole
branch could be one scalar — useful to know before building fusion around a
3073-d vector.

## Running it

Needs `data/processed/balanced_5group.csv` from `scripts/run_dataset.sh`, and
the images it points at.

```bash
# plumbing only, no downloads, ~1 minute
python -m pytest fnd/tests/test_match_mismatch.py fnd/tests/test_compare_match.py -q

# the real thing
bash scripts/run_match_ablation.sh
```

or step by step:

```bash
CSV=data/processed/balanced_5group.csv

python -m fnd.extract_match --csv $CSV --backbone clip   --out features/v_match_clip.pt
python -m fnd.extract_match --csv $CSV --backbone qwenvl --out features/v_match_qwenvl.pt \
    --batch-size 4                      # the VLM needs a much smaller batch than CLIP

python -m fnd.probe_match   --features features/v_match_clip.pt   --csv $CSV \
    --out outputs/match_probe_clip
python -m fnd.probe_match   --features features/v_match_qwenvl.pt --csv $CSV \
    --out outputs/match_probe_qwenvl

python -m fnd.compare_match --csv $CSV \
    --features clip=features/v_match_clip.pt \
    --features qwenvl=features/v_match_qwenvl.pt \
    --out outputs/match_compare
```

Extract once per backbone. Both are frozen, so the vectors do not change
between runs, and every later experiment — the ablations below, the fusion
stage — reads the cached files.

### Worth running while you are there

```bash
# what each part of the CLIP feature vector is worth
python -m fnd.extract_match --csv $CSV --backbone clip --features sim \
    --out features/v_match_clip_sim.pt
python -m fnd.extract_match --csv $CSV --backbone clip --features concat \
    --out features/v_match_clip_concat.pt
python -m fnd.compare_match --csv $CSV \
    --features cosine_only=features/v_match_clip_sim.pt \
    --features concat=features/v_match_clip_concat.pt \
    --features interaction=features/v_match_clip.pt \
    --out outputs/match_ablation_features

# which layer of the VLM carries the agreement signal, and how to pool it
python -m fnd.extract_match --csv $CSV --backbone qwenvl --layer 18 --batch-size 4 \
    --out features/v_match_qwenvl_l18.pt
python -m fnd.extract_match --csv $CSV --backbone qwenvl --pooling masked_mean \
    --batch-size 4 --out features/v_match_qwenvl_mean.pt
```

The last one is not a formality. With causal attention, the image tokens come
first and never see the caption, so mean pooling averages in positions that
could not possibly have judged the pair; `last` takes the one position that saw
everything. That is why `last` is the default, and the flag exists so the claim
can be checked rather than believed.

## What this does not decide

The branch's encoder, and nothing else. Integrating the winner with the image-
forensic and textual-forensic branches in `models/pipeline_v3.py` is the next
piece of work, and it inherits exactly one thing from here: which file
`v_match` is cached from.
