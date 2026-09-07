# Project walkthrough (plain language)

## Part 1: building the dataset (`fnd/data/`)

**`records.py`, the vocabulary.** Defines the five scenarios by name and number
and the three yes/no facts that define each one: is the caption fake, is the
picture fake, is the pairing wrong. Defines `Sample`, one caption-picture pair
with its label and provenance. Every other file speaks in terms of `Sample`, so
the source datasets, which all label things differently, are converted once
into one shape.

**`mmfakebench.py`, the labeler.** MMFakeBench's `fake_cls` only says which of
its three buckets a pair is in, and one bucket mixes two of our scenarios. This
file reads `fake_cls` and `image_source`; a rule table turns them into our
scenario. Every rule also checks `text_source` is on the expected side. After
labeling: every folder must map to exactly one scenario, and the four bucket
totals must equal the paper's (3300/3300/1100/3300). A record that fits no rule
stops the program and is printed. Nothing is dropped silently.

**`hashing.py`, the fingerprints.** Captions are lowercased and stripped of
punctuation before comparing. Pictures get a checksum of the file bytes (exact
copies) and a difference hash: shrink to 9x8 grey pixels and record which
neighbour is brighter (re-saved or resized copies).

**`build.py`, selection and split.**
1. Selection: per scenario, walk candidates in a fixed shuffled order
   (MMFakeBench first, then top-ups) and keep a pair unless the same caption +
   picture was already kept. Stop at N = smallest scenario (1,650).
2. Split: pairs sharing a caption or a picture are linked; each linked bunch
   goes to one split as a whole, so nothing appears on both sides of the exam.
   Then 70/15/15 per scenario.
3. Write: one CSV row per pair + `manifest.json` with every count and every
   skipped record.

**`verify.py`, the independent examiner.** Reads only the finished CSV and
re-checks: equal scenarios, unique ids, no repeated pair, no caption or picture
in two splits, split proportions, scenario number = group, and (with images)
every file exists. Separate from the builder so a builder bug cannot hide.

## Part 2: the model (`fnd/models/fnd_clip.py`), spec sections 2.1-2.4

Three "eyes", then a decision on how much to trust each eye per pair.
- Picture: ResNet-50 (ImageNet weights, fine-tuned) -> 2,048 numbers.
- Caption: BERT-base-uncased summary token -> 768 numbers.
- Agreement: CLIP image and text encoders (frozen) -> 512 + 512 numbers; their
  cosine similarity scores how well caption fits picture; the joined 1,024
  numbers are multiplied by that score, so a badly matching pair contributes
  little through this eye.
Each eye is squeezed to 256 numbers. An attention layer scores each eye, the
scores become weights summing to 1, the weighted mix goes to a two-layer
classifier with one output: probability the pair is fake. The three weights and
the similarity are returned for analysis.

## Part 3: training and measuring

**`torch_dataset.py`** reads one split, opens each picture, prepares it for
ResNet and for CLIP, tokenizes the caption for BERT and for CLIP.

**`metrics.py`** accuracy, precision, recall, F1, AUC, and accuracy inside each
scenario (which kind of fake is missed).

**`train.py`** each epoch: train on train, score val, keep the best F1
checkpoint, stop after 2 epochs without improvement. Real pairs are drawn 4x
more often during training so each epoch is half real / half fake (the PDF
allows oversampling MMFakeBench). The best checkpoint is scored once on test:
`test_metrics.json` and `test_predictions.csv` (probability, similarity,
attention weights per test pair).

**Tests (`fnd/tests/`)**: 39 tests, no downloads: one per labeling rule and
rejection case, planted duplicates and shared pictures, model shapes, and a
full training loop on tiny synthetic data.

**Scripts**: `run_dataset.sh` from download to VERIFY PASS;
`run_train.sh` from install to `test_metrics.json`.
