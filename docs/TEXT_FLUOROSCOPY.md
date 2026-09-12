# Text Fluoroscopy: the textual-forensic stream (guidelines section 4)

Produces `v_textfor`, one of the three vectors the fusion stage reasons over.
Files: `fnd/models/text_fluoroscopy.py` (the module), `fnd/extract_textfor.py`
(the CLI), `fnd/probe_textfor.py` (scores this stream alone),
`fnd/tests/test_text_fluoroscopy.py`, `fnd/tests/test_probe_textfor.py`,
`scripts/run_textfor.sh`.

## What it does, in plain language

Most AI-text detectors work from the outside: measure how surprised a model is
by the words, and turn that into a score. This one works from the inside. A
frozen LLM reads the caption, and instead of asking it for a verdict we read
its internal activations while it processes the text. Machine-written prose
leaves a recognisable trace in those activations that is not obvious in the
words themselves.

The output is deliberately **a vector, not a score**. A detector that returned
"73% AI" would end the conversation there. A vector can be cross-attended
against the image-forensic and semantic vectors, which is what lets the system
tell *AI text with a genuine photo* (scenario `fake_text_real_image`) apart from
*AI text with a generated photo* (`fake_text_fake_image`). Those two scenarios
are the ones this stream is mainly responsible for.

## The five steps

| Step | Operation | Shape |
|---|---|---|
| 4.1 | tokenize (AutoTokenizer, padding + truncation) | `input_ids`, `attention_mask` |
| 4.2 | one frozen forward pass, `output_hidden_states=True` | tuple of `num_layers + 1` |
| 4.3 | take one layer's hidden state | `(B, S, H)` |
| 4.4 | masked mean pooling over the sequence | `(B, H)` |
| 4.5 | `Linear(H, 768)` + GELU | `(B, 768)` |

No training and no text generation anywhere: a single forward pass under
`torch.no_grad()`, not a decoding loop. That is what makes it cheap enough to
run per request when the system is deployed.

**Why the mask in 4.4 is not optional.** Padding positions carry no signal, and
averaging them in shrinks every vector toward zero by an amount that depends on
how long the caption was. That would turn caption length into a feature, which
is exactly the sort of spurious cue that survives training and dies on new data.
`test_pooling_is_length_invariant` encodes the same caption padded to two
different lengths and asserts one identical vector.

**Why the projection in 4.5 exists.** Stage 4 runs multi-head cross-attention
between the three vectors, and attention needs `Q`, `K`, `V` to share one
embedding size. CLIP and UnivFD both emit 768; Qwen3.5-9B emits 4096. The
projection is what makes the three comparable.

## The model: Qwen3.5-9B

| | Qwen3.5-9B | Qwen2-7B | paper's gte-Qwen1.5-7B |
|---|---|---|---|
| layers | **32** | 28 | 32 |
| hidden size | **4096** | 3584 | 4096 |
| layer 30 valid? | **yes** | no | yes |

Switching to Qwen3.5-9B resolves both numbers the guidelines got from the
paper. Layer 30 exists, hidden size really is 4096, and the geometry matches
the paper's encoder exactly — so the paper's layer choice transfers literally
instead of having to be rescaled. `resolve_layer` still validates the index
against whatever model is loaded, so pointing this at a 28-layer model fails
in the first second rather than an hour into a run.

**What the switch costs.** Qwen3.5 is not a plain decoder stack. Its layout is
`8 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))` with
sparse Mixture-of-Experts, so three quarters of its blocks are linear-attention
rather than standard attention. Two consequences:

- **Layer 30 is a Gated DeltaNet block**, not an attention block (every 4th
  layer — 4, 8, ... 32 — is Gated Attention). Whether the forensic signal
  sits in the same place in a hybrid stack as in a pure attention stack is
  not something the paper or anyone else has studied. Layers 28 and 32 are
  attention blocks, so comparing 28 / 30 / 32 is a cheap and genuinely
  novel ablation if time allows.
- **Layer truncation is now off by default.** At layer 30 of 32 it saved
  about 6%, and `test_layer_truncation_keeps_the_same_vector` only proves
  the two paths agree for a standard decoder stack. Enable `--truncate-layers`
  only after checking a truncated run reproduces an untruncated one on this
  model.

**Practical notes for the 4090.** 9B parameters in bf16 is roughly 18 GB of
24 GB, so the default batch size is 4; drop to 2 if it OOMs, raise it if
`nvidia-smi` shows room. Qwen3.5 also needs a recent `transformers` — an
older pin fails with an unknown-model-type error on load, which is the first
thing to check if the smoke test dies immediately.

## One deliberate departure from the guidelines

**Features are cached unprojected.** The guidelines describe shipping
`v_textfor [768]`. An untrained `Linear(4096, 768)` is a *random* projection:
it discards signal arbitrarily and nothing downstream can recover it, because
the weights are frozen inside the cache file. So `extract_textfor.py` caches
the `(N, H)` pooled vectors, and `TextForensicProjection` is exported for the
fusion module to own, so it trains during stage 2. Cost: a larger cache file.
Set this back only with a reason.

## Which layer, and where "30" came from

The method is Yang et al., *Text Fluoroscopy: Detecting LLM-Generated Text
through Intrinsic Features*, EMNLP 2024
([ACL Anthology](https://aclanthology.org/2024.emnlp-main.885/)). Its encoder
is gte-Qwen1.5-7B-instruct, 32 layers, hidden 4096.

The paper does not fix a layer at all. It **selects one per input** as the
layer whose vocabulary-space distribution diverges most from both the first
and last layers:

`M = arg max_j { KL[q_N || q_j] + KL[q_0 || q_j] }`

Its ablation reports that this typically lands on layer 30, and that fixing
it there gives nearly equivalent accuracy at a large speed saving. So "layer
30" is the *empirical result* of that criterion on a 32-layer model — which
is what we now have.

**We use layer 30.** With Qwen3.5-9B matching the paper's depth and width,
this is the paper's own setting rather than a number carried across
architectures. Implementing the KL criterion itself remains the more faithful
option if there is time: it needs the LM head and a vocabulary projection per
candidate layer, and costs speed.

**This still needs the supervisor's sign-off before the full extraction**, and
the chosen layer belongs in the report — not least because layer 30 lands on a
DeltaNet block here.

One further difference from the paper, noted for honesty: it reads the **last
token's** hidden state (gte-Qwen models are trained for last-token pooling),
while the guidelines specify masked mean pooling over the sequence. We follow
the guidelines. Mean pooling is the more robust default for a model not
trained with a dedicated pooling token, but it is a departure and the report
should say so.

## Why the features are cached at all

The model is frozen, so a caption's pooled vector is identical in epoch 1 and epoch
50 — same input, same weights, same output. Recomputing it every epoch would
spend hours re-deriving numbers that cannot change.

| | recompute every epoch | cached |
|---|---|---|
| per epoch (8,250 rows) | ~25 min | ~1 s (load a tensor) |
| 50 epochs | ~20 h | ~2 min |

Identical model, identical accuracy. Caching is only valid *because* the
backbone is frozen; if the model were being fine-tuned the vectors would change
every step and this would be plainly wrong.

## Layer truncation

`truncate_layers` (default **off**, see "The model" above) drops the blocks above
the one being read, saving compute and VRAM proportionally. There is a trap in it worth recording:
transformers builds `hidden_states` as
`[embeddings, out_1, ..., out_{N-1}, norm(out_N)]` — every entry is the raw
block output *except the last*, which has the model's final RMSNorm applied.
Cutting the stack naively makes the layer we read the new last one, so it gets
silently normed and the numbers differ from an untruncated run. The module
replaces `model.norm` with `Identity` when it truncates, and
`test_layer_truncation_keeps_the_same_vector` asserts both paths agree.

The setting must match between cached extraction and live inference, or the
deployed system sees different features than the ones it was trained on.

## Output format

```python
{
  "sample_ids": ["mmfb_0001", ...],      # row order
  "features":   FloatTensor (N, H),      # pooled, float32
  "splits":     ["train", ...],          # copied from the CSV
  "meta":       {...}                    # model, layer, dtype, pooling, n
}
```

`sample_ids` ship with the tensor on purpose. Stage 2 has to line this file up
with `v_semantic` and `v_imgfor`, generated by different people on different
runs. **Join on `sample_id`; do not assume three files share a row order.**

## Running it

```bash
# smoke test: small model, CPU, four rows - proves the code, not the features
python -m fnd.extract_textfor --csv data/processed/balanced_5group.csv \
    --model Qwen/Qwen2-0.5B-Instruct --device cpu --limit 4 --out /tmp/probe.pt

# the real run (RTX 4090)
python -m fnd.extract_textfor --csv data/processed/balanced_5group.csv \
    --out features/v_textfor.pt --limit 200      # hand this to stages 4/5 first
python -m fnd.extract_textfor --csv data/processed/balanced_5group.csv \
    --out features/v_textfor.pt
```

Or `bash scripts/run_textfor.sh`, which installs, runs the tests, does the
200-row subset and then the full extraction.

Qwen3.5-9B is ~18 GB and downloads on first use into `~/.cache/huggingface`; the
repo may be gated, in which case `huggingface-cli login` is needed once on that
machine. Expect roughly 4-8 samples/s in bf16 at batch 4 on a 4090, so about
20-35 min for the 8,250-row build. Drop `--batch-size` to 2 if VRAM runs out. Use `nohup` or
`screen` for the full run so a dropped connection does not kill it.

`features/` and `*.pt` are gitignored: the script travels through git, the
tensors are regenerated on whichever machine has the GPU.

## Scoring this stream on its own

The extractor produces vectors, not predictions, so this stream has no
accuracy until something classifies those vectors. `fnd/probe_textfor.py`
trains a small head on the cached features and reports what Qwen2 alone can
do, before any fusion:

```bash
python -m fnd.probe_textfor --features features/v_textfor.pt \
    --csv data/processed/balanced_5group.csv --out outputs/textfor_probe
```

The head is the paper's: three fully connected layers with Tanh
(H → 1024 → 512 → out). Features are standardised with **train** statistics
only — computing mean and variance over the whole set would leak test
information into the features.

### Which binary label, and why it matters

Three heads train in one run, on the same features and splits:

| Target | Question | Read it as |
|---|---|---|
| `text_fake` | was the **caption** machine-written? | **this stream's score** |
| `label_binary` | is the **post** fake? | the gap, not a grade |
| 5-class | which of the five scenarios? | what fusion has to beat |

The distinction is easy to miss and changes the conclusion. `label_binary`
means "genuine vs everything else", so it is 1 for an out-of-context pair and
for a real caption with a tampered image — both of which have **genuine
human-written text**. the model cannot see an image or a mismatched pairing, so
scoring this stream against `label_binary` asks it to call real human writing
"fake" in two of the five scenarios. Working features would look broken.

`text_fake` is 1 only for `fake_text_real_image` and `fake_text_fake_image` —
the two scenarios whose captions are actually machine-written. That is the
number that says whether this stream works.

`label_binary` is still reported, because the gap between the two is the
honest statement of what one stream can and cannot contribute, and it is the
argument for why fusion is needed at all.

### What `text_fake` actually contains

`text_fake=1` is not one phenomenon. Reading `FOLDER_TO_GROUP` in
`fnd/data/mmfakebench.py`, MMFakeBench builds it from three different things:

| Source folders | The text is |
|---|---|
| `chatgpt_match`, `fever_AI`, `llm_rewrite`, `llm_*_generation` | AI-generated |
| `rumor_match`, `politicat_match`, `gossipcop_match` | human-written rumours |
| `DGM4_text_edit_senti`, `coco_text_edit` | algorithmically edited (word swaps) |

Text Fluoroscopy detects **machine generation**. A human-written rumour from
PolitiFact carries no generation fingerprint, so this stream should separate
the first group well and struggle on the second — however well it is
implemented. The guidelines name this class "AI-Text", but the data under it
is broader than that.

The probe therefore reports `text_fake` accuracy broken down by source
sub-category. That is what turns "the stream scores X" into a statement about
what it actually detects, and it is the most defensible finding this stream
can contribute: *this method generalises to machine-generated captions and
not to human rumours* is a real result, not a disappointing number.

`ooc` is not affected by any of this. All three of its sources are
NewsCLIPpings, which builds mismatched pairs from a genuine human-written
caption and another article's image — nothing is written or edited, only the
pairing is wrong, so `text_fake=0` is correct for it.

### Class balance

The five scenario groups are equal in size, but they do not divide evenly by
any binary question:

| Target | Positive : negative | Why |
|---|---|---|
| `text_fake` | 40 / 60 | 2 groups have machine captions, 3 have human ones |
| `label_binary` | 20 / 80 | only `genuine` is real |
| 5-class | 20 each | equal by construction |

The loss is re-weighted by inverse class frequency (`pos_weight` for the
binary heads, per-class weights for the 5-class head); `--no-class-weight`
turns it off. Deleting rows to force a 50/50 split would be the wrong fix
twice over: it throws away real examples, and because the same CSV feeds all
three streams, a subset here tears the row set away from `v_semantic` and
`v_imgfor` at fusion time.

AUC is the figure to quote when the prior is in question — it is computed
from the ranking and does not move with class proportions at all.

Reported for each: accuracy / precision / recall / F1 / AUC (binary),
accuracy and macro-F1 and F1 per class (5-class), a confusion matrix, and
`text_fake` accuracy inside each scenario. Every score sits beside a
majority-class and a random baseline on the same test split, because an
accuracy figure cannot be judged without knowing what guessing would score.
Written to `metrics.json`, `predictions.csv` and `report.txt`.

Two things this is for. It answers "does this stream carry signal at all?"
before anyone builds fusion on top of it — if the probe is at chance, the
problem is here, not in the fusion module. And it gives the report its
ablation row: the fused system has to beat each stream alone, or the fusion
is not earning its complexity.

It is a floor, not a ceiling. A small head on frozen features will score
below the fused system; that is the expected result, not a failure.
