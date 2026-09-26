# Image branch

## Input and output

`ImageBranch` takes a list of PIL images. Only their pixels enter the encoders;
caption text, labels, scenario IDs and file metadata are not features.

```python
from fnd.forensic.inference import ImageBranch
model = ImageBranch("models/image/news/bundle.json", device="cuda")
result = model(images)
v_image = result["features"]  # [batch_size, 768]
fake_scores = result["probability"]  # [batch_size], independent diagnostic
```

The returned feature tensor is on the configured device. It is the input intended
for fusion. A standalone binary decision is `fake_scores >= model.config['threshold']`.
Label 0 means authentic; label 1 includes image editing and generation.

## Available trained representations

| Bundle | Encoders | Raw features | Threshold |
|---|---|---:|---:|
| `models/image/news/bundle.json` | CLIP-L/14 + RGB ResNet-50 + DCT ResNet-50 | 4,864 | 0.4267422370663268 |
| `models/image/broad/bundle.json` | Same encoder types; different trained classifier/projection | 4,864 | 0.32796830298574464 |
| `models/image/ai/bundle.json` | RGB ResNet-50 with JPEG-robust training | 2,048 | 0.938739974857322 |

Each exports 768 features through a training-fitted projection that preserves its
classifier direction. These representations are not interchangeable just because
their dimensions match. Train fusion using the exact bundle chosen for inference.

The news and broad variants are additional project experiments. The original
image specification's two baselines are separate RGB and DCT ResNet classifiers
fine-tuned on GenImage. Their clean GenImage accuracy at threshold 0.5 was 92.49%
and 86.15%, respectively. The exported AI variant is `rgb_robust`, not `rgb_guide`.

## Preprocessing and training

- RGB: resize the full image to 224x224, normalize, ResNet-50, global average pool,
  2,048 features. The guide starts from the author's Fourier-mask pretrained model;
  early layers stay frozen while later stages and the classification head train.
- DCT: resize to 224x224, extract Y luminance, perform block DCT independently at
  8x8 and 16x16, log-scale and reassemble, average both maps, then normalize using
  fixed training-set statistics. A one-channel ResNet-50 yields 2,048 features.
  This network starts from ImageNet weights with a new first convolution. Layers
  1/2 are frozen for five epochs; all layers train from epoch six.
- CLIP-L/14: its frozen image encoder yields a normalized 768-dimensional vector.
  No CLIP text encoder, BERT, or BLIP input is used in the image branch.

The news classifier learns from corrected news-image labels on frozen features;
the broad classifier additionally uses DGM4 examples. A fake caption does not make
an authentic image fake: S1/S2/S4 have image label 0; S3/S5 have image label 1.

Training and evaluation entry points live in `fnd/forensic/`: `preprocess`, `train`,
`cache`, `probe`, `robust_probe`, `select`, `evaluate_locked`, `export_selected`,
and `verify_release`. These original experiment pipelines require their recorded
datasets, checkpoints and caches; this code checkout does not contain those data.

The required upstream implementation is a pinned Git submodule at
`external/FakeImageDetection` (commit `55b7142375ae874ee4940031660183763919cfd8`).
Initialize submodules before use. RGB retraining also needs the original author
checkpoint at `external/FakeImageDetection/checkpoints/mask_15/rn50ft_spectralmask.pth`.
That is the author's legacy filename for the Fourier-mask checkpoint. Inference
uses our supplied trained bundle files and does not need that initialization file.

## Trained files

The matching `ARTIFACTS.json` lists exact filenames, sizes and hashes. The news
and broad directories require `rgb_robust.pt`, `dct_guide.pt` and `head.pt`; the AI
directory requires `rgb_robust.pt` and `head.pt`. Head files contain the trained
classifier and feature projection. Keep each head with its own bundle.

These files are available locally in the existing full Windows demo under
`outputs/image_branch_2026_09_14/selected_news`, `selected`, and `selected_ai`.
They are also available as `image-weights.zip` in the
[model release](https://github.com/Rashidbm/multimodal-fake-news-v2/releases/tag/semantic-image-models-2026-09-26).
Extract it into the repository root to populate all three `models/image/` variants.
The weights are release assets, not Git source files. CLIP-L must also be downloaded into the model cache;
see the root README. DCT normalization statistics are embedded in each bundle.

```text
python -m fnd.forensic.inference --bundle models/image/news/bundle.json --image /path/to/image.jpg --device cuda --features-out image_features.pt
```

## Measured results and limits

| Variant and test | Accuracy | Balanced accuracy | Authentic recall | Fake recall |
|---|---:|---:|---:|---:|
| News variant / corrected news images | 97.42% | 97.53% | 96.89% | 98.18% |
| AI variant / GenImage subset | 91.01% | 91.01% | 96.23% | 85.80% |
| Broad variant / DGM4 edits | 61.16% | 61.16% | 56.11% | 66.20% |

These rows are different selected variants, not one model across all datasets.
The [machine-readable results](../reports/image/selected_metrics.json) also contain
the broad model's results on other datasets and corruption conditions.

GenImage evaluation used an 8,000-image test subset with all eight generator types
seen in training; it is not an unseen-generator benchmark. The news test contains
1,201 images and has previous project evaluation history. DGM4 contains 999 test
images and exposes substantial difficulty detecting local face edits. Do not
interpret the news score as universal manipulation-detection performance.
