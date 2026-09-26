# Multimodal fake-news detection

The semantic and image branches expose trained **768-dimensional representations**
for the fusion model. Their classifier scores are available for independent branch
evaluation; the fusion inputs are the feature vectors.

| Branch | Input | Feature output | Implementation |
|---|---|---|---|
| Semantic | Image + caption | `[B, 768]`, CPU float32 | `fnd.predict_semantic_fusion.SemanticFusionPredictor` |
| Image | Image pixels only | `[B, 768]`, model device | `fnd.forensic.inference.ImageBranch` |

Read [semantic documentation](docs/SEMANTIC_BRANCH.md) and
[image documentation](docs/IMAGE_BRANCH.md) for training, model files and metrics.

## Setup

```text
git submodule update --init --recursive
python -m pip install -r requirements-semantic.txt -r requirements-forensic.txt
```

On NVIDIA machines, install the appropriate CUDA-enabled PyTorch/torchvision build
first. Python 3.13.11, torch 2.11.0 and torchvision 0.26.0 were verified on macOS.
Windows/CUDA execution is not newly verified here.

**Trained binary files are transferred separately.** Put the files listed in each
`ARTIFACTS.json` beside its `bundle.json` under `models/semantic/` and the selected
`models/image/{news,broad,ai}/` directory. The existing full Windows demo contains
these files; the fusion-only archive does not. Cloning code does not download our
trained weights. Their SHA-256 hashes are checked during inference.

The image loader requires the pinned CLIP-L weights in the Hugging Face cache.
Download them once before offline use:

```python
from transformers import CLIPModel
CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
)
```

## Inputs for the fusion developer

```python
from PIL import Image
from fnd.predict_semantic_fusion import SemanticFusionPredictor
from fnd.forensic.inference import ImageBranch

device = "cuda"  # "cpu" also works; use "mps" on a supported Mac
semantic = SemanticFusionPredictor("models/semantic/bundle.json", device=device)
image_branch = ImageBranch("models/image/news/bundle.json", device=device)

with Image.open("example.jpg") as image:
    image = image.convert("RGB")
    semantic_output = semantic.predict(
        [image], ["The actual caption"], return_semantic=True
    )
    image_output = image_branch([image])

v_semantic = semantic_output["semantic"].to(device)  # [1, 768]
v_image = image_output["features"].to(device)       # [1, 768]
# Join these with the text branch's representation using the same sample_id.
# The fusion model receives the vectors, not the binary branch decisions.
```

Select one image bundle and use it consistently for train, validation and test.
The news bundle above is the existing news-specific extension; it is not the
same model as either standalone RGB/DCT specification baseline. Switching bundles
changes the feature representation and requires regenerating caches and retraining
the fusion head. Preserve sample IDs and the team's fixed data splits.

The corrected mappings are in [configs/dataset](configs/dataset). Do not equate
pair-level fake labels with fake images or AI-written text. The original five
scenario numbers and the V3 output ordering differ; use the explicit translation.

## Evaluation

[Semantic results](reports/semantic) describe the previously evaluated five-scenario
binary test. [Image results](reports/image) distinguish the news, AI-generation,
and broader manipulation variants. These are individual branch results, not final
fusion accuracy.

Run the tests from the repository root:

```text
python -m pytest fnd/tests -q
```
