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
[Fusion and dashboard](docs/Qwen3VL_FUSION.md) covers the CLIP-L vs Qwen3-VL-Embedding
comparison, fusion training and results, and the web dashboard.

## Setup

```text
git submodule update --init --recursive
python -m pip install -r requirements-semantic.txt -r requirements-forensic.txt
```

On NVIDIA machines, install the appropriate CUDA-enabled PyTorch/torchvision build
first. Python 3.13.11, torch 2.11.0 and torchvision 0.26.0 were verified on macOS.
Windows/CUDA execution is not newly verified here.

**Download the trained weights from the [model release](https://github.com/Rashidbm/multimodal-fake-news-v2/releases/tag/semantic-image-models-2026-09-26).**
Download `semantic-weights.zip` and `image-weights.zip`, plus their `.sha256` files.
The archives contain `models/semantic/` and `models/image/{news,broad,ai}/` and can
be extracted directly into this repository. Cloning code does not download the
weights. No retraining is needed for inference or feature extraction.

On Windows, place the downloads in the repository root. Verify each archive's
SHA-256 with `Get-FileHash` against its accompanying `.sha256` file, then run:

```powershell
Expand-Archive .\semantic-weights.zip -DestinationPath . -Force
Expand-Archive .\image-weights.zip -DestinationPath . -Force
```

Each bundle's `ARTIFACTS.json` lists its trained files; inference also checks
their hashes. The release contains neither dataset images nor the text model or
a trained multimodal fusion checkpoint.

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
