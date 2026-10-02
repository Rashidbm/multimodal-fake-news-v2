# Dashboard

`dashboard/` is a FastAPI web backend that runs the semantic, image and text branches live behind
the semester-1 web interface. The CLIP-L semantic model is used; Qwen3-VL-Embedding is not involved.

## Start

```text
pip install -r requirements-semantic.txt -r requirements-forensic.txt -r requirements-dashboard.txt
python -m dashboard.server
```

Open http://127.0.0.1:8000.

`requirements-dashboard.txt` (fastapi, uvicorn, python-multipart) is optional and only needed here.

## Configuration

| Setting | How | Notes |
|---|---|---|
| Text model | `MULTIGUARD_TEXT_MODEL` or `--text-model` | Qwen3.5-9B local directory or HF id. No default; required unless `--no-text`. Example (Training PC): `$env:MULTIGUARD_TEXT_MODEL = "D:\models\Qwen3.5-9B"` |
| Model caches | `HF_HOME` | The server runs offline (`HF_HUB_OFFLINE=1`) and never downloads models |
| Semantic bundle | `--semantic-bundle` | default `models/semantic/bundle.json` |
| Image bundle | `--image-bundle` | default `models/image/news/bundle.json` (also `broad`, `ai`) |
| Devices | `--branch-device`, `--text-device` | semantic/image on the first GPU, text on the second if two GPUs exist |
| Text layer / length | `--text-layer 30`, `--text-max-len 64` | must match the cached training features |
| Skip text | `--no-text` | the fusion then cannot run |
| Fusion model | `--fusion package.module:factory` | see the `dashboard/server.py` docstring |

The trained weights must be extracted into `models/semantic/` and `models/image/` as described in the
README.

## Per-request inputs and outputs

| Branch | Runs on | Output |
|---|---|---|
| Semantic | image + caption | `v_semantic [1, 768]`, pair fake score, BLIP mismatch score |
| Image | image only | `v_imgfor [1, 768]`, image fake probability |
| Text | caption only | `v_textfor [1, 4096]` (Qwen3.5-9B layer 30, masked mean) |

## What works

- The three branches run live on an uploaded image and text and return their scores and feature
  dimensions.
- The page shows the image-branch score, the BLIP mismatch score and, as overall risk, the semantic
  branch's pair fake score.
- Fusion hook: a factory passed with `--fusion` is run on the three vectors and its five-class
  probabilities are shown.

## What is not wired

- **No trained fusion checkpoint is connected.** Without `--fusion`, the verdict reads **"Final
  prediction pending"**, the class probabilities stay empty and `final_available` is false. No
  prediction is invented. The checkpoint from [FUSION_RESULTS.md](FUSION_RESULTS.md) still needs a
  runtime factory.
- The text AI-authorship and text-pattern rows stay empty: the text branch yields features only.
- End-to-end inference through the fused model is therefore not yet available.
