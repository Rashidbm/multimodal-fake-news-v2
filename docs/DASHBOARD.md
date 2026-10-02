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
| Fusion model | `--fusion dashboard.v3_fusion:load_fusion` | see the Fusion plugin section; any `package.module:factory` works |

The trained weights must be extracted into `models/semantic/` and `models/image/` as described in the
README.

## Per-request inputs and outputs

| Branch | Runs on | Output |
|---|---|---|
| Semantic | image + caption | `v_semantic [1, 768]`, pair fake score, BLIP mismatch score |
| Image | image only | `v_imgfor [1, 768]`, image fake probability |
| Text | caption only | `v_textfor [1, 4096]` (Qwen3.5-9B layer 30, masked mean) |

## Fusion plugin (five-class verdict)

`dashboard/v3_fusion.py` plugs the trained V3 fusion checkpoint (the `best.pt` written by
`fnd.train_fusion`, see [FUSION_RESULTS.md](FUSION_RESULTS.md)) into the dashboard:

```text
python -m dashboard.server --fusion dashboard.v3_fusion:load_fusion
```

| Item | Detail |
|---|---|
| Checkpoint path | `FUSION_CHECKPOINT` environment variable; default `outputs/fusion_v3/best.pt` |
| Inputs | semantic `[1, 768]`, image `[1, 768]`, text `[1, 4096]` |
| Output | softmax of `main_logits`, `[1, 5]`, in the checkpoint's class order (`fnd.train_fusion.CLASSES`) |
| Loading | `weights_only=True` and strict `load_state_dict` into `V3FusionModule`; a missing file raises an error that names `FUSION_CHECKPOINT` |
| Needs | the text branch (do not pass `--no-text`) |

Example (PowerShell): `$env:FUSION_CHECKPOINT = "D:\models\fusion\best.pt"`.

- **With a valid checkpoint**, the dashboard returns the real five-class fused verdict and class
  probabilities, and "overall" risk is `1 - P(genuine)`.
- **Without `--fusion`**, the three branches still run independently and the verdict reads **"Final
  prediction pending"**; no prediction is invented.

The checkpoint is a training artifact and is not tracked in Git. The plug is unit-tested with a
synthetic checkpoint (`fnd/tests/test_dashboard_fusion.py`). A full live end-to-end run of the
dashboard with the real checkpoint and all three branches has **not** been verified in this repository's
documentation yet.

## What works

- The three branches run live on an uploaded image and text and return their scores and feature
  dimensions.
- The page shows the image-branch score, the BLIP mismatch score and, as overall risk, the semantic
  branch's pair fake score.

## What is not wired

- The fusion checkpoint is not shipped with the repository; supply it as described above.
- The text AI-authorship and text-pattern rows stay empty: the text branch yields features only.
