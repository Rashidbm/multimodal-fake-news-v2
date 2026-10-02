# Dashboard

`dashboard/` is a FastAPI web backend that runs the semantic, image and text branches live behind
the semester-1 web interface. The CLIP-L semantic model is used; Qwen3-VL-Embedding is not involved.

## Quick Start — Training PC demo

Verified on the Training PC (2 x RTX 4090, system Python 3.13.7, no venv; dependencies are installed
globally and `HF_HOME` is already set to `D:\hf_cache` at user level).

```powershell
cd C:\Users\497-MultiGuard\Desktop\Multiguard_windows_demo\multimodal-fake-news-v2

$env:FUSION_CHECKPOINT   = "D:\MultiGuard\fusion\outputs\v3_clip_semantic\best.pt"
$env:MULTIGUARD_TEXT_MODEL = "D:\models\Qwen3.5-9B"

python -m dashboard.server --fusion dashboard.v3_fusion:load_fusion
```

Wait for `Open http://127.0.0.1:8000` (about 20 s to load), then open **http://127.0.0.1:8000**
(`--host` / `--port` change it).

**Confirm the fusion is active** in the console before the "ready" line:

```text
fusion checkpoint D:\MultiGuard\fusion\outputs\v3_clip_semantic\best.pt (epoch 8, val macro-F1 0.8428)
```

or open http://127.0.0.1:8000/api/health and check `"final_available": true`.

**Try a sample:** paste a caption into the text box, upload an image (samples are in
`dashboard/static/images/`), click **Analyze Article**. Expected: a verdict card with one of five
classes (Real, Out-of-Context, Manipulated, Fake/Edited Text, Fully Fabricated), the class
probabilities, and the module scores (image manipulation, cross-modal mismatch, overall risk).
Takes about 1-2 s per request after loading. If `final_available` is false the page shows
"Final prediction pending" instead of a verdict.

First-time setup only, if packages are missing:
`pip install -r requirements-semantic.txt -r requirements-forensic.txt -r requirements-dashboard.txt`
(the model weights must be in `models/semantic/` and `models/image/`, see the README).

### Required models / checkpoints

| Component | Path / source | How configured |
|---|---|---|
| Fusion checkpoint | `D:\MultiGuard\fusion\outputs\v3_clip_semantic\best.pt` | `FUSION_CHECKPOINT` (default `outputs/fusion_v3/best.pt`, which does not exist) |
| Text branch (Qwen3.5-9B) | `D:\models\Qwen3.5-9B` | `MULTIGUARD_TEXT_MODEL` or `--text-model`; required, no default |
| Semantic branch (CLIP-L) | `models/semantic/bundle.json` in the repo | `--semantic-bundle` (automatic) |
| Image branch | `models/image/news/bundle.json` in the repo | `--image-bundle` (automatic) |
| HF caches (BLIP, CLIP) | `D:\hf_cache` | `HF_HOME`; the server runs offline and never downloads |

The text model is Qwen3.5-9B (text forensics). The Qwen3-VL-Embedding semantic experiment is not
used and not required.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Set MULTIGUARD_TEXT_MODEL (or pass --text-model)...` | Set the text-model variable, or pass `--no-text` (the fusion then cannot run) |
| `FileNotFoundError: fusion checkpoint ... not found: set FUSION_CHECKPOINT` | The variable is unset in this PowerShell window (it is per-session) or the path is wrong |
| `Error(s) in loading state_dict` / missing keys | Wrong checkpoint format. `MultiGuard_FUSION_READY\outputs\fusion_meeting\best.pt` is an older format and does not load; use the file above |
| `A model is missing from ...hub` | `HF_HOME` not set to `D:\hf_cache` (`$env:HF_HOME = "D:\hf_cache"`) or the cache is incomplete |
| `ModuleNotFoundError: fastapi / multipart` | Install `requirements-dashboard.txt` |
| `address already in use` | Another server holds port 8000: stop it or pass `--port 8001` |

## Generic start (without fusion)

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
