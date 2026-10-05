# Local fact-checking: run this first

**Start with Gemma 4 31B FP8 on the two RTX 4090s, using local SearXNG for web search.** The user submits text and an image. The model generates search queries, reads retrieved evidence, and returns a verdict with quotations and source links. No paid API key is required for this configuration. Search and the initial model download require internet access.

This is a separate fact-checking branch. The existing semantic/forensic fusion is unchanged. No new accuracy result has been measured yet.

## GPU PC setup

Use Linux, or WSL2 with working NVIDIA Docker support, Docker Compose with `gpus` support, Python 3.10+, and both 24 GB GPUs available. Allow substantial disk space for the model download and containers. Run from the repository root:

```bash
python3 -m venv .venv-factcheck
source .venv-factcheck/bin/activate
pip install -r requirements-factcheck.txt
bash scripts/start_factcheck_local.sh
source deploy/factcheck/local.env.example
```

The first start downloads weights. Follow progress:

```bash
docker compose -f deploy/factcheck/compose.yaml logs -f vlm
```

After the server finishes loading:

```bash
python -m src.factcheck doctor
```

The model revision, chat template and container image digests are pinned. Startup also records resolved image IDs in `outputs/factcheck/container-images.json`; preserve that file with experiment results.

The FP8 configuration is a **deployment candidate**, not a verified two-4090 deployment. Its model weights are compressed to reduce memory; vision and some other layers remain at their original precision. Two 24 GB cards do not behave as one 48 GB card: tensor parallelism splits supported layers across them. Initial settings use one simultaneous request and a 32K context. If startup reports insufficient KV-cache memory, try `--max-model-len 16384` in the Compose command and reduce the pipeline's evidence budget before interpreting comparisons. Do not silently change settings between runs. GPU startup, tool parsing and available memory must pass the smoke test on the actual PC.

## First real test

Prepare manifests on the GPU PC, where the dataset images reside. The locally prepared manifests have absolute Mac paths and should not be copied as-is.

```bash
python -m src.factcheck prepare-mmfake --split val --balanced --limit 20 --output outputs/factcheck/mmfake-val-20.jsonl
bash scripts/run_factcheck_pilot.sh outputs/factcheck/mmfake-val-20.jsonl outputs/factcheck/gemma4-val-smoke
```

Fix execution failures using validation data. Then freeze configuration and run the test pilot:

```bash
python -m src.factcheck prepare-mmfake --split test --balanced --limit 200 --output outputs/factcheck/mmfake-test-200.jsonl
bash scripts/run_factcheck_pilot.sh outputs/factcheck/mmfake-test-200.jsonl outputs/factcheck/gemma4-test-pilot
```

These commands use the cached MMFakeBench Arrow files and extracted images already present in this project. `prepare-mmfake --root PATH` can point to another location. Reuse existing prepared manifests instead of overwriting them. Repeating the pilot script resumes existing runs with the identical input/configuration; errors remain recorded. To retry failures after changing code or configuration, use a fresh output directory.

## What the comparisons answer

| Run | What happens | What it tests |
|---|---|---|
| closed_book | Original text and image → verdict | Starting performance without retrieval |
| search | Model generates queries → searches → reads sources → verdict | Does retrieval improve the complete system? |
| direct | Saved search evidence → fresh verdict | Control for the assessment experiment |
| assessed | Same saved evidence → relevance/support assessment → fresh verdict | Does the extra assessment step actually help? |

`comparison.md` summarizes matched results. Each run includes `metrics.json`, `report.md`, its configuration and per-example JSON containing predictions, evidence, queries, errors and citations. Errors and abstentions count against overall accuracy. Use accuracy, macro-F1, class results, error counts and latency together. Replay latency excludes evidence collection. The balanced 200-example pilot is not the official full-test distribution and must not be compared directly with papers' full-test results.

Do not tune on the test pilot repeatedly. After validation choices, run the full test (`--limit 0`, without `--balanced`) and examine event/source duplicates when estimating uncertainty. Search can find fact-check pages or dataset-related pages that expose labels; a live-web pilot is not a leakage-controlled historical evaluation. Use `--exclude-domain`, per-example `excluded_urls`, and `cutoff_date` where the protocol requires them. A publication-date filter cannot prove that an old page was not subsequently updated; historical evaluation needs archived evidence.

## One claim or API

```bash
python -m src.factcheck run --text 'Your exact claim' --image /path/to/image.jpg --output outputs/factcheck/example.json
uvicorn app.factcheck_server:app --host 127.0.0.1 --port 8081
```

Open `http://127.0.0.1:8081/docs` for the upload API. POST `/factcheck` accepts `text`, `image`, optional `mode`, and optional `task`. Default task labels are supported, refuted, conflicting, insufficient. MMFakeBench evaluation uses its separate four-way distortion labels. API results include `status`; callers must check it before using a prediction. API health checks process health only; CLI `doctor` checks backend readiness. No connection to the existing fusion is made yet.

## Why this candidate

The review found Gemma 4 31B + search at **87.2% three-bin accuracy** versus **77.2% without search** in VeriTaS v2's Q1 2026 experiment. That supports testing this local model with retrieval. It does **not** establish our FP8 weights, SearXNG retrieval, prompts or MMFakeBench setup at 87.2%. The evidence assessment step is a separate hypothesis, informed by VILLAIN's ablation, and must earn its place through the direct/assessed comparison.

The review's 90.5% Gemini result is from a different hosted model/setup; it is not an accuracy guarantee or a broadly standardized multimodal leaderboard win. See the [full literature review](FACT_CHECKING_LITERATURE_REVIEW_2026-10-05.md) for source tables, benchmark definitions and qualifications.

Primary deployment sources: [Red Hat FP8 checkpoint](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-dynamic), [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html), [SearXNG search API](https://docs.searxng.org/dev/search_api.html).

## Present limits

- This implementation has text web search and source extraction. It does **not yet implement reverse-image search or Google Fact Check retrieval**. Web queries about image content are not reverse-image search.
- SearXNG is free software, but upstream engines can block or rate-limit requests. Failures are recorded; changing backend changes the experiment.
- Retrieval retains up to ten results per query and attempts three HTML pages. It uses deterministic opening excerpts, not a trained passage selector. Useful evidence later in a page may be missed. PDF extraction is not implemented.
- Citation IDs and exact quotations are checked. Whether a quotation truly supports the conclusion still needs a separate audit. A successful request is not proof of a correct fact check.
- The model can still invent context or misread evidence despite safeguards. Synthetic-image detection remains the existing forensic branch's job; this research branch should not replace it.
- No Jev, distillation, early-exit shortcut, or fusion training is included. Measure the baseline first; distillation requires a successful teacher and separate evaluation.
- Offline software tests use fake model/search responses. They do not validate GPU compatibility, live retrieval quality or scientific accuracy.

Run software tests with `python -m pytest tests/factcheck -q`.

## Implementation map

| Path | Role |
|---|---|
| `src/factcheck/pipeline.py` | The four modes, citation checking, error recording |
| `src/factcheck/retrieval.py` | SearXNG client, domain/URL/date filters, page fetch and HTML text extraction, evidence items `E1..En` |
| `src/factcheck/vlm.py` | OpenAI-compatible client for vLLM: image as data URL, forced tool call, JSON-schema output |
| `src/factcheck/prompts.py` | Prompts and label sets (`claim`, `mmfakebench`); `PROMPT_VERSION` is part of the config hash |
| `src/factcheck/evaluate.py`, `metrics.py` | Resumable manifest runs, `metrics.json`, `report.md`, `comparison.md` |
| `src/factcheck/mmfake.py` | `prepare-mmfake` manifests (writes `<manifest>.meta.json` beside each) |
| `src/factcheck/doctor.py` | Backend readiness checks |
| `app/factcheck_server.py` | Upload API |
| `deploy/factcheck/` | Compose file, SearXNG settings, client environment |
| `scripts/start_factcheck_local.sh`, `scripts/run_factcheck_pilot.sh` | Start backends; run the four comparisons and the comparison report |

**Model revision pin.** The Compose file serves the model at `FACTCHECK_MODEL_REVISION`. On first start, `start_factcheck_local.sh` resolves the current Hugging Face commit of `RedHatAI/gemma-4-31B-it-FP8-dynamic`, writes it to `deploy/factcheck/model.lock`, and reuses that file afterwards. Commit `model.lock` so every run and machine serves the same weights. Image digests are pinned in `compose.yaml`: `vllm/vllm-openai:v0.30.0` and `searxng/searxng:2026.10.2-19ffbcd30`. The chat template is the Gemma 4 tool template shipped inside the pinned vLLM image. vLLM listens on `127.0.0.1:8010`, so it does not collide with the dashboard on port 8000. SearXNG listens on `127.0.0.1:8888`.

**Statuses.** `ok` means a verdict was accepted. `abstained` means a verdict was produced but rejected by a safeguard: for the default task, a supported, refuted or conflicting verdict needs at least one valid citation when evidence was provided. `error` means a stage failed; the failing stage and message are in `errors`. Non-fatal problems, such as a failed page fetch or one failed query, appear in `errors` on an `ok` result.

**Resume rules.** A run directory stores its configuration hash. Re-running with a different configuration, prompt version or manifest is refused. An example file is reused when its input hash matches. Replay modes (`direct`, `assessed`) read `<out>/search/examples/*.json` and never search again. A search example that failed is recorded as a replay error, so matched comparisons stay matched.
