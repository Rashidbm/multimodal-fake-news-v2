"""Upload API for the fact-checking branch.

    uvicorn app.factcheck_server:app --host 127.0.0.1 --port 8081

POST /factcheck   multipart form: text, image, optional mode, optional task
GET  /health      process health only (use ``python -m src.factcheck doctor`` for the backends)

Every response carries ``status`` (ok | abstained | error).  Check it before
using ``prediction``.  This service is not connected to the fusion dashboard.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.factcheck import prompts
from src.factcheck.config import Config
from src.factcheck.pipeline import FactChecker

API_MODES = ("closed_book", "search", "assessed")

app = FastAPI(title="Fact-check (local Gemma 4 + SearXNG)",
              description="Research branch. Results include `status`; check it before using `prediction`.")
_checker: FactChecker | None = None


def checker() -> FactChecker:
    global _checker
    if _checker is None:
        _checker = FactChecker(Config.from_env())
    return _checker


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "scope": "process only; run `python -m src.factcheck doctor` for backend readiness"}


@app.post("/factcheck")
def factcheck(text: str = Form(...), image: UploadFile = File(...),
              mode: str = Form("search"), task: str = Form(prompts.DEFAULT_TASK)) -> dict:
    if mode not in API_MODES:
        raise HTTPException(422, f"mode must be one of {API_MODES}")
    if task not in prompts.TASKS:
        raise HTTPException(422, f"task must be one of {sorted(prompts.TASKS)}")
    fc = checker()
    data = image.file.read(fc.cfg.max_image_bytes + 1)
    if len(data) > fc.cfg.max_image_bytes:
        raise HTTPException(413, f"image larger than {fc.cfg.max_image_bytes} bytes")
    suffix = Path(image.filename or "").suffix[:8]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        result = fc.run_example({"id": "api", "text": text, "image": str(path), "task": task}, mode)
    result.pop("model_calls", None)
    for e in result["errors"]:
        e.pop("traceback", None)
    return result
