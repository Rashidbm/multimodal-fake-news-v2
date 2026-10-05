"""Run configuration for the fact-checking branch.

Everything that can change a prediction lives in ``Config`` and is hashed into
``config_sha256``.  Service locations (URLs) are excluded from the hash: moving
the same server to another port is not a different experiment.  Changing any
other field is, and ``evaluate`` refuses to resume a run directory whose saved
configuration differs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field

from .prompts import PROMPT_VERSION

DEFAULT_VLM_URL = "http://127.0.0.1:8010/v1"
DEFAULT_SEARCH_URL = "http://127.0.0.1:8888"
DEFAULT_MODEL = "gemma-4-31b-it-fp8"

MODES = ("closed_book", "search", "direct", "assessed")
REPLAY_MODES = ("direct", "assessed")

# Fields that say where a service runs, not how the experiment behaves.
_LOCATION_FIELDS = ("vlm_url", "search_url")


@dataclass(frozen=True)
class Config:
    vlm_url: str = DEFAULT_VLM_URL
    search_url: str = DEFAULT_SEARCH_URL
    model: str = DEFAULT_MODEL

    # Model sampling.  Greedy decoding with a fixed seed; one request at a time.
    temperature: float = 0.0
    seed: int = 0
    max_output_tokens: int = 1536
    enable_thinking: bool = False

    # Retrieval.
    max_queries: int = 3
    results_per_query: int = 10
    pages_to_fetch: int = 3
    search_language: str = "en"
    snippet_chars: int = 500
    page_excerpt_chars: int = 3000
    evidence_budget_chars: int = 16000
    max_page_bytes: int = 2_000_000
    exclude_domains: tuple[str, ...] = ()
    cutoff_date: str | None = None          # YYYY-MM-DD; per-example value wins

    # Safeguards.
    require_citation: bool = True           # claim task: supported/refuted/conflicting need a valid citation
    max_image_bytes: int = 20_000_000

    # Timeouts (seconds).
    vlm_timeout: float = 600.0
    search_timeout: float = 20.0
    fetch_timeout: float = 15.0

    prompt_version: str = PROMPT_VERSION
    user_agent: str = "multimodal-fake-news-factcheck/1 (research; local)"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        base = dict(
            vlm_url=os.environ.get("FACTCHECK_VLM_URL", DEFAULT_VLM_URL),
            search_url=os.environ.get("FACTCHECK_SEARCH_URL", DEFAULT_SEARCH_URL),
            model=os.environ.get("FACTCHECK_MODEL", DEFAULT_MODEL),
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        if "exclude_domains" in base:
            base["exclude_domains"] = tuple(sorted({d.lower().strip(".").removeprefix("www.")
                                                    for d in base["exclude_domains"]}))
        return cls(**base)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["exclude_domains"] = list(self.exclude_domains)
        return d

    def identity(self) -> dict:
        d = self.to_dict()
        for k in _LOCATION_FIELDS:
            d.pop(k)
        return d

    def sha256(self) -> str:
        return sha256_json(self.identity())


def sha256_json(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
