"""One example through one mode.

closed_book  text + image -> verdict
search       text + image -> queries -> search/fetch -> evidence -> verdict
direct       saved evidence (from a search run) -> fresh verdict
assessed     saved evidence -> per-item relevance/stance assessment -> fresh verdict

``run_example`` never raises for a model, search or input failure: the result
has ``status`` = "error" and an ``errors`` list, so a failed example is
recorded and counted rather than silently dropped.  ``status`` = "abstained"
means a verdict was produced but rejected by a safeguard (no valid citation).
"""
from __future__ import annotations

import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import prompts
from .config import MODES, REPLAY_MODES, Config, sha256_file, sha256_json
from .retrieval import PageFetcher, SearxClient, collect_evidence
from .vlm import VLMClient, VLMError, image_data_url, user_content

SCHEMA_VERSION = 1

SEARCH_TOOL_NAME = "web_search"


def search_tool(max_queries: int) -> dict:
    return {"type": "function", "function": {
        "name": SEARCH_TOOL_NAME,
        "description": "Search the web. Returns result titles, snippets and page excerpts.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["queries"], "properties": {
            "queries": {"type": "array", "minItems": 1, "maxItems": max_queries,
                        "items": {"type": "string", "maxLength": 200}}}}}}


def verdict_schema(task: str) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": ["label", "confidence", "rationale", "citations"],
            "properties": {
                "label": {"type": "string", "enum": prompts.labels(task)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
                "citations": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False, "required": ["evidence_id", "quote"],
                    "properties": {"evidence_id": {"type": "string"}, "quote": {"type": "string"}}}}}}


ASSESS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["assessments"], "properties": {
    "assessments": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["evidence_id", "relevance", "stance", "note"],
        "properties": {"evidence_id": {"type": "string"},
                       "relevance": {"type": "string", "enum": ["high", "low", "none"]},
                       "stance": {"type": "string", "enum": ["supports", "refutes", "neutral"]},
                       "note": {"type": "string"}}}}}}


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def check_citations(citations: list[dict], evidence: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep citations whose id exists and whose quote occurs verbatim (whitespace/case/quote-mark
    normalized) in that item.  This checks the quotation exists, not that it supports the verdict."""
    by_id = {ev["id"]: ev for ev in evidence}
    valid, invalid = [], []
    for c in citations or []:
        eid, quote = str(c.get("evidence_id", "")).strip(), str(c.get("quote", ""))
        if eid not in by_id:
            invalid.append({**c, "reason": "unknown_evidence_id"})
        elif len(_norm(quote)) < 8:
            invalid.append({**c, "reason": "quote_too_short"})
        elif _norm(quote) not in _norm(by_id[eid]["text"]):
            invalid.append({**c, "reason": "quote_not_in_evidence"})
        else:
            valid.append({"evidence_id": eid, "quote": quote, "url": by_id[eid]["url"]})
    return valid, invalid


def clean_queries(raw, max_queries: int) -> list[str]:
    out, seen = [], set()
    for q in raw if isinstance(raw, list) else []:
        q = re.sub(r"\s+", " ", str(q)).strip()[:200]
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:max_queries]


def input_sha256(example: dict) -> str:
    """Identity of an example's input: text, image bytes, label, task and evidence restrictions."""
    image = example.get("image")
    return sha256_json({
        "text": example.get("text", ""),
        "image_sha256": sha256_file(image) if image and Path(image).is_file() else None,
        "task": example.get("task", prompts.DEFAULT_TASK),
        "label": example.get("label"),
        "excluded_urls": sorted(example.get("excluded_urls") or []),
        "cutoff_date": example.get("cutoff_date"),
    })


class FactChecker:
    def __init__(self, cfg: Config, vlm: VLMClient | None = None, searcher: SearxClient | None = None,
                 fetcher: PageFetcher | None = None):
        self.cfg = cfg
        self.vlm = vlm or VLMClient(cfg)
        self.searcher = searcher or SearxClient(cfg)
        self.fetcher = fetcher or PageFetcher(cfg)

    # -- model steps ------------------------------------------------------

    def _messages(self, system: str, user: str, image_url: str | None) -> list[dict]:
        return [{"role": "system", "content": system},
                {"role": "user", "content": user_content(user, image_url)}]

    def generate_queries(self, text: str, image_url: str | None) -> tuple[list[str], dict]:
        msgs = self._messages(prompts.QUERY_SYSTEM, prompts.query_user(text, self.cfg.max_queries), image_url)
        args, raw = self.vlm.tool_arguments(msgs, search_tool(self.cfg.max_queries))
        return clean_queries(args.get("queries"), self.cfg.max_queries), raw

    def assess(self, text: str, image_url: str | None, evidence: list[dict]) -> tuple[dict, dict]:
        msgs = self._messages(prompts.ASSESS_SYSTEM, prompts.assess_user(text, evidence), image_url)
        obj, raw = self.vlm.json_object(msgs, "evidence_assessment", ASSESS_SCHEMA)
        ids = {ev["id"] for ev in evidence}
        out = {}
        for a in obj.get("assessments", []):
            eid = str(a.get("evidence_id", "")).strip()
            if eid in ids and eid not in out:
                out[eid] = {"relevance": a.get("relevance"), "stance": a.get("stance"), "note": a.get("note", "")}
        return out, raw

    def verdict(self, task: str, text: str, image_url: str | None, evidence: list[dict] | None,
                assessments: dict | None = None) -> tuple[dict, dict]:
        msgs = self._messages(prompts.verdict_system(task, evidence is not None),
                              prompts.verdict_user(text, evidence, assessments), image_url)
        obj, raw = self.vlm.json_object(msgs, "verdict", verdict_schema(task))
        label = str(obj.get("label", "")).strip()
        if label not in prompts.labels(task):
            raise VLMError(f"label {label!r} is not one of {prompts.labels(task)}")
        try:
            conf = float(obj.get("confidence"))
        except (TypeError, ValueError):
            conf = None
        return {"label": label, "confidence": conf, "rationale": str(obj.get("rationale", "")),
                "citations": obj.get("citations") or []}, raw

    # -- one example --------------------------------------------------------

    def run_example(self, example: dict, mode: str, saved: dict | None = None) -> dict:
        """``saved`` is the search-mode result file for this example (replay modes only).
        Without it, direct/assessed collect evidence live (used by the API)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        task = example.get("task") or prompts.DEFAULT_TASK
        result = {
            "schema": SCHEMA_VERSION, "id": example.get("id"), "mode": mode, "task": task,
            "status": "error", "prediction": None, "label": example.get("label"), "correct": False,
            "confidence": None, "rationale": None, "citations": [], "citation_errors": [],
            "queries": [], "retrieval": None, "evidence": None, "evidence_origin": None,
            "assessment": None, "errors": [], "warnings": [], "timings": {}, "model_calls": [],
            "config_sha256": self.cfg.sha256(), "model": self.cfg.model,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        t_start = time.perf_counter()
        stage = "input"
        try:
            prompts.labels(task)
            text = str(example.get("text", "")).strip()
            if not text:
                raise ValueError("example has no text")
            image_url = None
            if example.get("image"):
                image_url = image_data_url(example["image"], self.cfg.max_image_bytes)

            evidence = None
            if mode == "search" or (mode in REPLAY_MODES and saved is None):
                stage = "queries"
                t = time.perf_counter()
                queries, raw = self.generate_queries(text, image_url)
                result["timings"]["queries_s"] = round(time.perf_counter() - t, 3)
                result["model_calls"].append(_call_record("queries", raw))
                result["queries"] = queries
                if not queries:
                    raise VLMError("model returned no usable search queries")
                stage = "retrieval"
                t = time.perf_counter()
                retrieval = collect_evidence(queries, self.cfg, self.searcher, self.fetcher,
                                             example.get("excluded_urls") or (), example.get("cutoff_date"))
                result["timings"]["retrieval_s"] = round(time.perf_counter() - t, 3)
                evidence = retrieval.pop("evidence")
                result["errors"].extend(retrieval.pop("errors"))
                result["retrieval"] = retrieval
                result["evidence_origin"] = {"source": "live"}
                if all(q["status"] == "error" for q in retrieval["queries"]):
                    raise RuntimeError("every search query failed")
            elif mode in REPLAY_MODES:
                stage = "replay"
                evidence = _saved_evidence(saved)
                result["queries"] = saved.get("queries", [])
                result["evidence_origin"] = {"source": "saved", "run_mode": saved.get("mode"),
                                             "config_sha256": saved.get("config_sha256"),
                                             "evidence_sha256": sha256_json(evidence)}
            result["evidence"] = evidence

            assessments = None
            if mode == "assessed":
                stage = "assessment"
                if evidence:
                    t = time.perf_counter()
                    assessments, raw = self.assess(text, image_url, evidence)
                    result["timings"]["assessment_s"] = round(time.perf_counter() - t, 3)
                    result["model_calls"].append(_call_record("assessment", raw))
                    missing = [ev["id"] for ev in evidence if ev["id"] not in assessments]
                    if missing:
                        result["warnings"].append(f"assessment skipped {', '.join(missing)}")
                    evidence = [ev for ev in evidence if assessments.get(ev["id"], {}).get("relevance") != "none"]
                result["assessment"] = assessments or {}

            stage = "verdict"
            t = time.perf_counter()
            verdict, raw = self.verdict(task, text, image_url, evidence, assessments)
            result["timings"]["verdict_s"] = round(time.perf_counter() - t, 3)
            result["model_calls"].append(_call_record("verdict", raw))

            valid, invalid = check_citations(verdict["citations"], evidence or [])
            result.update(prediction=verdict["label"], confidence=verdict["confidence"],
                          rationale=verdict["rationale"], citations=valid, citation_errors=invalid,
                          status="ok")
            if (evidence is not None and self.cfg.require_citation and not valid
                    and verdict["label"] in prompts.CITATION_REQUIRED.get(task, set())):
                result["status"] = "abstained"
                result["warnings"].append("verdict has no valid citation; counted as abstention")
        except Exception as e:  # noqa: BLE001 - every failure must be recorded, not raised
            result["errors"].append({"stage": stage, "message": f"{type(e).__name__}: {e}",
                                     "traceback": traceback.format_exc(limit=3)})
            result["status"] = "error"
        result["latency_s"] = round(time.perf_counter() - t_start, 3)
        result["correct"] = bool(result["status"] == "ok" and result["label"] is not None
                                 and result["prediction"] == result["label"])
        return result


def _saved_evidence(saved: dict) -> list[dict]:
    if saved.get("mode") != "search":
        raise ValueError(f"saved evidence must come from a search run, got mode {saved.get('mode')!r}")
    if saved.get("evidence") is None:
        stages = ", ".join(e.get("stage", "?") for e in saved.get("errors", [])) or "unknown"
        raise ValueError(f"saved search run has no evidence (failed at: {stages})")
    return saved["evidence"]


def _call_record(step: str, raw: dict) -> dict:
    msg = raw.get("message", {})
    return {"step": step, "latency_s": raw.get("latency_s"), "usage": raw.get("usage"),
            "finish_reason": raw.get("finish_reason"), "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"), "reasoning": msg.get("reasoning_content")}
