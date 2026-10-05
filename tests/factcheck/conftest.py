"""Fake model server, SearXNG and web pages for offline tests.

Nothing here touches the network or a GPU.  Passing these tests says the code
paths work; it says nothing about the real model, live retrieval or accuracy.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from src.factcheck.config import Config
from src.factcheck.doctor import tiny_png
from src.factcheck.pipeline import FactChecker
from src.factcheck.retrieval import PageFetcher, SearxClient
from src.factcheck.vlm import VLMClient

ARTICLE = ("<html><head><title>Bridge opened</title><script>var x=1;</script></head><body>"
           "<nav>Home | News</nav><article><p>The city council confirmed on Monday that the new bridge "
           "opened to traffic after three years of construction.</p><p>" + "Officials said more. " * 20 +
           "</p></article><footer>Copyright</footer></body></html>")


class FakeBackend:
    """Routes requests by host.  Tweak the attributes to steer a test."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.verdict = {"label": "supported", "confidence": 0.8, "rationale": "Council confirmed it.",
                        "citations": [{"evidence_id": "E1", "quote": "confirmed on Monday that the new bridge"}]}
        self.queries = ["new bridge opened Monday", "city council bridge"]
        self.assessment_relevance = "high"
        self.model_status = 200
        self.search_results = {
            "new bridge opened Monday": [
                {"url": "https://news.example.com/bridge", "title": "Bridge opens", "content": "The new bridge opened.",
                 "engine": "duckduckgo", "publishedDate": "2024-03-04T10:00:00"},
                {"url": "https://blocked.example.org/x", "title": "Blocked", "content": "should be excluded"},
                {"url": "https://late.example.net/a", "title": "Later", "content": "after cutoff",
                 "publishedDate": "2025-01-01"},
            ],
            "city council bridge": [
                {"url": "https://www.news.example.com/bridge/", "title": "dup", "content": "duplicate URL"},
                {"url": "https://docs.example.com/report.pdf", "title": "Report", "content": "PDF report",
                 "publishedDate": "2024-02-01"},
            ],
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        body = json.loads(request.content) if request.content else {}
        self.calls.append((f"{host}{path}", body))
        if host == "vlm.test":
            return self._vlm(path, body)
        if host == "search.test":
            q = request.url.params.get("q")
            results = self.search_results.get(q, [{"url": "https://other.example.com/", "title": "Other",
                                                   "content": "Something else."}])
            return httpx.Response(200, json={"results": results, "unresponsive_engines": []})
        if path.endswith(".pdf"):
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4")
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=ARTICLE)

    def _vlm(self, path, body):
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake-model", "max_model_len": 32768}]})
        if self.model_status != 200:
            return httpx.Response(self.model_status, text="server exploded")
        if body.get("tool_choice"):
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "web_search", "arguments": json.dumps({"queries": self.queries})}}]}
        elif (body.get("response_format") or {}).get("json_schema", {}).get("name") == "evidence_assessment":
            ids = [line[1:line.index("]")] for line in body["messages"][1]["content"][-1]["text"].splitlines()
                   if line.startswith("[E")]
            msg = {"role": "assistant", "content": json.dumps({"assessments": [
                {"evidence_id": i, "relevance": self.assessment_relevance, "stance": "supports", "note": "on topic"}
                for i in ids]})}
        elif body.get("response_format"):
            msg = {"role": "assistant", "content": json.dumps(self.verdict)}
        else:
            text = json.dumps(body["messages"])
            msg = {"role": "assistant", "content": "red" if "colour" in text else "READY"}
        return httpx.Response(200, json={"choices": [{"message": msg, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    def count(self, prefix: str) -> int:
        return sum(1 for url, _ in self.calls if url.startswith(prefix))


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def cfg():
    return Config(vlm_url="http://vlm.test/v1", search_url="http://search.test", model="fake-model",
                  exclude_domains=("blocked.example.org",))


@pytest.fixture
def checker(cfg, backend):
    http = httpx.Client(transport=httpx.MockTransport(backend.handler))
    return FactChecker(cfg, VLMClient(cfg, http), SearxClient(cfg, http), PageFetcher(cfg, http))


@pytest.fixture
def image(tmp_path) -> str:
    p = tmp_path / "img.png"
    p.write_bytes(base64.b64decode(tiny_png().split(",", 1)[1]))
    return str(p)


def write_manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path
