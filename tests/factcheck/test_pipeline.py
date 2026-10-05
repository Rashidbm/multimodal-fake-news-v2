import json

import pytest

from src.factcheck.config import Config
from src.factcheck.doctor import run_doctor
from src.factcheck.pipeline import check_citations, clean_queries
from src.factcheck.retrieval import collect_evidence, html_to_text, normalize_url
from src.factcheck.vlm import extract_json

from .conftest import ARTICLE


def test_citations_require_known_id_and_exact_quote():
    evidence = [{"id": "E1", "url": "u1", "text": "The council  confirmed the bridge\nopened on Monday."}]
    valid, invalid = check_citations([
        {"evidence_id": "E1", "quote": "confirmed the bridge opened"},          # whitespace differs: ok
        {"evidence_id": "E1", "quote": "the bridge collapsed on Monday"},       # invented
        {"evidence_id": "E9", "quote": "confirmed the bridge opened"},          # unknown id
        {"evidence_id": "E1", "quote": "the"},                                  # too short to mean anything
    ], evidence)
    assert [c["quote"] for c in valid] == ["confirmed the bridge opened"] and valid[0]["url"] == "u1"
    assert [c["reason"] for c in invalid] == ["quote_not_in_evidence", "unknown_evidence_id", "quote_too_short"]


def test_html_extraction_skips_boilerplate_and_prefers_article():
    title, text = html_to_text(ARTICLE)
    assert title == "Bridge opened"
    assert text.startswith("The city council confirmed")
    assert "var x" not in text and "Home | News" not in text and "Copyright" not in text


def test_query_cleaning_and_json_extraction():
    assert clean_queries([" a  b ", "A B", "", "c"], 2) == ["a b", "c"]
    assert clean_queries("not a list", 3) == []
    assert extract_json('Sure:\n```json\n{"label": "x"}\n```') == {"label": "x"}


def test_collect_evidence_filters_dedupes_and_fetches(checker, cfg):
    out = collect_evidence(["new bridge opened Monday", "city council bridge"], cfg, checker.searcher,
                           checker.fetcher, excluded_urls=["https://late.example.net/a"])
    urls = [e["url"] for e in out["evidence"]]
    assert urls == ["https://news.example.com/bridge", "https://docs.example.com/report.pdf"]
    assert out["filtered"] == {"excluded_domain": 1, "excluded_url": 1}
    assert [f["status"] for f in out["fetches"]] == ["ok", "unsupported"]
    assert out["evidence"][0]["fetched"] and "Page excerpt: The city council" in out["evidence"][0]["text"]
    assert out["evidence"][0]["query_indices"] == [0, 1]           # duplicate URL merged across queries
    assert [e["id"] for e in out["evidence"]] == ["E1", "E2"]


def test_cutoff_drops_later_and_undated_results(checker, cfg):
    out = collect_evidence(["new bridge opened Monday", "city council bridge"], cfg, checker.searcher,
                           checker.fetcher, cutoff_date="2024-03-31")
    assert [e["url"] for e in out["evidence"]] == ["https://news.example.com/bridge",
                                                   "https://docs.example.com/report.pdf"]
    assert out["filtered"] == {"excluded_domain": 1, "after_cutoff": 1, "undated_with_cutoff": 1}
    with pytest.raises(ValueError):
        collect_evidence(["x"], cfg, checker.searcher, checker.fetcher, cutoff_date="March 2024")


def test_evidence_budget_drops_items(checker, backend):
    small = Config(**{**checker.cfg.to_dict(), "exclude_domains": ("blocked.example.org",), "evidence_budget_chars": 50})
    out = collect_evidence(["new bridge opened Monday"], small, checker.searcher, checker.fetcher)
    assert out["dropped_by_budget"] >= 1 and out["evidence_chars"] <= 50


def test_search_mode_end_to_end(checker, backend, image):
    r = checker.run_example({"id": "a", "text": "The new bridge opened on Monday.", "image": image,
                             "label": "supported"}, "search")
    assert r["status"] == "ok" and r["correct"] and r["prediction"] == "supported"
    assert r["queries"] == backend.queries
    assert r["citations"][0]["url"] == "https://news.example.com/bridge"
    assert r["evidence_origin"] == {"source": "live"}
    assert {c["step"] for c in r["model_calls"]} == {"queries", "verdict"}
    # The image reaches the model as a data URL.
    first = next(body for url, body in backend.calls if url.startswith("vlm.test/v1/chat"))
    assert first["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_uncited_claim_verdict_is_an_abstention(checker, backend, image):
    backend.verdict = {**backend.verdict, "citations": [{"evidence_id": "E1", "quote": "the bridge fell into the river"}]}
    r = checker.run_example({"id": "a", "text": "Bridge opened.", "image": image, "label": "supported"}, "search")
    assert r["status"] == "abstained" and not r["correct"]
    assert r["citation_errors"][0]["reason"] == "quote_not_in_evidence"


def test_closed_book_needs_no_citation(checker, backend, image):
    backend.verdict = {"label": "refuted", "confidence": 0.6, "rationale": "r", "citations": []}
    r = checker.run_example({"id": "a", "text": "Bridge opened.", "image": image, "label": "refuted"}, "closed_book")
    assert r["status"] == "ok" and r["correct"] and r["evidence"] is None
    assert backend.count("search.test") == 0


def test_failures_are_recorded_not_raised(checker, backend, image, tmp_path):
    backend.model_status = 500
    r = checker.run_example({"id": "a", "text": "Bridge opened.", "image": image}, "search")
    assert r["status"] == "error" and r["errors"][-1]["stage"] == "queries" and "HTTP 500" in r["errors"][-1]["message"]
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    r = checker.run_example({"id": "b", "text": "x", "image": str(bad)}, "closed_book")
    assert r["status"] == "error" and r["errors"][-1]["stage"] == "input"
    r = checker.run_example({"id": "c", "text": "x", "task": "nope"}, "closed_book")
    assert r["status"] == "error"


def test_invalid_label_is_an_error(checker, backend, image):
    backend.verdict = {**backend.verdict, "label": "probably_true"}
    r = checker.run_example({"id": "a", "text": "x", "image": image}, "closed_book")
    assert r["status"] == "error" and "probably_true" in r["errors"][-1]["message"]


def test_assessed_mode_filters_irrelevant_evidence(checker, backend, image):
    backend.assessment_relevance = "none"
    backend.verdict = {"label": "insufficient", "confidence": 0.5, "rationale": "nothing relevant", "citations": []}
    r = checker.run_example({"id": "a", "text": "x", "image": image, "label": "insufficient"}, "assessed")
    assert r["status"] == "ok" and set(r["assessment"]) == {ev["id"] for ev in r["evidence"]}
    verdict_call = [b for u, b in backend.calls if u.startswith("vlm.test/v1/chat")][-1]
    assert "[E1]" not in verdict_call["messages"][1]["content"][-1]["text"]


def test_config_hash_ignores_service_location():
    a = Config(vlm_url="http://a/v1")
    assert a.sha256() == Config(vlm_url="http://b/v1").sha256()
    assert a.sha256() != Config(max_queries=4).sha256()


def test_normalize_url():
    assert normalize_url("http://WWW.Example.com/a/#frag") == normalize_url("https://example.com/a")


def test_doctor_with_fake_backends(checker):
    lines = []
    assert run_doctor(checker.cfg, checker.vlm, checker.searcher, log=lines.append)
    assert sum(line.startswith("PASS") for line in lines) == 6


def test_doctor_reports_missing_model(checker):
    cfg = Config(**{**checker.cfg.to_dict(), "exclude_domains": (), "model": "other"})
    checker.vlm.cfg = cfg
    lines = []
    assert not run_doctor(cfg, checker.vlm, checker.searcher, log=lines.append)
    assert lines[0].startswith("FAIL  model server")


def test_result_is_json_serialisable(checker, image):
    r = checker.run_example({"id": "a", "text": "x", "image": image}, "search")
    json.dumps(r)
