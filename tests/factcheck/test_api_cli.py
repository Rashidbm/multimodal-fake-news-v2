import pytest

from src.factcheck import cli


def test_api_upload(checker, backend, image, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from app import factcheck_server

    monkeypatch.setattr(factcheck_server, "_checker", checker)
    client = TestClient(factcheck_server.app)
    assert client.get("/health").json()["status"] == "ok"
    with open(image, "rb") as f:
        r = client.post("/factcheck", data={"text": "The new bridge opened.", "mode": "search"},
                        files={"image": ("img.png", f, "image/png")})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok" and body["prediction"] == "supported"
    assert "model_calls" not in body
    with open(image, "rb") as f:
        r = client.post("/factcheck", data={"text": "x", "mode": "direct"}, files={"image": ("i.png", f)})
    assert r.status_code == 422


def test_cli_prepare_reuses_existing_manifest(tmp_path, capsys):
    out = tmp_path / "m.jsonl"
    out.write_text("{}\n")
    assert cli.main(["prepare-mmfake", "--split", "val", "--output", str(out), "--root", str(tmp_path)]) == 0
    assert "reusing it" in capsys.readouterr().out and out.read_text() == "{}\n"


def test_cli_exclude_domains_reach_config(monkeypatch):
    monkeypatch.setenv("FACTCHECK_MODEL", "m1")
    p = cli.argparse.Namespace(vlm_url=None, search_url=None, model=None, cutoff_date="2024-01-01",
                               max_queries=None, evidence_budget_chars=8000,
                               exclude_domain=["WWW.Snopes.com.", "politifact.com"])
    cfg = cli._config(p)
    assert cfg.model == "m1" and cfg.cutoff_date == "2024-01-01" and cfg.evidence_budget_chars == 8000
    assert cfg.exclude_domains == ("politifact.com", "snopes.com")
