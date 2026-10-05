import json

import pytest

from src.factcheck.config import Config
from src.factcheck.evaluate import RunConfigMismatch, compare, run_manifest
from src.factcheck.metrics import compute_metrics, mcnemar_exact
from src.factcheck.mmfake import build_manifest, write_manifest as write_mmfake

from .conftest import write_manifest


def _rows(image):
    return [{"id": "a", "text": "The new bridge opened.", "image": image, "label": "supported"},
            {"id": "b/2", "text": "The bridge collapsed.", "image": image, "label": "refuted"}]


def test_run_resume_and_refuse_changed_config(checker, backend, image, tmp_path):
    m = write_manifest(tmp_path / "m.jsonl", _rows(image))
    out = tmp_path / "exp"
    met = run_manifest(m, out, "closed_book", checker.cfg, checker=checker, log=lambda *_: None)
    assert met["n"] == 2 and met["accuracy"] == 0.5 and met["ran"] == 2
    assert (out / "closed_book" / "examples" / "b_2.json").is_file()
    assert (out / "closed_book" / "report.md").read_text().startswith("# Fact-check run: closed_book")
    calls = len(backend.calls)
    met = run_manifest(m, out, "closed_book", checker.cfg, checker=checker, log=lambda *_: None)
    assert met["resumed"] == 2 and met["ran"] == 0 and len(backend.calls) == calls

    changed = Config(**{**checker.cfg.to_dict(), "exclude_domains": tuple(checker.cfg.exclude_domains),
                        "max_queries": 5})
    with pytest.raises(RunConfigMismatch, match="max_queries"):
        run_manifest(m, out, "closed_book", changed, checker=checker, log=lambda *_: None)


def test_errors_are_kept_on_resume(checker, backend, image, tmp_path):
    m = write_manifest(tmp_path / "m.jsonl", _rows(image))
    backend.model_status = 503
    met = run_manifest(m, tmp_path / "exp", "closed_book", checker.cfg, checker=checker, log=lambda *_: None)
    assert met["status"]["error"] == 2 and met["accuracy"] == 0.0
    backend.model_status = 200
    met = run_manifest(m, tmp_path / "exp", "closed_book", checker.cfg, checker=checker, log=lambda *_: None)
    assert met["status"]["error"] == 2 and met["ran"] == 0


def test_replay_modes_reuse_saved_evidence_and_compare(checker, backend, image, tmp_path):
    m = write_manifest(tmp_path / "m.jsonl", _rows(image))
    out = tmp_path / "exp"
    quiet = {"checker": checker, "log": lambda *_: None}
    run_manifest(m, out, "closed_book", checker.cfg, **quiet)
    run_manifest(m, out, "search", checker.cfg, **quiet)
    searches = backend.count("search.test")
    with pytest.raises(ValueError, match="evidence-from"):
        run_manifest(m, out, "direct", checker.cfg, **quiet)
    run_manifest(m, out, "direct", checker.cfg, evidence_from=out / "search", **quiet)
    run_manifest(m, out, "assessed", checker.cfg, evidence_from=out / "search", **quiet)
    assert backend.count("search.test") == searches            # replay made no new searches

    saved = json.loads((out / "search" / "examples" / "a.json").read_text())
    direct = json.loads((out / "direct" / "examples" / "a.json").read_text())
    assert direct["evidence"] == saved["evidence"] and direct["evidence_origin"]["source"] == "saved"
    metrics = json.loads((out / "direct" / "metrics.json").read_text())
    assert "replay latency excludes evidence collection" in metrics["latency_s"]["note"]

    text = compare(out, log=lambda *_: None)
    assert "Matched examples present in every run: **2**" in text
    assert "### search vs closed_book" in text and "### assessed vs direct" in text
    assert "identical saved evidence: 2/2" in text
    assert (out / "comparison.json").is_file()


def test_replay_of_failed_search_is_an_error(checker, backend, image, tmp_path):
    m = write_manifest(tmp_path / "m.jsonl", _rows(image)[:1])
    out = tmp_path / "exp"
    backend.model_status = 500
    run_manifest(m, out, "search", checker.cfg, checker=checker, log=lambda *_: None)
    backend.model_status = 200
    met = run_manifest(m, out, "direct", checker.cfg, evidence_from=out / "search", checker=checker,
                       log=lambda *_: None)
    assert met["status"]["error"] == 1 and met["errors_by_stage"] == {"replay": 1}


def test_metrics_count_errors_and_abstentions_as_wrong():
    rows = [
        {"status": "ok", "prediction": "supported", "label": "supported", "correct": True, "errors": [], "latency_s": 1},
        {"status": "ok", "prediction": "supported", "label": "refuted", "correct": False, "errors": [], "latency_s": 3},
        {"status": "abstained", "prediction": "refuted", "label": "refuted", "correct": False, "errors": [], "latency_s": 2},
        {"status": "error", "prediction": None, "label": "supported", "correct": False,
         "errors": [{"stage": "verdict"}], "latency_s": 9},
    ]
    m = compute_metrics(rows, ["supported", "refuted"])
    assert m["accuracy"] == 0.25 and m["accuracy_answered"] == 0.5
    assert m["per_class"]["supported"]["recall"] == 0.5 and m["per_class"]["refuted"]["recall"] == 0.0
    assert m["macro_f1"] == round((2 * 0.5 * 0.5 / 1.0 + 0) / 2, 4)
    assert m["confusion"]["refuted"] == {"supported": 1, "refuted": 0, "abstained": 1, "error": 0}
    assert m["latency_s"]["median"] == 2 and m["errors_by_stage"] == {"verdict": 1}


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(3, 3) == 1.0


def _mmfake_root(tmp_path, image_bytes):
    folders = {"original": "/real/bbc_val_10", "textual_veracity_distortion": "/fake/rumor_match_val_10",
               "visual_veracity_distortion": "/fake/Fakeddit_photo_edit_val_10",
               "mismatch": "/fake/Newsclipings_person_val_10"}
    records = []
    for label, folder in folders.items():
        for i in range(3):
            rel = f"{folder}/{i}.png"
            p = tmp_path / "MMFakeBench_val" / rel.lstrip("/")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(image_bytes)
            records.append({"text": f"{label} {i}", "image_path": rel, "fake_cls": label,
                            "gt_answers": "True" if label == "original" else "Fake",
                            "text_source": "", "image_source": ""})
    (tmp_path / "MMFakeBench_val.json").write_text(json.dumps(records))
    return tmp_path


def test_prepare_mmfake_balanced(tmp_path, image):
    root = _mmfake_root(tmp_path / "mmfb", open(image, "rb").read())
    rows, meta = build_manifest(root, "val", limit=8, balanced=True, seed=0)
    assert len(rows) == 8 and meta["labels"] == {c: 2 for c in meta["labels"]} and len(meta["labels"]) == 4
    assert all(r["task"] == "mmfakebench" and r["image"].startswith("/") for r in rows)
    assert rows == build_manifest(root, "val", limit=8, balanced=True, seed=0)[0]   # deterministic
    with pytest.raises(ValueError, match="divisible"):
        build_manifest(root, "val", limit=6, balanced=True)
    out = write_mmfake(rows, meta, tmp_path / "m.jsonl")
    assert json.loads((tmp_path / "m.jsonl.meta.json").read_text())["n"] == 8
    with pytest.raises(FileExistsError):
        write_mmfake(rows, meta, out)
    full, meta = build_manifest(root, "val", limit=0, balanced=False)
    assert len(full) == 12 and meta["note"] == "full split"


def test_prepare_mmfake_requires_images(tmp_path, image):
    root = _mmfake_root(tmp_path / "mmfb", open(image, "rb").read())
    next((root / "MMFakeBench_val").rglob("*.png")).unlink()
    with pytest.raises(FileNotFoundError, match="1 of 12"):
        build_manifest(root, "val", limit=0, balanced=False)


def test_mmfakebench_task_runs(checker, backend, image, tmp_path):
    backend.verdict = {"label": "mismatch", "confidence": 0.7, "rationale": "r", "citations": []}
    m = write_manifest(tmp_path / "m.jsonl", [{"id": "x", "task": "mmfakebench", "text": "t", "image": image,
                                               "label": "mismatch"}])
    met = run_manifest(m, tmp_path / "exp", "search", checker.cfg, checker=checker, log=lambda *_: None)
    assert met["accuracy"] == 1.0 and met["status"]["abstained"] == 0      # no citation rule for this task
    assert list(met["per_class"]) == ["original", "textual_veracity_distortion",
                                      "visual_veracity_distortion", "mismatch"]
