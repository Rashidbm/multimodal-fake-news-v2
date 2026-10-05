"""Command line: python -m src.factcheck <command> ...

    doctor           check the model server and SearXNG
    run              fact-check one text + image
    prepare-mmfake   write an MMFakeBench manifest (JSONL)
    evaluate         run a manifest through one mode (resumable)
    compare          write comparison.md for an output directory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import prompts
from .config import MODES, Config


def _add_config_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("backend and retrieval (defaults from FACTCHECK_* environment variables)")
    g.add_argument("--vlm-url", help="OpenAI-compatible base URL (FACTCHECK_VLM_URL)")
    g.add_argument("--search-url", help="SearXNG base URL (FACTCHECK_SEARCH_URL)")
    g.add_argument("--model", help="served model name (FACTCHECK_MODEL)")
    g.add_argument("--exclude-domain", action="append", default=[], metavar="DOMAIN",
                   help="drop search results from this domain and its subdomains (repeatable)")
    g.add_argument("--cutoff-date", metavar="YYYY-MM-DD",
                   help="drop results published after this date, and undated results; per-example cutoff_date wins")
    g.add_argument("--max-queries", type=int)
    g.add_argument("--evidence-budget-chars", type=int)


def _config(args) -> Config:
    overrides = dict(vlm_url=args.vlm_url, search_url=args.search_url, model=args.model,
                     cutoff_date=args.cutoff_date, max_queries=args.max_queries,
                     evidence_budget_chars=args.evidence_budget_chars)
    if args.exclude_domain:
        overrides["exclude_domains"] = args.exclude_domain
    return Config.from_env(**overrides)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.factcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="check backend readiness")
    _add_config_args(p)

    p = sub.add_parser("run", help="fact-check one claim")
    p.add_argument("--text", required=True)
    p.add_argument("--image", help="path to the post's image")
    p.add_argument("--mode", default="search", choices=["closed_book", "search", "assessed"],
                   help="assessed = live search followed by the evidence assessment step")
    p.add_argument("--task", default=prompts.DEFAULT_TASK, choices=sorted(prompts.TASKS))
    p.add_argument("--exclude-url", action="append", default=[], metavar="URL")
    p.add_argument("--output", help="write the full result JSON here")
    _add_config_args(p)

    p = sub.add_parser("prepare-mmfake", help="write an MMFakeBench manifest")
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--root", default=None, help="MMFakeBench folder (default data/raw/MMFakeBench)")
    p.add_argument("--balanced", action="store_true", help="equal examples per fake_cls label")
    p.add_argument("--limit", type=int, default=0, help="number of examples; 0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--force", action="store_true", help="overwrite an existing manifest")

    p = sub.add_parser("evaluate", help="run a manifest through one mode")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True, help="experiment directory; the mode is a subfolder")
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--evidence-from", help="search run directory to replay (direct and assessed modes)")
    _add_config_args(p)

    p = sub.add_parser("compare", help="compare the runs in an experiment directory")
    p.add_argument("output")

    args = ap.parse_args(argv)

    if args.cmd == "doctor":
        from .doctor import run_doctor
        return 0 if run_doctor(_config(args)) else 1

    if args.cmd == "run":
        from .pipeline import FactChecker
        cfg = _config(args)
        example = {"id": "cli", "text": args.text, "image": args.image, "task": args.task,
                   "excluded_urls": args.exclude_url}
        result = FactChecker(cfg).run_example(example, args.mode)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"status:     {result['status']}")
        print(f"prediction: {result['prediction']}  (confidence {result['confidence']})")
        if result["rationale"]:
            print(f"rationale:  {result['rationale']}")
        for c in result["citations"]:
            print(f"  [{c['evidence_id']}] \"{c['quote']}\"\n      {c['url']}")
        for e in result["errors"]:
            print(f"error ({e['stage']}): {e['message']}")
        for w in result["warnings"]:
            print(f"warning: {w}")
        if args.output:
            print(f"full result: {args.output}")
        return 0 if result["status"] != "error" else 1

    if args.cmd == "prepare-mmfake":
        from .mmfake import DEFAULT_ROOT, build_manifest, write_manifest
        out = Path(args.output)
        if out.exists() and not args.force:
            print(f"{out} already exists; reusing it (pass --force to rebuild)")
            return 0
        rows, meta = build_manifest(args.root or DEFAULT_ROOT, args.split, args.limit, args.balanced, args.seed)
        write_manifest(rows, meta, out, force=args.force)
        print(f"wrote {len(rows)} examples to {out}  labels {meta['labels']}  ({meta['note']})")
        return 0

    if args.cmd == "evaluate":
        from .evaluate import run_manifest
        metrics = run_manifest(args.manifest, args.output, args.mode, _config(args), args.evidence_from)
        return 0 if metrics["n"] else 1

    if args.cmd == "compare":
        from .evaluate import compare
        compare(args.output)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
