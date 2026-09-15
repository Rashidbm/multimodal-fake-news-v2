#!/usr/bin/env python
"""
Recover each caption's source, and derive an AI-authorship label from it.

    python scripts/enrich_provenance.py --csv data/processed/balanced_5group.csv \\
        --raw data/raw/MMFakeBench --out data/processed/enriched.csv

WHY
---
`text_fake` is not `text_was_written_by_AI`. The fake-text scenarios mix
LLM-generated captions, human-written rumours and algorithmic word edits, so
scoring Text Fluoroscopy - a machine-generation detector - against text_fake
measures it on a task it was not built for.

The branch instructions allow an AI-authorship head provided the labels are
*derived from the generating process* rather than assumed. In MMFakeBench the
source folder IS the generating process: the paper documents which
sub-categories were produced by an LLM. This script matches each caption back
to its folder and writes:

    subcategory   the source folder, e.g. llm_rewrite, gossipcop_match
    domain        topic bucket, for the confound check below
    ai_text       1 = LLM-generated, 0 = human-written, blank = excluded

Rows whose provenance is ambiguous for authorship - algorithmic word edits,
and any caption that appears under two different folders - are left blank and
take no part in the AI head. That exclusion is the thing that makes the label
defensible.

THE CONFOUND, AND THE CONTROL
-----------------------------
If every AI caption is gossip and every human caption is BBC news, a
classifier scores well by detecting topic, not authorship. The `domain`
column exists so you can re-run the probe on a single topic:

    python -m fnd.probe_textfor --csv data/processed/enriched.csv \\
        --features features/v_textfor.pt --where domain=gossip \\
        --out outputs/textfor_probe_gossip

gossipcop_match and gossipcop_midjourney are human-written gossip;
llm_gossip_md_generation is LLM-written gossip. Same topic, different
authorship. The gap between the full score and the matched score is the
topic leakage, and reporting both is far stronger than reporting either.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Authorship of the caption, by MMFakeBench sub-category folder.
AI_WRITTEN = {
    "chatgpt_match", "fever_AI", "llm_rewrite",
    "llm_gossip_md_generation", "llm_science_md_generation",
}
HUMAN_WRITTEN = {
    # authentic journalism and photo corpora
    "bbc", "guardian", "usa_today", "wash", "coco", "fakeddit",
    "Newsclipings_person", "Newsclipings_scene", "Newsclipings_semantic",
    # human-written rumours: false, but written by people
    "rumor_match", "politicat_match", "gossipcop_match",
    # human caption, manipulated image - the text is untouched
    "gossipcop_midjourney", "Fakeddit_photo_edit",
    "antifact_image_generation", "coco_image_edit",
}
# Algorithmic word edits: neither cleanly human-authored nor LLM-generated.
EXCLUDED = {"DGM4_text_edit_senti", "coco_text_edit"}

DOMAIN = {
    "gossipcop_match": "gossip", "gossipcop_midjourney": "gossip",
    "llm_gossip_md_generation": "gossip",
    "llm_science_md_generation": "science",
    "fever_AI": "fact", "politicat_match": "politics", "rumor_match": "rumour",
    "bbc": "news", "guardian": "news", "usa_today": "news", "wash": "news",
    "Newsclipings_person": "news", "Newsclipings_scene": "news",
    "Newsclipings_semantic": "news",
    "coco": "captions", "coco_image_edit": "captions", "coco_text_edit": "captions",
    "fakeddit": "social", "Fakeddit_photo_edit": "social",
    "chatgpt_match": "mixed", "llm_rewrite": "mixed",
    "antifact_image_generation": "mixed", "DGM4_text_edit_senti": "news",
}

_SUFFIX = re.compile(r"_(val|test)_\d+$")


def subcategory(image_path: str) -> str | None:
    parts = str(image_path).strip("/").replace("\\", "/").split("/")
    return _SUFFIX.sub("", parts[1]) if len(parts) >= 2 else None


def key(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"[^a-z0-9]+", "", t)


def build_index(raw: Path) -> tuple[dict[str, str], set[str]]:
    """caption -> folder, plus the captions that appear under several."""
    seen: dict[str, set[str]] = defaultdict(set)
    files = sorted(raw.glob("MMFakeBench_*.json"))
    if not files:
        raise FileNotFoundError(f"no MMFakeBench_*.json in {raw}")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", list(data.values()))
        for rec in data:
            sub = subcategory(rec.get("image_path") or "")
            txt = rec.get("text")
            if sub and txt:
                seen[key(txt)].add(sub)
    ambiguous = {k for k, v in seen.items() if len(v) > 1}
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}, ambiguous


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Add source, domain and ai_text to a caption CSV.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--raw", required=True, help="folder holding MMFakeBench_*.json")
    ap.add_argument("--out", default="text_samples_enriched.csv")
    args = ap.parse_args(argv)

    index, ambiguous = build_index(Path(args.raw))
    print(f"indexed {len(index)} unique captions from the MMFakeBench JSON")
    if ambiguous:
        print(f"  {len(ambiguous)} captions appear under more than one folder -> excluded")

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{args.csv} is empty")

    matched = unmatched = 0
    by_sub, by_ai, by_domain = Counter(), Counter(), Counter()
    for r in rows:
        k = key(r.get("text", ""))
        sub = None if k in ambiguous else index.get(k)
        if sub is None:
            unmatched += 1
            r["subcategory"], r["domain"], r["ai_text"] = "", "", ""
            continue
        matched += 1
        r["subcategory"] = sub
        r["domain"] = DOMAIN.get(sub, "other")
        if sub in AI_WRITTEN:
            r["ai_text"] = 1
        elif sub in HUMAN_WRITTEN:
            r["ai_text"] = 0
        else:
            r["ai_text"] = ""          # EXCLUDED, or a folder not in the table
        by_sub[sub] += 1
        by_domain[r["domain"]] += 1
        by_ai[r["ai_text"]] += 1

    print(f"\nmatched {matched}/{len(rows)} captions "
          f"({100*matched/len(rows):.1f}%); {unmatched} unmatched")
    print("\n  ai_text:")
    print(f"    1 (LLM-generated)  {by_ai[1]}")
    print(f"    0 (human-written)  {by_ai[0]}")
    print(f"    blank (excluded)   {by_ai['']}")

    print("\n  by source:")
    for sub, n in by_sub.most_common():
        tag = "AI" if sub in AI_WRITTEN else ("human" if sub in HUMAN_WRITTEN else "excluded")
        print(f"    {sub:<28} {n:>6}   {tag}")

    print("\n  by domain (for the matched-domain control):")
    for dom, n in by_domain.most_common():
        print(f"    {dom:<12} {n}")

    gossip_ai = sum(by_sub[s] for s in ("llm_gossip_md_generation",) if s in by_sub)
    gossip_hu = sum(by_sub[s] for s in ("gossipcop_match", "gossipcop_midjourney") if s in by_sub)
    if gossip_ai and gossip_hu:
        print(f"\n  domain-matched control available: gossip has {gossip_ai} AI and "
              f"{gossip_hu} human captions")
        print("    python -m fnd.probe_textfor --csv <this file> --features <features.pt> \\")
        print("        --where domain=gossip --out outputs/textfor_probe_gossip")
    else:
        print("\n  no domain-matched control available in this file "
              "(need both AI and human captions in one domain)")

    fields = list(rows[0])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten {args.out}")
    if unmatched:
        print(f"note: {unmatched} rows have no source and take no part in the AI head "
              "or the per-source breakdown")
    return 0


if __name__ == "__main__":
    sys.exit(main())
