"""Build the balanced five-group dataset from mapped Samples.

Three stages, each with its own report so every number can be checked:

1. SELECT   (select_balanced)
   Per group, walk the candidates in a fixed order (MMFakeBench first,
   then the top-up datasets, each shuffled with the seed) and keep a
   record unless the same (caption, image content) pair was kept before.
   Stop at N per group, where N = the smallest group unless --target
   forces a number.  Image content = sha1 of the file bytes (exact copy)
   plus a difference hash (same picture re-encoded).

2. SPLIT    (assign_splits)
   Records that share an image (sha1 or dhash) or a caption are linked
   into one cluster.  A cluster goes to one split, chosen to keep the
   per-group 70/15/15 proportions.  So a picture or caption never appears
   on both sides of train/test.  This is what leaked last time.

3. WRITE    (write_outputs)
   CSV with one row per record + manifest.json with seed, counts, and
   every skipped/duplicate count.  Run fnd.data.verify on the CSV next.

Image mode:
   required  every image must exist and open; otherwise the record is
             skipped and counted.  Use this on the PC.
   optional  images may be absent (this sandbox); duplicates are then
             detected on caption + image PATH only, and the manifest says
             so loudly.  Never train from an 'optional' build.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import ImageInfoCache, text_key
from .records import GROUPS, LABEL_INDEX, Sample

SOURCE_PRIORITY = ("mmfakebench", "newsclippings", "dgm4")
DEFAULT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ----------------------------------------------------------------------------
# Stage 1: balanced selection with de-duplication
# ----------------------------------------------------------------------------

@dataclass
class SelectionReport:
    target: int = 0                      # N actually used
    image_mode: str = "required"
    available: Counter = field(default_factory=Counter)          # per group, before selection
    selected: Counter = field(default_factory=Counter)           # per group, after
    per_group_source: dict = field(default_factory=lambda: defaultdict(Counter))
    skipped_duplicate: Counter = field(default_factory=Counter)  # per group
    skipped_missing_image: Counter = field(default_factory=Counter)
    skipped_corrupt_image: Counter = field(default_factory=Counter)
    duplicate_examples: list = field(default_factory=list)       # first few (kept, skipped)
    images_hashed: int = 0
    images_unverified: int = 0           # optional mode: keyed by path, not content

    def format(self) -> str:
        lines = [f"SELECTION  target N = {self.target}   image mode = {self.image_mode}"]
        lines.append(f"  {'group':22} {'avail':>6} {'kept':>6} {'dup':>5} {'noimg':>6} {'bad':>4}   sources")
        for g in GROUPS:
            srcs = ", ".join(f"{s}={n}" for s, n in self.per_group_source[g].items())
            lines.append(f"  {g:22} {self.available[g]:6d} {self.selected[g]:6d} "
                         f"{self.skipped_duplicate[g]:5d} {self.skipped_missing_image[g]:6d} "
                         f"{self.skipped_corrupt_image[g]:4d}   {srcs}")
        lines.append(f"  images hashed by content: {self.images_hashed}; "
                     f"keyed by path only (unverified): {self.images_unverified}")
        if self.duplicate_examples:
            lines.append("  duplicate examples (kept <- skipped): " + "; ".join(
                f"{a} <- {b}" for a, b in self.duplicate_examples[:5]))
        return "\n".join(lines)


def _order_candidates(samples: list[Sample], seed: int, priority=SOURCE_PRIORITY) -> dict[str, list[Sample]]:
    """Per group: MMFakeBench first, then top-ups, each block shuffled by seed."""
    by_group: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        if s.source not in priority:
            raise ValueError(f"sample {s.sample_id} has source {s.source!r} not in priority list {priority}")
        by_group[s.group][s.source].append(s)
    rng = random.Random(seed)
    ordered: dict[str, list[Sample]] = {}
    for g in GROUPS:
        seq: list[Sample] = []
        for src in priority:
            block = sorted(by_group[g].get(src, []), key=lambda s: s.sample_id)
            rng.shuffle(block)
            seq.extend(block)
        ordered[g] = seq
    return ordered


def select_balanced(samples: list[Sample], target: int | None = None, seed: int = 42,
                    image_mode: str = "required", priority=SOURCE_PRIORITY,
                    perceptual: bool = True) -> tuple[list[Sample], SelectionReport]:
    if image_mode not in ("required", "optional"):
        raise ValueError("image_mode must be 'required' or 'optional'")
    ordered = _order_candidates(samples, seed, priority)
    rep = SelectionReport(image_mode=image_mode)
    for g in GROUPS:
        rep.available[g] = len(ordered[g])
    if any(rep.available[g] == 0 for g in GROUPS):
        raise ValueError(f"a group has no candidates: {dict(rep.available)}")

    goal = target if target else min(rep.available[g] for g in GROUPS)
    cache = ImageInfoCache(perceptual=perceptual)
    seen_pairs: dict[tuple[str, str], str] = {}
    chosen: dict[str, list[Sample]] = {g: [] for g in GROUPS}

    for g in GROUPS:
        for s in ordered[g]:
            if len(chosen[g]) >= goal:
                break
            tk = text_key(s.text)
            info = cache.get(s.image_path)
            if info.ok:
                ik = "sha1:" + info.sha1
                s.extra.update(image_sha1=info.sha1, image_dhash=info.dhash, image_ok=1)
            else:
                if image_mode == "required":
                    if info.error == "missing":
                        rep.skipped_missing_image[g] += 1
                    else:
                        rep.skipped_corrupt_image[g] += 1
                    continue
                ik = "path:" + str(Path(s.image_path).as_posix()).lower()
                s.extra.update(image_sha1="", image_dhash="", image_ok=0)
            key = (tk, ik)
            if key in seen_pairs:
                rep.skipped_duplicate[g] += 1
                if len(rep.duplicate_examples) < 20:
                    rep.duplicate_examples.append((seen_pairs[key], s.sample_id))
                continue
            seen_pairs[key] = s.sample_id
            s.extra.update(text_key=tk, image_key=ik)
            chosen[g].append(s)

    n = min(len(chosen[g]) for g in GROUPS)
    if target and n < target:
        raise ValueError(f"requested {target} per group but only {n} available after de-duplication: "
                         f"{ {g: len(chosen[g]) for g in GROUPS} }")
    rep.target = n
    selected: list[Sample] = []
    for g in GROUPS:
        chosen[g] = chosen[g][:n]          # deterministic truncation
        rep.selected[g] = len(chosen[g])
        for s in chosen[g]:
            rep.per_group_source[g][s.source] += 1
            if s.extra.get("image_ok"):
                rep.images_hashed += 1
            else:
                rep.images_unverified += 1
        selected.extend(chosen[g])
    return selected, rep


# ----------------------------------------------------------------------------
# Stage 2: leak-free stratified split
# ----------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


@dataclass
class SplitReport:
    fractions: dict = field(default_factory=dict)
    clusters: int = 0
    clusters_multi: int = 0              # clusters with >1 record (shared image or caption)
    largest_cluster: int = 0
    counts: dict = field(default_factory=lambda: defaultdict(Counter))  # split -> group -> n

    def format(self) -> str:
        lines = [f"SPLIT  fractions = {self.fractions}   clusters = {self.clusters} "
                 f"(shared-image/caption clusters = {self.clusters_multi}, largest = {self.largest_cluster})"]
        lines.append(f"  {'group':22} " + " ".join(f"{sp:>6}" for sp in self.fractions))
        for g in GROUPS:
            lines.append(f"  {g:22} " + " ".join(f"{self.counts[sp][g]:6d}" for sp in self.fractions))
        lines.append(f"  {'total':22} " + " ".join(
            f"{sum(self.counts[sp].values()):6d}" for sp in self.fractions))
        return "\n".join(lines)


def assign_splits(selected: list[Sample], fractions: dict[str, float] | None = None,
                  seed: int = 42) -> tuple[dict[str, str], SplitReport]:
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1: {fractions}")

    # Link records sharing a caption or an image (bytes or perceptual hash).
    uf = _UnionFind(len(selected))
    first_seen: dict[tuple[str, str], int] = {}
    for i, s in enumerate(selected):
        keys = [("text", s.extra["text_key"]), ("image", s.extra["image_key"])]
        if s.extra.get("image_dhash"):
            keys.append(("dhash", s.extra["image_dhash"]))
        for k in keys:
            if k in first_seen:
                uf.union(first_seen[k], i)
            else:
                first_seen[k] = i
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(selected)):
        clusters[uf.find(i)].append(i)
    cluster_list = list(clusters.values())
    random.Random(seed).shuffle(cluster_list)
    cluster_list.sort(key=len, reverse=True)      # big clusters first; stable sort keeps shuffle order

    n_per_group = Counter(s.group for s in selected)
    quota = {sp: {g: frac * n_per_group[g] for g in GROUPS} for sp, frac in fractions.items()}
    cur = {sp: Counter() for sp in fractions}
    split_of: dict[str, str] = {}
    rep = SplitReport(fractions=fractions, clusters=len(cluster_list),
                      clusters_multi=sum(1 for c in cluster_list if len(c) > 1),
                      largest_cluster=max(len(c) for c in cluster_list))

    for comp in cluster_list:
        c = Counter(selected[i].group for i in comp)

        def fill_after(sp: str) -> float:
            return max((cur[sp][g] + c[g]) / quota[sp][g] for g in c if quota[sp][g] > 0)

        best = min(fractions, key=lambda sp: (fill_after(sp), sum(cur[sp].values())))
        for i in comp:
            split_of[selected[i].sample_id] = best
            selected[i].extra["cluster"] = uf.find(i)
        cur[best].update(c)
    for sp in fractions:
        rep.counts[sp] = cur[sp]
    return split_of, rep


# ----------------------------------------------------------------------------
# Stage 3: write outputs
# ----------------------------------------------------------------------------

CSV_COLUMNS = [
    "sample_id", "source", "source_split", "scenario", "group", "label_index", "label_binary",
    "text_fake", "image_fake", "ooc", "text", "image_path", "raw_label", "subcategory",
    "text_source", "image_source", "rule", "text_key", "image_key", "image_sha1",
    "image_dhash", "image_ok", "cluster", "split",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_outputs(selected: list[Sample], split_of: dict[str, str], out_dir: str | Path, name: str,
                  sel_rep: SelectionReport, split_rep: SplitReport, config: dict) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    manifest_path = out_dir / f"{name}.manifest.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for s in selected:
            row = s.to_row()
            row.update(
                label_index=LABEL_INDEX[s.group],
                text_key=s.extra["text_key"], image_key=s.extra["image_key"],
                image_sha1=s.extra.get("image_sha1", ""), image_dhash=s.extra.get("image_dhash", ""),
                image_ok=s.extra.get("image_ok", 0), cluster=s.extra.get("cluster", ""),
                split=split_of[s.sample_id],
            )
            w.writerow({k: row[k] for k in CSV_COLUMNS})

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "config": config,
        "rows": len(selected),
        "n_per_group": sel_rep.target,
        "warning": (None if sel_rep.image_mode == "required" else
                    "image_mode=optional: duplicates checked by caption + PATH only; do not train on this"),
        "selection": {
            "target": sel_rep.target,
            "image_mode": sel_rep.image_mode,
            "available": dict(sel_rep.available),
            "selected": dict(sel_rep.selected),
            "per_group_source": {g: dict(c) for g, c in sel_rep.per_group_source.items()},
            "skipped_duplicate": dict(sel_rep.skipped_duplicate),
            "skipped_missing_image": dict(sel_rep.skipped_missing_image),
            "skipped_corrupt_image": dict(sel_rep.skipped_corrupt_image),
            "duplicate_examples": sel_rep.duplicate_examples,
            "images_hashed": sel_rep.images_hashed,
            "images_unverified": sel_rep.images_unverified,
        },
        "split": {"fractions": split_rep.fractions, "clusters": split_rep.clusters,
                  "clusters_multi": split_rep.clusters_multi, "largest_cluster": split_rep.largest_cluster,
                  "counts": {sp: dict(c) for sp, c in split_rep.counts.items()}},
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return csv_path, manifest_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the balanced five-group dataset.")
    ap.add_argument("--mmfakebench", required=True, help="folder with MMFakeBench_val.json / _test.json")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--name", default="balanced_5group")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", type=int, default=None, help="force N per group (default: smallest group)")
    ap.add_argument("--images", choices=["required", "optional"], default="required")
    ap.add_argument("--fractions", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args(argv)

    from .mmfakebench import load_mmfakebench
    samples, load_rep = load_mmfakebench(args.mmfakebench, require_images=(args.images == "required"))
    print(load_rep.format())
    print()

    selected, sel_rep = select_balanced(samples, target=args.target, seed=args.seed, image_mode=args.images)
    print(sel_rep.format())
    print()

    fractions = dict(zip(("train", "val", "test"), args.fractions))
    split_of, split_rep = assign_splits(selected, fractions, seed=args.seed)
    print(split_rep.format())
    print()

    config = {"mmfakebench": str(args.mmfakebench), "seed": args.seed, "target": args.target,
              "images": args.images, "fractions": fractions}
    csv_path, manifest_path = write_outputs(selected, split_of, args.out, args.name, sel_rep, split_rep, config)
    print(f"wrote {csv_path} and {manifest_path}")

    from .verify import verify_csv
    failures = verify_csv(csv_path, fractions=fractions, check_files=(args.images == "required"))
    print("VERIFY: " + ("PASS" if not failures else "FAIL\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
