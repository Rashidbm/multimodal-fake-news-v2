"""Independent re-check of a built dataset CSV.

Deliberately reads only the CSV (never the builder's in-memory state), so a
bug in the builder cannot hide itself.  Every invariant the project promises
is one function below; ``verify_csv`` returns the list of failures.

    python -m fnd.data.verify data/processed/balanced_5group.csv [--check-files]
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from .records import GROUPS, SCENARIO_NUMBER

REQUIRED = {"sample_id", "scenario", "group", "split", "text", "image_path", "text_key", "image_key",
            "image_sha1", "image_dhash", "image_ok", "label_binary"}


def read_rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_rows(rows: list[dict], fractions: dict[str, float] | None = None,
                check_files: bool = False, tolerance: float = 0.03) -> list[str]:
    fails: list[str] = []
    if not rows:
        return ["csv is empty"]
    missing = REQUIRED - set(rows[0].keys())
    if missing:
        return [f"missing columns: {sorted(missing)}"]

    # 1. unique ids
    ids = Counter(r["sample_id"] for r in rows)
    dup_ids = [k for k, n in ids.items() if n > 1]
    if dup_ids:
        fails.append(f"duplicate sample_id: {dup_ids[:5]}")

    # 2. every group present, all equal size
    per_group = Counter(r["group"] for r in rows)
    unknown = set(per_group) - set(GROUPS)
    if unknown:
        fails.append(f"unknown groups: {unknown}")
    sizes = {g: per_group.get(g, 0) for g in GROUPS}
    if len(set(sizes.values())) != 1:
        fails.append(f"groups are not balanced: {sizes}")

    # 3. splits are the three expected values
    splits = Counter(r["split"] for r in rows)
    bad_splits = set(splits) - {"train", "val", "test"}
    if bad_splits:
        fails.append(f"unexpected split values: {bad_splits}")

    # 4. no duplicate (caption, image) pair anywhere
    pairs = Counter((r["text_key"], r["image_key"]) for r in rows)
    dup_pairs = [k for k, n in pairs.items() if n > 1]
    if dup_pairs:
        fails.append(f"{len(dup_pairs)} duplicate caption+image pairs, e.g. {dup_pairs[:3]}")

    # 5. no caption / image content crosses splits
    for col in ("text_key", "image_key", "image_sha1", "image_dhash"):
        seen: dict[str, set] = defaultdict(set)
        for r in rows:
            if r[col]:
                seen[r[col]].add(r["split"])
        leaks = [k for k, s in seen.items() if len(s) > 1]
        if leaks:
            fails.append(f"{len(leaks)} values of {col} appear in more than one split, e.g. {leaks[:3]}")

    # 6. per-group split proportions close to the requested fractions
    if fractions:
        for g in GROUPS:
            n = sizes[g] or 1
            got = Counter(r["split"] for r in rows if r["group"] == g)
            for sp, frac in fractions.items():
                actual = got[sp] / n
                if abs(actual - frac) > tolerance:
                    fails.append(f"group {g}: split {sp} is {actual:.3f} of the group, wanted {frac:.2f}")

    # 7. binary label consistent with group
    for r in rows:
        want = "0" if r["group"] == "genuine" else "1"
        if r["label_binary"] != want:
            fails.append(f"{r['sample_id']}: label_binary {r['label_binary']} but group {r['group']}")
            break

    # 8. scenario number matches the group
    for r in rows:
        if r["scenario"] != str(SCENARIO_NUMBER.get(r["group"], -1)):
            fails.append(f"{r['sample_id']}: scenario {r['scenario']} does not match group {r['group']}")
            break

    # 9. image content verified (and files present) when asked
    if check_files:
        not_ok = [r["sample_id"] for r in rows if r["image_ok"] != "1"]
        if not_ok:
            fails.append(f"{len(not_ok)} rows have image_ok != 1, e.g. {not_ok[:3]}")
        absent = [r["sample_id"] for r in rows if not Path(r["image_path"]).is_file()]
        if absent:
            fails.append(f"{len(absent)} image files not found, e.g. {absent[:3]}")
    return fails


def verify_csv(path: str | Path, fractions: dict[str, float] | None = None,
               check_files: bool = False) -> list[str]:
    return verify_rows(read_rows(path), fractions, check_files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify a built dataset CSV.")
    ap.add_argument("csv")
    ap.add_argument("--check-files", action="store_true", help="also require every image file to exist")
    ap.add_argument("--fractions", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args(argv)
    fractions = dict(zip(("train", "val", "test"), args.fractions))
    rows = read_rows(args.csv)
    fails = verify_rows(rows, fractions, args.check_files)
    per_group = Counter(r["group"] for r in rows)
    per_split = Counter(r["split"] for r in rows)
    print(f"{len(rows)} rows; per group {dict(per_group)}; per split {dict(per_split)}")
    print("VERIFY: " + ("PASS" if not fails else "FAIL\n  " + "\n  ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
