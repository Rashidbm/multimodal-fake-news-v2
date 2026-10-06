"""Build the Image Branch v2 image-truth mapping and task manifests from the team dataset.

Scope: the team dataset only, and only images whose ORIGINAL project split is ``train``.
No GenImage, no DGM4, no external data. Original project val/test rows are never read
into a manifest. See docs/IMAGE_DATA_MAPPING.md.

    python -m fnd.data.build_image_branch_v2 \
        --dataset-root "D:\\MultiGuard\\data\\fnd_team_dataset_2026-09-14" \
        --output-dir data/image_branch_v2

Image truth is assigned from RULES (source + subcategory + pixel-size checks), never from
``scenario``, ``label_binary`` or ``image_fake`` alone. Those columns are copied through
as original metadata only. No images are copied or modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

SEED = "image_branch_v2_20261006"
SPLIT_FRACTIONS = (0.70, 0.85)          # cumulative: train < 70%, val < 85%, test remainder
NEAR_DUPLICATE_RADIUS = 4               # dHash Hamming distance
MIN_CAPTION_TOKENS = 4                  # captions shorter than this never link images
TRUTHS = ("REAL", "AI_GENERATED", "MANIPULATED")
JOINT_LABEL = {"REAL": 0, "AI_GENERATED": 1, "MANIPULATED": 2}

# ---------------------------------------------------------------------------------------
# Mapping rules. Key: (source.lower(), subcategory). ``size`` is a pixel-size check taken
# from the audit: an image that fails it is EXCLUDED (low confidence), not force-labelled.
# domain/subsource/family drive balancing and splitting; they are not ground truth.
# ---------------------------------------------------------------------------------------
_NEWS = dict(domain="news")
RULES = {
    # -- NewsCLIPpings: both matched (S4) and mismatched/OOC (S1) images are authentic.
    ("newsclippings", ""): dict(
        rule="NC_PAIRED_REAL", truth="REAL", confidence="high", subsource="nc_jpeg", **_NEWS,
        reason="Official NewsCLIPpings pair; falsified=true only means the caption is mismatched, the image is authentic"),
    **{("newsclippings", s): dict(
        rule="NC_SUPPLEMENT_REAL", truth="REAL", confidence="high", subsource="nc_jpeg", **_NEWS,
        reason="Supplementary matched NewsCLIPpings image (genuine)") for s in ("bbc", "guardian", "usa_today", "washington_post")},
    # -- MMFakeBench real / untouched-image groups
    **{("mmfakebench", s): dict(
        rule="MMFB_GENUINE_REAL", truth="REAL", confidence="high", subsource="mmfb_visualnews_png", **_NEWS,
        reason="Paper section 3.2 real data (VisualNews source)") for s in ("bbc", "guardian", "usa_today", "wash")},
    ("mmfakebench", "coco"): dict(
        rule="MMFB_GENUINE_REAL", truth="REAL", confidence="high", domain="coco", subsource="mmfb_coco_real",
        reason="Paper section 3.2 real MS-COCO photo"),
    ("mmfakebench", "fakeddit"): dict(
        rule="MMFB_GENUINE_REAL", truth="REAL", confidence="medium-high", domain="fakeddit", subsource="mmfb_fakeddit_real",
        reason="Paper section 3.2 real Fakeddit image (user phone photos, memes)"),
    **{("mmfakebench", s): dict(
        rule="MMFB_OOC_REAL", truth="REAL", confidence="high", subsource="mmfb_visualnews_png", **_NEWS,
        reason="Out-of-context: image untouched, caption mismatched") for s in ("Newsclipings_person", "Newsclipings_scene", "Newsclipings_semantic")},
    **{("mmfakebench", s): dict(
        rule="MMFB_FAKE_TEXT_REAL_IMAGE", truth="REAL", confidence="high", subsource="mmfb_visualnews_png", **_NEWS,
        reason="Fake text on a repurposed real VisualNews photo; the image is untouched") for s in ("rumor_match", "politicat_match", "gossipcop_match", "chatgpt_match")},
    ("mmfakebench", "DGM4_text_edit_senti"): dict(
        rule="MMFB_TEXT_EDIT_REAL_IMAGE", truth="REAL", confidence="high", subsource="mmfb_visualnews_png", **_NEWS,
        reason="Only the caption is antonym-edited; the DGM4 original image is untouched"),
    # -- Manipulated (Photoshop-style edits of real photos)
    ("mmfakebench", "Fakeddit_photo_edit"): dict(
        rule="MMFB_PHOTO_EDIT_MANIPULATED", truth="MANIPULATED", confidence="medium-high", domain="fakeddit",
        subsource="fakeddit_photo_edit", reason="PS-edited Fakeddit 'manipulated content' (paper 3.1.2); composites at native sizes 280-5285 px"),
    # -- Fully AI-generated
    ("mmfakebench", "antifact_image_generation"): dict(
        rule="MMFB_ANTIFACT_AI", truth="AI_GENERATED", confidence="high", subsource="midjourney_style_1024", family="midjourney_style_1024",
        size=(1024, 1024), reason="Fully generated image (paper 3.1.2), 1024x1024, Midjourney-like style; generator not recorded. Other sizes are excluded", **_NEWS),
    ("mmfakebench", "fever_AI"): dict(
        rule="MMFB_FEVER_AI", truth="AI_GENERATED", confidence="high", subsource="dalle3_style_1024", family="dalle3_style_1024",
        size=(1024, 1024), reason="Generated support image (paper 3.1.1: SDXL, DALL-E 3, Midjourney V6); 15 PNGs carry sd_xl_base_1.0 metadata", **_NEWS),
    **{("mmfakebench", s): dict(
        rule="MMFB_LLM_AI", truth="AI_GENERATED", confidence="high", subsource="llm_1024_mixed", family="llm_1024_mixed",
        size=(1024, 1024), reason="Generated image paired with a GPT rumor (paper 3.1.1), 1024x1024", **_NEWS)
        for s in ("llm_rewrite", "llm_gossip_md_generation", "llm_science_md_generation")},
    ("mmfakebench", "gossipcop_midjourney"): dict(
        rule="MMFB_MIDJOURNEY_AI", truth="AI_GENERATED", confidence="high", subsource="midjourney_v6_named_1024",
        family="midjourney_v6_named_1024", size=(1024, 1024), reason="Named Midjourney V6 image, 1024x1024", **_NEWS),
    **{("mmfakebench", s): dict(
        rule="MMFB_COCO_CF_AI", truth="AI_GENERATED", confidence="medium-high", domain="coco", subsource="sd512_counterfactual",
        family="sd512_counterfactual", size=(512, 512),
        reason="COCO-Counterfactuals image: uniform 512x512 (real COCO here is 640x480 etc.) and visually diffusion-generated; not an edit of a real photo")
        for s in ("coco_image_edit", "coco_text_edit")},
}
EXCLUDE_RULE = dict(rule="EXCLUDED_SIZE_MISMATCH", truth="EXCLUDE", confidence="low", domain="", subsource="",
                    reason="Group is normally AI-generated but this image does not have the group's generated size; likely an AI-edited real photo or another tool")
MIDJOURNEY_FAMILIES = {"midjourney_style_1024", "midjourney_v6_named_1024"}

COMMON_COLUMNS = ["image_id", "image_path", "source", "subcategory", "all_subcategories", "original_project_split",
                  "image_truth", "label", "confidence", "mapping_rule", "ai_family", "domain", "subsource", "group_id",
                  "split", "balance_role", "width", "height", "format", "mode", "file_size", "sha256", "n_caption_rows",
                  "original_scenarios", "original_image_fake"]
AI_EXTRA = ["split_h_mj", "split_h_cf"]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sort_hash(text):
    return hashlib.sha256((SEED + text).encode()).hexdigest()


def normalize_caption(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def scan_image(args):
    root, row, verify = args
    path = root / row["image_path"]
    raw_size = path.stat().st_size
    if verify and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"sha256 mismatch: {path}")
    with Image.open(path) as image:
        return row["image_asset_id"], dict(width=image.width, height=image.height, format=image.format, mode=image.mode, file_size=raw_size)


def load_train_images(root, verify_hashes=True, workers=8):
    """One record per unique image whose ORIGINAL project split is train."""
    master = read_csv(root / "data/master.csv")
    sample_images = {r["image_asset_id"]: r for r in read_csv(root / "data/image_samples.csv")}
    assets = {}
    for row in master:
        if row["split"] != "train":
            continue
        a = assets.setdefault(row["image_asset_id"], dict(
            image_id=row["image_asset_id"], image_path=row["image_path"], sha256=row["image_sha256"], sha1=row["image_sha1"],
            dhash=row["image_dhash"], sources=set(), subcats=set(), scenarios=set(), fake=set(), captions=set(), rows=[]))
        a["sources"].add(row["source"]); a["subcats"].add(row["subcategory"]); a["scenarios"].add(row["scenario"])
        a["fake"].add(row["image_fake"]); a["captions"].add(normalize_caption(row["text"])); a["rows"].append(row["sample_id"])
    for key, a in assets.items():
        if sample_images[key]["split"] != "train":
            raise ValueError(f"image_samples split disagrees with master for {key}")
        if len(a["fake"]) != 1 or len(a["sources"]) != 1:
            raise ValueError(f"image {key} has inconsistent source/label rows")
    with ThreadPoolExecutor(workers) as pool:
        scans = dict(pool.map(scan_image, [(root, dict(image_asset_id=k, image_path=assets[k]["image_path"], sha256=assets[k]["sha256"]), verify_hashes) for k in sorted(assets)]))
    for key, a in assets.items():
        a.update(scans[key])
    return assets, len(master)


def assign_truth(asset):
    """Apply RULES to every subcategory the image appears under; they must agree."""
    found = set()
    for sub in asset["subcats"]:
        rule = RULES.get((next(iter(asset["sources"])).lower(), sub))
        if rule is None:
            raise KeyError(f"No mapping rule for source={asset['sources']} subcategory={sub!r}; add one explicitly")
        if rule.get("size") and (asset["width"], asset["height"]) != rule["size"]:
            rule = EXCLUDE_RULE
        found.add((rule["rule"], rule["truth"], rule["confidence"], rule["domain"], rule["subsource"], rule.get("family", "")))
    truths = {f[1] for f in found}
    if len(truths) != 1:
        raise ValueError(f"Conflicting truth for {asset['image_id']}: {found}")
    rule_id, truth, confidence, domain, subsource, family = sorted(found)[0]
    asset.update(image_truth=truth, mapping_rule="|".join(sorted(f[0] for f in found)), confidence=confidence,
                 domain=domain, subsource=subsource, ai_family=family)


def build_groups(assets):
    """Union-find over exact identity, dHash near-duplicates and shared long captions."""
    ids = sorted(assets)
    n = len(ids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    hashes = np.array([int(assets[i]["dhash"], 16) for i in ids], dtype=np.uint64)
    popcount = np.array([bin(v).count("1") for v in range(256)], dtype=np.uint8)
    for start in range(0, n, 500):
        distance = popcount[(hashes[start:start + 500, None] ^ hashes[None, :]).view(np.uint8).reshape(-1, n, 8)].sum(2)
        for r in range(distance.shape[0]):
            distance[r, start + r] = 99
        for r, c in zip(*np.where(distance <= NEAR_DUPLICATE_RADIUS)):
            union(start + r, c)
    by_caption = defaultdict(list)
    for k, i in enumerate(ids):
        for caption in assets[i]["captions"]:
            if len(caption.split()) >= MIN_CAPTION_TOKENS:
                by_caption[caption].append(k)
    for members in by_caption.values():
        for k in members[1:]:
            union(members[0], k)
    for k, i in enumerate(ids):
        assets[i]["group_id"] = "g%06d" % find(k)


def assign_splits(usable):
    """70/15/15 by group, stratified by (domain, truth, subsource); one split per image for every task."""
    groups = defaultdict(list)
    for a in usable:
        groups[a["group_id"]].append(a)
    strata = defaultdict(list)
    for gid, members in groups.items():
        strata[min((m["domain"], m["image_truth"], m["subsource"]) for m in members)].append(gid)
    for _, gids in sorted(strata.items()):
        gids.sort(key=sort_hash)
        total, seen = sum(len(groups[g]) for g in gids), 0
        for gid in gids:
            fraction = seen / total
            split = "train" if fraction < SPLIT_FRACTIONS[0] else "val" if fraction < SPLIT_FRACTIONS[1] else "test"
            for m in groups[gid]:
                m["split"] = split
            seen += len(groups[gid])


def pick(items, k):
    return sorted(items, key=lambda a: sort_hash(a["image_id"]))[:k]


def assign_roles(usable):
    """Within each (split, domain) pair one positive source with its matched real source, 1:1.

    core            -> used for headline val/test metrics and as the balanced training core
    reserve_real    -> extra real images in the same domain (resample per epoch in training)
    reserve_positive-> extra positives (only in the COCO domain, where real COCO is the limit)
    """
    cell = defaultdict(list)
    for a in usable:
        a["balance_role"], a["pool"] = "", ""
        cell[(a["split"], a["domain"], a["image_truth"])].append(a)
    for split in ("train", "val", "test"):
        for domain, positive in (("news", "AI_GENERATED"), ("coco", "AI_GENERATED"), ("fakeddit", "MANIPULATED")):
            pos, real = cell[(split, domain, positive)], cell[(split, domain, "REAL")]
            k = min(len(pos), len(real))
            if domain == "news":     # half NewsCLIPpings JPEG, half MMFB PNG, so file format is not a class cue
                jpeg = [a for a in real if a["subsource"] == "nc_jpeg"]
                png = [a for a in real if a["subsource"] != "nc_jpeg"]
                chosen_real, chosen_pos = pick(jpeg, k // 2) + pick(png, k - k // 2), pos
            else:
                chosen_real, chosen_pos = pick(real, k), pick(pos, k)
            real_ids, pos_ids = {a["image_id"] for a in chosen_real}, {a["image_id"] for a in chosen_pos}
            for a in real:
                a["balance_role"] = "core" if a["image_id"] in real_ids else "reserve_real"
            for a in pos:
                a["balance_role"] = "core" if a["image_id"] in pos_ids else "reserve_positive"
            for a in real + pos:
                a["pool"] = "manip" if domain == "fakeddit" else "ai"


def assign_holdouts(usable):
    """Source-held-out columns for the AI detector (AI pool only)."""
    ai = [a for a in usable if a["pool"] == "ai"]
    for a in ai:
        a["split_h_mj"] = a["split_h_cf"] = ""
    midjourney = [a for a in ai if a["ai_family"] in MIDJOURNEY_FAMILIES]
    real_test = [a for a in ai if a["image_truth"] == "REAL" and a["domain"] == "news" and a["split"] == "test"]
    test_real = {a["image_id"] for a in pick(real_test, len(midjourney))}
    for a in ai:     # H-MJ: every Midjourney-family AI image plus matched real test images are the unseen test
        if a["image_truth"] == "AI_GENERATED":
            a["split_h_mj"] = "test" if a["ai_family"] in MIDJOURNEY_FAMILIES else (a["split"] if a["split"] != "test" else "unused")
        else:
            a["split_h_mj"] = "test" if a["image_id"] in test_real else (a["split"] if a["split"] != "test" else "unused")
        # H-CF: COCO domain (SD-512 counterfactuals vs real COCO) is entirely test; train/val use the news domain only
        a["split_h_cf"] = "test" if a["domain"] == "coco" else (a["split"] if a["split"] != "test" else "unused")
    coco_real = [a for a in ai if a["domain"] == "coco" and a["image_truth"] == "REAL"]
    coco_ai = pick([a for a in ai if a["domain"] == "coco" and a["image_truth"] == "AI_GENERATED"], len(coco_real))
    keep = {a["image_id"] for a in coco_real + coco_ai}
    for a in ai:
        if a["domain"] == "coco" and a["image_id"] not in keep:
            a["split_h_cf"] = "unused"


def build(root, verify_hashes=True, workers=8):
    root = Path(root)
    assets, master_rows = load_train_images(root, verify_hashes, workers)
    protected = read_csv(root / "data/protected_external_evaluations.csv")
    protected_sha1 = {r["image_sha1"] for r in protected}
    protected_dhash = {r["image_dhash"] for r in protected}
    for a in assets.values():
        assign_truth(a)
        a["protected"] = a["sha1"] in protected_sha1 or a["dhash"] in protected_dhash
    build_groups(assets)
    ordered = [assets[k] for k in sorted(assets)]
    usable = [a for a in ordered if a["image_truth"] != "EXCLUDE"]
    for a in ordered:
        a["split"] = "excluded"
    assign_splits(usable)
    assign_roles(usable)
    assign_holdouts(usable)
    tables = dict(
        ai_detector=[a for a in usable if a["pool"] == "ai"],
        manip_detector=[a for a in usable if a["pool"] == "manip"],
        joint_forensic=[a for a in usable if a["pool"] in ("ai", "manip")])
    return dict(assets=ordered, usable=usable, excluded=[a for a in ordered if a["image_truth"] == "EXCLUDE"],
                tables=tables, master_rows=master_rows, protected_rows=len(protected))


def label_of(task, a):
    if task == "joint_forensic":
        return JOINT_LABEL[a["image_truth"]]
    return int(a["image_truth"] != "REAL")


def to_row(task, a, columns):
    row = dict(a, source=next(iter(a["sources"])), subcategory=sorted(a["subcats"])[0],
               all_subcategories="|".join(sorted(a["subcats"])), original_project_split="train", label=label_of(task, a) if a["image_truth"] != "EXCLUDE" else "",
               n_caption_rows=len(a["rows"]), original_scenarios="|".join(sorted(a["scenarios"])), original_image_fake=next(iter(a["fake"])))
    return {c: row.get(c, "") for c in columns}


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def task_files(task, items):
    """name -> rows. train keeps core + reserve rows; val/test keep core rows only."""
    columns = COMMON_COLUMNS + (AI_EXTRA if task == "ai_detector" else [])
    ordered = sorted(items, key=lambda a: a["image_id"])
    files = {f"{task}_all.csv": ordered}
    for split in ("train", "val", "test"):
        files[f"{task}_{split}.csv"] = [a for a in ordered if a["split"] == split and (split == "train" or a["balance_role"] == "core")]
    return {name: [to_row(task, a, columns) for a in rows] for name, rows in files.items()}, columns


def mapping_rule_rows():
    rows = [dict(source=src, subcategory=sub or "(none)", rule=r["rule"], final_image_truth=r["truth"], confidence=r["confidence"],
                 required_size=("x".join(map(str, r["size"])) if r.get("size") else ""), domain=r["domain"], reason=r["reason"])
            for (src, sub), r in sorted(RULES.items())]
    rows.append(dict(source="mmfakebench", subcategory="(size check fails)", rule=EXCLUDE_RULE["rule"], final_image_truth="EXCLUDE",
                     confidence="low", required_size="", domain="", reason=EXCLUDE_RULE["reason"]))
    return rows, ["source", "subcategory", "rule", "final_image_truth", "confidence", "required_size", "domain", "reason"]


def validate(result, root):
    """Raise AssertionError on any violated invariant. Returns a dict of checks performed."""
    root = Path(root)
    usable, tables = result["usable"], result["tables"]
    sample_split = {r["image_asset_id"]: r["split"] for r in read_csv(root / "data/image_samples.csv")}
    protected_sha1 = {r["image_sha1"] for r in read_csv(root / "data/protected_external_evaluations.csv")}
    allowed = {"ai_detector": {"REAL", "AI_GENERATED"}, "manip_detector": {"REAL", "MANIPULATED"}, "joint_forensic": set(TRUTHS)}
    checks = {}
    for a in usable:
        assert (root / a["image_path"]).is_file(), f"missing image {a['image_path']}"
        assert sample_split[a["image_id"]] == "train", f"{a['image_id']} is not original train"
        assert not a["protected"] and a["sha1"] not in protected_sha1, f"protected fingerprint {a['image_id']}"
        assert a["image_truth"] in TRUTHS and a["split"] in ("train", "val", "test")
        assert not Path(a["image_path"]).is_absolute() and ".." not in Path(a["image_path"]).parts
    checks["images_exist_and_original_train"] = len(usable)
    assert len({a["image_id"] for a in usable}) == len(usable) and len({a["sha256"] for a in usable}) == len(usable)
    groups = defaultdict(set)
    for a in usable:
        groups[a["group_id"]].add(a["split"])
    assert all(len(s) == 1 for s in groups.values()), "a group crosses train/val/test"
    checks["groups"] = len(groups)
    for task, items in tables.items():
        files, _ = task_files(task, items)
        assert {a["image_truth"] for a in items} <= allowed[task], f"{task} has a disallowed truth class"
        assert all(a["image_truth"] != "EXCLUDE" for a in items)
        seen = {}
        for name in (f"{task}_train.csv", f"{task}_val.csv", f"{task}_test.csv"):
            for row in files[name]:
                assert row["image_id"] not in seen, f"{task}: {row['image_id']} in two splits"
                seen[row["image_id"]] = name
                assert row["split"] == name.split("_")[-1][:-4]
        assert len(files[f"{task}_all.csv"]) == len(items)
        for name in (f"{task}_val.csv", f"{task}_test.csv"):
            assert {r["balance_role"] for r in files[name]} <= {"core"}
        checks[task] = len(items)
    assert not {a["image_id"] for a in result["excluded"]} & {a["image_id"] for t in tables.values() for a in t}, "excluded image in a task manifest"
    return checks


def summary(result):
    def counts(items, key):
        return {str(k): v for k, v in sorted(Counter(key(a) for a in items).items())}
    out = dict(
        master_rows=result["master_rows"], original_train_unique_images=len(result["assets"]),
        truth=counts(result["assets"], lambda a: a["image_truth"]),
        truth_by_rule=counts(result["assets"], lambda a: f"{a['image_truth']}/{a['mapping_rule']}"),
        excluded=len(result["excluded"]), protected_overlap=sum(a["protected"] for a in result["assets"]))
    for task, items in result["tables"].items():
        out[task] = dict(total=len(items), split_truth=counts(items, lambda a: f"{a['split']}/{a['image_truth']}"),
                         role=counts(items, lambda a: f"{a['split']}/{a['balance_role']}/{a['image_truth']}"))
    return out


def main(argv=None):
    global SEED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, help="fnd_team_dataset_2026-09-14 folder (contains data/ and images/)")
    ap.add_argument("--output-dir", default="data/image_branch_v2")
    ap.add_argument("--seed", default=SEED, help=f"split/selection seed string (default {SEED}); changing it changes every split")
    ap.add_argument("--skip-hash-check", action="store_true", help="do not re-hash image bytes (faster)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-determinism", action="store_true", help="rebuild and require identical manifests")
    args = ap.parse_args(argv)
    SEED = args.seed
    root, out = Path(args.dataset_root), Path(args.output_dir)
    result = build(root, not args.skip_hash_check, args.workers)
    checks = validate(result, root)
    outputs = {}
    for task, items in result["tables"].items():
        files, columns = task_files(task, items)
        for name, rows in files.items():
            outputs[name] = (rows, columns)
    excluded_columns = [c for c in COMMON_COLUMNS if c not in ("balance_role", "label")]
    outputs["excluded_ambiguous.csv"] = ([to_row("ai_detector", a, excluded_columns) for a in sorted(result["excluded"], key=lambda a: a["image_id"])], excluded_columns)
    rule_rows, rule_columns = mapping_rule_rows()
    outputs["mapping_rules.csv"] = (rule_rows, rule_columns)
    if args.check_determinism:
        again = build(root, False, args.workers)
        for task, items in again["tables"].items():
            files, _ = task_files(task, items)
            for name, rows in files.items():
                assert rows == outputs[name][0], f"{name} differs between runs"
        checks["deterministic_rerun"] = True
    out.mkdir(parents=True, exist_ok=True)
    for name, (rows, columns) in outputs.items():
        write_csv(out / name, rows, columns)
    report = dict(seed=SEED, checks=checks, **summary(result))
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
