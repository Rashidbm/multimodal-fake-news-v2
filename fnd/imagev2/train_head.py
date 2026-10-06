"""Train the shared bottleneck/head on cached frozen features and evaluate it.

    python -m fnd.imagev2.train_head --arm A1 --features D:\\...\\features --runs D:\\...\\runs \\
        --manifest-dir data/image_branch_v2 --lambdas 0 0.25 --seeds 13 29 47 --protocols main h_mj h_cf

Protocols (all use only original-project-train images):
  main  joint_forensic: train = split train (core + reserve), val/test = core rows
  h_mj  ai_detector split_h_mj: every Midjourney-family AI image + matched reals are the test set
  h_cf  ai_detector split_h_cf: train/val on the news domain; test = COCO counterfactuals vs real COCO

Loss: BCEWithLogits(binary) + lambda_aux * CE(3-class, label_smoothing 0.1). Selection: worst-domain balanced
accuracy on the clean validation set at threshold 0.5 (ties: lower validation BCE). The test set is never
used for selection or thresholds.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics as M
from .canonical import EVAL_VIEWS, TRAIN_VIEWS
from .manifest import arrays, holdout_columns, load_joint
from .model import ImageHead
from .sampler import DomainClassSampler

DEFAULTS = dict(epoch_size=4096, batch_size=128, lr=5e-4, weight_decay=1e-2, patience=8, max_epochs=60, hidden=1024, dropout=0.2, label_smoothing=0.1)
BASE_ARMS = {"A0": "A1", "A1": "A1", "A1s": "A1s", "B1": "B1_dinov2", "B1_dinov3": "B1_dinov3"}


def load_features(root, arm):
    """view -> (ids, X) for the arm. A0 = first 768 columns of A1; A1s = CLIP from A1 + legacy-squash RGB/DCT."""
    root = Path(root)
    base = root / BASE_ARMS[arm]
    out = {}
    for ids_path in sorted(base.glob("*_ids.npy")):
        view = ids_path.name[:-8]
        if view.endswith(".tmp"):
            continue
        out[view] = (np.load(ids_path), np.load(base / f"{view}.npy"))
    if arm == "A0":
        out = {v: (i, x[:, :768]) for v, (i, x) in out.items()}
    if arm == "A1s":
        clip_ids, clip = np.load(root / "A1" / "clean_ids.npy"), np.load(root / "A1" / "clean.npy")[:, :768]
        ids, rest = out["clean"]
        if not np.array_equal(ids, clip_ids):
            raise ValueError("A1 and A1s clean id order differ")
        out = {"clean": (ids, np.concatenate([clip, rest], axis=1))}
    return out


def select_rows(protocol, rows, arr, holdouts):
    """Boolean masks (train, val, test) over `rows` for the protocol."""
    if protocol == "main":
        core = arr["role"] == "core"
        return arr["split"] == "train", (arr["split"] == "val") & core, (arr["split"] == "test") & core
    column = {"h_mj": 0, "h_cf": 1}[protocol]
    label = np.array([holdouts.get(i, ("", ""))[column] for i in arr["ids"]])
    return label == "train", label == "val", label == "test"


def predict(model, x, device, batch=2048):
    model.eval()
    out = {"logit": [], "aux": [], "v": []}
    with torch.no_grad():
        for s in range(0, len(x), batch):
            r = model(x[s:s + batch].to(device))
            for k in out:
                out[k].append(r[k].float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run_one(cfg, store, rows, arr, holdouts, run_dir, device):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    train_mask, val_mask, test_mask = select_rows(cfg["protocol"], rows, arr, holdouts)
    tr, va, te = [np.where(m)[0] for m in (train_mask, val_mask, test_mask)]
    train_views = [v for v in cfg["train_views"] if v in store]
    pos = {v: {i: k for k, i in enumerate(store[v][0])} for v in store}
    D = store["clean"][1].shape[1]

    def gather(view, idx):
        p = [pos[view][i] for i in arr["ids"][idx]]
        return torch.from_numpy(store[view][1][p])

    train_stack = torch.stack([gather(v, tr) for v in train_views], 1).to(device)       # [N_tr, V, D]
    y3_train = torch.from_numpy(arr["y3"][tr]).to(device)
    model = ImageHead(D, cfg["hidden"], 768, cfg["dropout"]).to(device)
    model.set_statistics(train_stack[:, 0].float())
    sampler = DomainClassSampler(arr["y3"][tr], arr["domain"][tr], arr["subsource"][tr], arr["role"][tr])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    eval_x = {(s, v): gather(v, idx).to(device) for s, idx in (("val", va), ("test", te)) for v in EVAL_VIEWS
              if v in store and all(i in pos[v] for i in arr["ids"][idx])}
    yv, dv = (arr["y3"][va] > 0).astype(int), arr["domain"][va]

    best, best_state, bad, curves = (-1.0, 1e9), None, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        draw = sampler.draw(rng, cfg["epoch_size"])
        views = rng.integers(0, len(train_views), len(draw))
        losses = []
        for s in range(0, len(draw), cfg["batch_size"]):
            b, vb = torch.from_numpy(draw[s:s + cfg["batch_size"]]).to(device), torch.from_numpy(views[s:s + cfg["batch_size"]]).to(device)
            out = model(train_stack[b, vb].float())
            y = y3_train[b]
            loss = F.binary_cross_entropy_with_logits(out["logit"], (y > 0).float())
            if cfg["lambda_aux"] > 0:
                loss = loss + cfg["lambda_aux"] * F.cross_entropy(out["aux"], y, label_smoothing=cfg["label_smoothing"])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.detach()))
        pv = predict(model, eval_x[("val", "clean")], device)
        prob = sigmoid(pv["logit"])
        worst = M.worst_group_bacc(yv, prob, dv)
        bce = float(-np.mean(yv * np.log(prob + 1e-9) + (1 - yv) * np.log(1 - prob + 1e-9)))
        curves.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_worst_domain_bacc=worst, val_bce=bce))
        if (worst, -bce) > (best[0], -best[1]):
            best, best_state, bad = (worst, bce), copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, dict(tr=tr, va=va, te=te), eval_x, curves, best, dict(pos=pos, gather=gather, train_views=train_views,
                                                                       sampler=sampler.report(cfg["epoch_size"]))


def evaluate(cfg, model, idx, eval_x, arr, store_helpers, device):
    out, preds = {}, {}
    va, te = idx["va"], idx["te"]
    yv = (arr["y3"][va] > 0).astype(int)
    pv = predict(model, eval_x[("val", "clean")], device)
    threshold = M.best_threshold(yv, sigmoid(pv["logit"]))
    out["threshold_val_tuned"] = threshold
    for split, ii in (("val", va), ("test", te)):
        y, dom, fam = (arr["y3"][ii] > 0).astype(int), arr["domain"][ii], arr["family"][ii]
        for view in EVAL_VIEWS:
            if (split, view) not in eval_x:
                continue
            p = predict(model, eval_x[(split, view)], device)
            prob = sigmoid(p["logit"])
            preds[f"{split}_{view}"] = dict(logit=p["logit"], aux=p["aux"], ids=arr["ids"][ii])
            entry = {}
            for name, thr in (("at_0.5", 0.5), ("at_val_tuned", threshold)):
                entry[name] = dict(binary=M.binary_metrics(y, prob, thr), by_domain=M.per_group(y, prob, dom, thr),
                                   worst_domain_bacc=M.worst_group_bacc(y, prob, dom, thr))
            fake = y == 1
            entry["fake_recall_by_family_at_0.5"] = {f or "MANIPULATED/none": float((prob[fake & (fam == f)] >= 0.5).mean())
                                                     for f in sorted(set(fam[fake]))}
            if cfg["lambda_aux"] > 0 and view == "clean":
                entry["three_class"] = M.three_class(arr["y3"][ii], p["aux"].argmax(1))
            out[f"{split}_{view}"] = entry
            if view == "clean":
                preds[f"{split}_clean"]["v"] = p["v"].astype(np.float16)
    return out, preds


def representation(cfg, model, idx, eval_x, arr, store, helpers, device):
    """Effective rank and source probes on the frozen exported v_imgfor."""
    va, te, tr = idx["va"], idx["te"], idx["tr"]
    v_val = predict(model, eval_x[("val", "clean")], device)["v"]
    v_test = predict(model, eval_x[("test", "clean")], device)["v"]
    out = dict(effective_rank=M.effective_rank(np.concatenate([v_val, v_test])))
    real_tr = tr[arr["y3"][tr] == 0]
    sub = np.random.default_rng(0).permutation(real_tr)[:4000]
    v_tr = predict(model, helpers["gather"]("clean", sub).to(device), device)["v"]
    real_te = te[arr["y3"][te] == 0]
    real_te_mask = arr["y3"][te] == 0
    v_te = v_test[real_te_mask]
    out["domain_probe_real"] = M.linear_probe(v_tr, arr["domain"][sub], v_te, arr["domain"][real_te])
    news_tr, news_te = arr["domain"][sub] == "news", arr["domain"][real_te] == "news"
    out["news_format_probe_real"] = M.linear_probe(v_tr[news_tr], arr["subsource"][sub][news_tr], v_te[news_te], arr["subsource"][real_te][news_te])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(BASE_ARMS))
    ap.add_argument("--features", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--manifest-dir", default="data/image_branch_v2")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 29, 47])
    ap.add_argument("--protocols", nargs="+", default=["main"], choices=["main", "h_mj", "h_cf"])
    ap.add_argument("--train-views", nargs="+", default=list(TRAIN_VIEWS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args(argv)
    rows = load_joint(args.manifest_dir)
    arr = arrays(rows)
    holdouts = holdout_columns(args.manifest_dir)
    store = load_features(args.features, args.arm)
    if not np.array_equal(store["clean"][0], arr["ids"]):
        raise ValueError("cached clean ids do not match the manifest order; re-extract")
    for protocol in args.protocols:
        for lam in args.lambdas:
            for seed in args.seeds:
                cfg = dict(arm=args.arm, protocol=protocol, lambda_aux=lam, seed=seed, train_views=args.train_views, tag=args.tag,
                           **{k: getattr(args, k) for k in DEFAULTS})
                name = f"{args.arm}{('_' + args.tag) if args.tag else ''}_{protocol}_lam{lam:g}_s{seed}"
                run_dir = Path(args.runs) / name
                if (run_dir / "metrics.json").exists():
                    print(name, "exists, skipping", flush=True)
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                started = time.time()
                model, idx, eval_x, curves, best, helpers = run_one(cfg, store, rows, arr, holdouts, run_dir, args.device)
                result, preds = evaluate(cfg, model, idx, eval_x, arr, helpers, args.device)
                result["selection"] = dict(best_val_worst_domain_bacc=best[0], best_val_bce=best[1], epochs=len(curves),
                                           best_epoch=int(np.argmax([c["val_worst_domain_bacc"] for c in curves])) + 1)
                if protocol == "main":
                    result["representation"] = representation(cfg, model, idx, eval_x, arr, store, helpers, args.device)
                    torch.save(dict(model=model.state_dict(), config=cfg), run_dir / "checkpoint.pt")
                result["sizes"] = {k: int(len(v)) for k, v in idx.items()}
                result["sampler"] = helpers["sampler"]
                (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
                (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
                (run_dir / "curves.json").write_text(json.dumps(curves))
                np.savez_compressed(run_dir / "predictions.npz", **{f"{k}__{kk}": vv for k, d in preds.items() for kk, vv in d.items()})
                (run_dir / "env.json").write_text(json.dumps(dict(python=platform.python_version(), torch=torch.__version__, device=args.device,
                                                                  seconds=round(time.time() - started, 1))))
                print(f"{name}: val worst-domain bacc {best[0]:.4f} test(clean) bacc "
                      f"{result['test_clean']['at_0.5']['binary']['balanced_accuracy']:.4f} ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
