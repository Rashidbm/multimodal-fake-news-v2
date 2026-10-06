"""Cache frozen-encoder features for the Image Branch v2 comparison.

    # GPU 0 (separate process): current pipeline, CLIP | RGB | DCT = 768 + 2048 + 2048
    set CUDA_VISIBLE_DEVICES=0
    python -m fnd.imagev2.extract --arm A1 --dataset-root D:\\...\\fnd_team_dataset_2026-09-14 \\
        --manifest-dir data/image_branch_v2 --bundle models/image/news/bundle.json --out D:\\...\\features\\A1
    # GPU 1: DINOv3 ViT-L/16 multi-layer tokens
    set CUDA_VISIBLE_DEVICES=1
    python -m fnd.imagev2.extract --arm B1 --model facebook/dinov3-vitl16-pretrain-lvd1689m ...

Arms: A1 (current encoders, canonical crop), A1s (RGB+DCT only, legacy squash geometry, clean view),
B1 (DINOv3). A0 (CLIP only) is the first 768 columns of A1: same canonical pixels, same weights.

View plan (per image id): clean for every row; aug1/aug2 for train rows; jpeg75/blur1 for core
val/test rows. Each view is stored as <view>.npy [N, D] float32 with <view>_ids.npy, written
atomically and skipped when it already exists (resumable). Backbones are frozen; no gradients.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .canonical import canonical
from .manifest import load_joint, sha256_file


class ViewRows(Dataset):
    def __init__(self, rows, root, view, geometry):
        self.rows, self.root, self.view, self.geometry = rows, str(root), view, geometry

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return i, canonical(Path(self.root) / r["image_path"], self.view, r["image_id"], self.geometry)


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def view_plan(rows):
    """view -> sorted row indices."""
    plan = {"clean": list(range(len(rows)))}
    plan["aug1"] = plan["aug2"] = [i for i, r in enumerate(rows) if r["split"] == "train"]
    robust = [i for i, r in enumerate(rows) if r["split"] in ("val", "test") and r["balance_role"] == "core"]
    plan["jpeg75"] = plan["blur1"] = robust
    return plan


class CurrentEncoders:
    """The deployed encoders (frozen). Mirrors ImageBranch.encode_images but accepts separate geometries."""

    def __init__(self, bundle, device):
        from fnd.cache_clip_large import embedding
        from fnd.forensic.inference import ImageBranch
        from fnd.forensic.preprocess import dct_map
        self.branch, self.device, self.embedding, self.dct_map = ImageBranch(bundle, device=device), device, embedding, dct_map
        self.stats = self.branch.kinds["dct_guide"]["dct_stats"]
        self.blocks = [("clip", 768), ("rgb", 2048), ("dct", 2048)]

    @torch.inference_mode()
    def features(self, crops, squashes=None, parts=("clip", "rgb", "dct")):
        b, squashes, out = self.branch, squashes or crops, {}
        if "clip" in parts:
            pixels = torch.stack([b.clip_transform(im) for im in crops]).to(self.device)
            out["clip"] = self.embedding(b.encoders["clip"].get_image_features(pixel_values=pixels))
        if "rgb" in parts:
            pixels = torch.stack([b.rgb_transform(im) for im in squashes]).to(self.device)
            out["rgb"] = b.encoders["rgb_robust"](pixels, True)[1]
        if "dct" in parts:
            maps = np.stack([(self.dct_map(im) - self.stats["mean"]) / (self.stats["std"] + 1e-8) for im in squashes])
            out["dct"] = b.encoders["dct_guide"](torch.from_numpy(maps).float().to(self.device), True)[1]
        return {k: v.float().cpu().numpy() for k, v in out.items()}

    def cat(self, parts):
        return np.concatenate([parts[n] for n, _ in self.blocks if n in parts], axis=1)


class DinoEncoders:
    """DINOv3 (or any HF DINO ViT with register tokens): CLS and mean-patch tokens at several layers."""

    def __init__(self, model_id, device, layers=None, revision=None, model=None):
        from transformers import AutoModel
        model = model if model is not None else AutoModel.from_pretrained(model_id, revision=revision)
        self.model = model.to(device).eval().requires_grad_(False)
        cfg = self.model.config
        n = cfg.num_hidden_layers
        self.layers = layers or [n // 2, round(2 * n / 3), round(5 * n / 6), n]
        self.n_register, self.hidden, self.patch = getattr(cfg, "num_register_tokens", 0), cfg.hidden_size, cfg.patch_size
        self.blocks = [(f"{kind}_L{t}", self.hidden) for t in self.layers for kind in ("cls", "mean")]
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.revision = getattr(cfg, "_commit_hash", None)

    @torch.inference_mode()
    def features(self, crops, **_):
        pixels = torch.from_numpy(np.stack([np.asarray(im, dtype=np.float32) / 255. for im in crops])).permute(0, 3, 1, 2).to(self.device)
        out = self.model(pixel_values=(pixels - self.mean) / self.std, output_hidden_states=True)
        grid = (pixels.shape[-1] // self.patch) ** 2
        parts = []
        for t in self.layers:
            h = out.hidden_states[t]
            if h.shape[1] != 1 + self.n_register + grid or h.shape[-1] != self.hidden:
                raise ValueError(f"Unexpected token layout {tuple(h.shape)} (registers={self.n_register}, patches={grid})")
            parts += [h[:, 0], h[:, 1 + self.n_register:].mean(1)]
        return {"all": torch.cat(parts, 1).float().cpu().numpy()}

    def cat(self, parts):
        return parts["all"]


def environment():
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cuda_visible_devices=__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def run_view(encoder, rows, indices, root, view, geometry, parts, out, workers, batch_size, verify):
    target, ids_path = out / f"{view}.npy", out / f"{view}_ids.npy"
    if target.exists() and ids_path.exists():
        print(f"{view}: exists, skipping", flush=True)
        return
    sel = [rows[i] for i in indices]
    loader = DataLoader(ViewRows(sel, root, view, geometry), batch_size=batch_size, num_workers=workers, collate_fn=collate)
    chunks, started = [], time.time()
    for step, (_, images) in enumerate(loader, 1):
        chunks.append(encoder.cat(encoder.features(images, parts=parts) if parts else encoder.features(images)))
        if verify and step == 1 and hasattr(encoder, "branch"):
            ref = encoder.branch.encode_images(images).float().cpu().numpy()
            if parts == ("clip", "rgb", "dct") and not np.allclose(ref, chunks[0], atol=1e-4):
                raise AssertionError("split encoders disagree with ImageBranch.encode_images")
        if step % 20 == 0 or step == len(loader):
            print(f"{view}: {step}/{len(loader)} batches {time.time() - started:.0f}s", flush=True)
    features = np.concatenate(chunks)
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite features in {view}")
    np.save(out / f"{view}.npy.tmp.npy", features)
    np.save(out / f"{view}_ids.npy.tmp.npy", np.array([r["image_id"] for r in sel]))
    (out / f"{view}.npy.tmp.npy").replace(target)
    (out / f"{view}_ids.npy.tmp.npy").replace(ids_path)
    print(f"{view}: saved {features.shape} in {time.time() - started:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["A1", "A1s", "B1"])
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--manifest-dir", default="data/image_branch_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", help="current-pipeline bundle.json (A arms)")
    ap.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--revision")
    ap.add_argument("--layers", type=int, nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, help="smoke test: first N rows only")
    ap.add_argument("--views", nargs="+", help="restrict views")
    args = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_joint(args.manifest_dir)
    if args.limit:
        rows = rows[:args.limit]
    plan = view_plan(rows)
    geometry, parts = "crop", ("clip", "rgb", "dct")
    if args.arm == "A1s":
        geometry, parts, plan = "squash", ("rgb", "dct"), {"clean": plan["clean"]}
    if args.views:
        plan = {k: v for k, v in plan.items() if k in args.views}
    if args.arm.startswith("A"):
        encoder = CurrentEncoders(args.bundle, args.device)
        provenance = dict(bundle=str(args.bundle), bundle_sha256=sha256_file(args.bundle))
    else:
        encoder = DinoEncoders(args.model, args.device, args.layers, args.revision)
        provenance = dict(model=args.model, revision=encoder.revision, layers=encoder.layers, n_register=encoder.n_register)
        parts = None
    blocks = [(n, d) for n, d in encoder.blocks if parts is None or n in parts]
    meta = dict(arm=args.arm, geometry=geometry, blocks=blocks, dim=sum(d for _, d in blocks), views={k: len(v) for k, v in plan.items()},
                manifest_sha256=sha256_file(Path(args.manifest_dir) / "joint_forensic_all.csv"), provenance=provenance,
                environment=environment(), canonical="shorter side 224 bicubic + center crop" if geometry == "crop" else "224x224 bilinear squash")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    for view, indices in plan.items():
        run_view(encoder, rows, indices, args.dataset_root, view, geometry, parts, out, args.workers, args.batch_size, verify=True)
    (out / "COMPLETE").write_text(json.dumps(meta["views"]))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
