"""Canonical 224x224 input shared by every arm, plus deterministic corruption/augmentation views.

Pipeline: decode -> RGB (alpha dropped, metadata ignored) -> view corruption at native
resolution -> geometry. Views:

    clean            no corruption
    jpeg75           JPEG quality 75 (fixed robustness view)
    blur1            Gaussian blur radius 1 (fixed robustness view)
    aug1, aug2       training augmentation, seeded per (view, image_id): with p=0.5 a random
                     downscale to 50-100% (random interpolation), then with p=0.5 JPEG quality 60-95

Geometry: 'crop' = resize shorter side to 224 (bicubic) then center crop (the CLIP convention);
'squash' = resize to 224x224 ignoring aspect (the legacy RGB/DCT convention).
"""
from __future__ import annotations

import hashlib
import io
import random

from PIL import Image, ImageFilter

SIZE = 224
AUG_SEED = "image_branch_v2_aug_20261006"
VIEWS = ("clean", "jpeg75", "blur1", "aug1", "aug2")
TRAIN_VIEWS = ("clean", "aug1", "aug2")
EVAL_VIEWS = ("clean", "jpeg75", "blur1")


def load_rgb(path):
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def jpeg(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def view_rng(view, image_id):
    return random.Random(int(hashlib.sha256(f"{AUG_SEED}|{view}|{image_id}".encode()).hexdigest(), 16) % 2**32)


def apply_view(image, view, image_id):
    if view == "clean":
        return image
    if view == "jpeg75":
        return jpeg(image, 75)
    if view == "blur1":
        return image.filter(ImageFilter.GaussianBlur(1))
    if view in ("aug1", "aug2"):
        rng = view_rng(view, image_id)
        if rng.random() < 0.5:
            scale = rng.uniform(0.5, 1.0)
            method = rng.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS])
            image = image.resize((max(32, round(image.width * scale)), max(32, round(image.height * scale))), method)
        if rng.random() < 0.5:
            image = jpeg(image, rng.randint(60, 95))
        return image
    raise ValueError(f"Unknown view {view!r}")


def crop224(image):
    w, h = image.size
    scale = SIZE / min(w, h)
    resized = image.resize((max(SIZE, round(w * scale)), max(SIZE, round(h * scale))), Image.Resampling.BICUBIC)
    left, top = (resized.width - SIZE) // 2, (resized.height - SIZE) // 2
    return resized.crop((left, top, left + SIZE, top + SIZE))


def squash224(image):
    return image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)


GEOMETRY = {"crop": crop224, "squash": squash224}


def canonical(path, view, image_id, geometry="crop"):
    return GEOMETRY[geometry](apply_view(load_rgb(path), view, image_id))
