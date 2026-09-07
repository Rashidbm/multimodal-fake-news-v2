"""Text normalisation and image fingerprints used for de-duplication.

Two samples are the *same sample* when they have the same normalised text
AND the same image content.  Image content is identified by

* ``sha1`` of the file bytes  -> exact duplicate files (even under different
  names / directories, which happens when the same photo is shipped by
  two datasets), and
* ``dhash`` (difference hash, 64 bit) -> the same picture re-encoded or
  resized.  Two images with identical dhash are treated as the same image
  for *grouping* purposes (they must land in the same split) but not as a
  duplicate *sample* unless the text also matches.

Both are cheap: one file read + one 9x8 grayscale resize per image.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Lower-case, strip accents/punctuation, collapse whitespace.

    'Barack Obama, in Paris!'  ->  'barack obama in paris'
    """
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = t.lower()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def text_key(text: str) -> str:
    """Short stable key for a normalised text (sha1 hex, 16 chars)."""
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def sha1_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash_image(img, hash_size: int = 8) -> str:
    """Difference hash of a PIL image as a 16-char hex string."""
    small = img.convert("L").resize((hash_size + 1, hash_size))
    px = small.tobytes()  # mode 'L': one byte per pixel, row-major
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * (hash_size + 1) + col]
            right = px[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:0{hash_size * hash_size // 4}x}"


@dataclass
class ImageInfo:
    ok: bool
    sha1: str = ""
    dhash: str = ""
    width: int = 0
    height: int = 0
    error: str = ""


def image_info(path: Path, perceptual: bool = True) -> ImageInfo:
    """Fingerprint one image file.  Never raises: a missing or corrupt file
    comes back with ``ok=False`` and the reason in ``error``."""
    p = Path(path)
    if not p.is_file():
        return ImageInfo(ok=False, error="missing")
    try:
        sha = sha1_file(p)
    except OSError as e:  # unreadable
        return ImageInfo(ok=False, error=f"unreadable: {e}")
    if not perceptual:
        return ImageInfo(ok=True, sha1=sha)
    try:
        from PIL import Image

        with Image.open(p) as im:
            im.load()
            w, h = im.size
            dh = dhash_image(im)
        return ImageInfo(ok=True, sha1=sha, dhash=dh, width=w, height=h)
    except Exception as e:  # corrupt / unsupported image
        return ImageInfo(ok=False, sha1=sha, error=f"corrupt: {type(e).__name__}: {e}")


class ImageInfoCache:
    """Memoises ``image_info`` per path so the same file is never hashed twice
    (NewsCLIPpings re-uses the same photo in many pairs)."""

    def __init__(self, perceptual: bool = True):
        self.perceptual = perceptual
        self._cache: dict[str, ImageInfo] = {}

    def get(self, path: str | Path) -> ImageInfo:
        key = str(path)
        info = self._cache.get(key)
        if info is None:
            info = image_info(Path(key), perceptual=self.perceptual)
            self._cache[key] = info
        return info

    def __len__(self) -> int:
        return len(self._cache)
