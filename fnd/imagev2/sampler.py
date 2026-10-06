"""Domain- and class-balanced epoch sampler.

Cells are (domain, class[, news-real sub-source]). Per epoch every cell contributes a fixed quota
of draws (without replacement when its pool is large enough, otherwise tiled permutations):

  * class shares: REAL 50% / AI 35% / MANIPULATED 15% (REAL 50% / AI 50% when no MANIPULATED rows)
  * AI is split across domains in proportion to its core counts; MANIPULATED is the fakeddit domain
  * REAL mirrors the positives of each domain 1:1, so inside every domain the classes are balanced
    and the domain itself carries no label information
  * real news images alternate 50/50 between the JPEG (NewsCLIPpings) and PNG (MMFB) sub-sources,
    so file format carries no label information either
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def cell_key(domain, y3, subsource):
    if y3 == 0:
        return (domain, "REAL", "nc_jpeg" if (domain == "news" and subsource == "nc_jpeg") else ("png" if domain == "news" else ""))
    return (domain, "AI" if y3 == 1 else "MAN", "")


class DomainClassSampler:
    def __init__(self, y3, domain, subsource, role, class_share=(0.5, 0.35, 0.15)):
        self.keys = [cell_key(d, int(y), s) for d, y, s in zip(domain, y3, subsource)]
        self.pools = {}
        for i, k in enumerate(self.keys):
            self.pools.setdefault(k, []).append(i)
        self.pools = {k: np.array(v) for k, v in self.pools.items()}
        core_pos = Counter(k[:2] for k, r in zip(self.keys, role) if k[1] != "REAL" and r == "core")
        has_man = any(k[1] == "MAN" for k in self.pools)
        share = {"AI": class_share[1] if has_man else 0.5, "MAN": class_share[2] if has_man else 0.0}
        self.quota = {}
        for cls in ("AI", "MAN"):
            total = sum(v for (d, c), v in core_pos.items() if c == cls)
            for (d, c), v in core_pos.items():
                if c != cls:
                    continue
                q = share[cls] * v / total
                self.quota[(d, cls, "")] = q
                if d == "news":
                    self.quota[(d, "REAL", "nc_jpeg")] = self.quota[(d, "REAL", "png")] = q / 2
                else:
                    self.quota[(d, "REAL", "")] = q
        missing = [k for k, q in self.quota.items() if q > 0 and k not in self.pools]
        if missing:
            raise ValueError(f"empty sampling pools for {missing}")

    def draw(self, rng, n):
        cells = sorted(k for k, q in self.quota.items() if q > 0)
        counts = {k: int(round(self.quota[k] * n)) for k in cells}
        counts[cells[0]] += n - sum(counts.values())
        picks = []
        for k in cells:
            pool, m = self.pools[k], counts[k]
            reps = -(-m // len(pool))
            picks.append(np.concatenate([rng.permutation(pool) for _ in range(reps)])[:m])
        out = np.concatenate(picks)
        return out[rng.permutation(len(out))]

    def report(self, n):
        rows = []
        for k in sorted(self.quota):
            pool = len(self.pools[k])
            drawn = round(self.quota[k] * n)
            rows.append(dict(domain=k[0], cls=k[1], sub=k[2], pool=pool, share=round(self.quota[k], 4), draws_per_epoch=drawn,
                             repeats_per_epoch=round(drawn / pool, 2)))
        return rows
