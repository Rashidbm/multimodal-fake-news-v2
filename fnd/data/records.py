"""The single record type every raw dataset is converted into.

The three source datasets label things differently (MMFakeBench: four
``fake_cls`` strings plus ``text_source``/``image_source``; DGM4:
``face_swap&text_attribute``-style strings; NewsCLIPpings: a boolean
``falsified``).  Everything downstream (balancing, de-duplication,
splitting, training) only ever sees ``Sample`` and its ``group``.

The five groups, named rather than numbered:

    ooc                   real caption + real photo, but the pairing is wrong
    fake_text_real_image  rumour or edited caption + untouched real photo
    real_text_fake_image  real caption + Photoshopped / AI-generated / AI-edited image
    genuine               real caption + real photo, correct pairing (the only REAL class)
    fake_text_fake_image  rumour or edited caption + generated image

A group is fully determined by three yes/no facts about the pair:
is the text fake, is the image fake, is the pairing wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

GROUPS: tuple[str, ...] = (
    "ooc",
    "fake_text_real_image",
    "real_text_fake_image",
    "genuine",
    "fake_text_fake_image",
)

GROUP_DESCRIPTION: dict[str, str] = {
    "ooc": "real text + real image, out-of-context pairing",
    "fake_text_real_image": "fake text + real image",
    "real_text_fake_image": "real text + fake image",
    "genuine": "real text + real image, genuine pairing",
    "fake_text_fake_image": "fake text + fake image",
}

# group -> (text_fake, image_fake, ooc)
GROUP_FLAGS: dict[str, tuple[int, int, int]] = {
    "ooc": (0, 0, 1),
    "fake_text_real_image": (1, 0, 0),
    "real_text_fake_image": (0, 1, 0),
    "genuine": (0, 0, 0),
    "fake_text_fake_image": (1, 1, 0),
}
_FLAGS_TO_GROUP = {v: k for k, v in GROUP_FLAGS.items()}

# Scenario number exactly as written in the supervisor's instructions (1..5).
SCENARIO_NUMBER: dict[str, int] = {g: i + 1 for i, g in enumerate(GROUPS)}

# Same order, 0-based, used ONLY when a model needs a class id (PyTorch starts at 0).
LABEL_INDEX: dict[str, int] = {g: i for i, g in enumerate(GROUPS)}


def binary_label(group: str) -> int:
    """V1 (FND-CLIP) is a real/fake classifier.  Only ``genuine`` is real;
    an out-of-context pair is fake news even though both halves are
    individually authentic."""
    return 0 if group == "genuine" else 1


def group_from_flags(text_fake: bool, image_fake: bool, ooc: bool) -> Optional[str]:
    """Inverse of GROUP_FLAGS.  None if the triple is not one of the five
    defined groups (e.g. fake text AND out-of-context)."""
    return _FLAGS_TO_GROUP.get((int(text_fake), int(image_fake), int(ooc)))


@dataclass
class Sample:
    """One (text, image) pair with its provenance and group."""

    sample_id: str          # globally unique, prefixed by source
    source: str             # 'mmfakebench' | 'newsclippings' | 'dgm4'
    source_split: str       # split inside the source dataset (val/test/train)
    text: str
    image_path: str         # where the image sits on disk
    group: str              # one of GROUPS
    raw_label: str          # the original label string, kept for auditing
    subcategory: str = ""   # finer origin, e.g. MMFakeBench folder 'fever_AI'
    text_source: str = ""   # where the text came from (MMFakeBench field)
    image_source: str = ""  # where the image came from (MMFakeBench field)
    rule: str = ""          # which mapping rule assigned the group
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.group not in GROUP_FLAGS:
            raise ValueError(f"unknown group {self.group!r} for {self.sample_id}")

    @property
    def text_fake(self) -> int:
        return GROUP_FLAGS[self.group][0]

    @property
    def image_fake(self) -> int:
        return GROUP_FLAGS[self.group][1]

    @property
    def ooc(self) -> int:
        return GROUP_FLAGS[self.group][2]

    @property
    def label_binary(self) -> int:
        return binary_label(self.group)

    @property
    def scenario(self) -> int:
        return SCENARIO_NUMBER[self.group]

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        d.update(
            scenario=self.scenario,
            text_fake=self.text_fake,
            image_fake=self.image_fake,
            ooc=self.ooc,
            label_binary=self.label_binary,
        )
        return d


class MappingError(ValueError):
    """Raised when a raw record cannot be mapped to a group, or when two
    independent signals (fields vs folder) disagree.

    We deliberately crash instead of silently dropping rows: the last
    version of this project lost an entire class that way."""
