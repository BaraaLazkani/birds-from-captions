"""CUB-200-2011 images paired with Reed et al. (2016) captions.

The dataset ships as two independent archives:

* ``CUB_200_2011.tgz``  -- 11,788 bird photos over 200 species, plus metadata
  (class labels, official split, bounding boxes).
* ``text.zip``          -- 10 human-written captions per image, collected by
  Reed et al. (2016).  These are the attribute-rich descriptions
  ("this bird has a very short orange bill and a black overall body colour")
  that make CUB a text-to-image benchmark rather than a classification one.

This module downloads both, then builds a single flat index of records that the
rest of the project consumes.  Nothing here touches torch -- it is pure
bookkeeping, so it stays fast and testable.
"""

from __future__ import annotations

import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .download import download_file

# Mirrors are tried in order.  The Caltech canonical URL is deliberately absent:
# it sits behind a record page that returns 403 to direct requests.
IMAGE_ARCHIVE_URLS = (
    "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz",
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
)
TEXT_ARCHIVE_URLS = (
    "https://huggingface.co/datasets/alkzar90/CC6204-Hackaton-Cub-Dataset/resolve/main/data/text.zip",
)

N_IMAGES = 11_788
N_CLASSES = 200
CAPTIONS_PER_IMAGE = 10


@dataclass
class CubRecord:
    """One image and everything we know about it."""

    image_id: int
    path: Path
    class_id: int  # 1..200
    class_name: str  # e.g. "Crested_Auklet"
    captions: list[str] = field(default_factory=list)
    # Bounding box in original pixel coordinates: (x, y, w, h).
    bbox: tuple[float, float, float, float] | None = None

    @property
    def is_usable(self) -> bool:
        return bool(self.captions) and self.path.exists()


def _read_pairs(path: Path) -> dict[int, str]:
    """Parse CUB's ``<id> <value>`` metadata files."""
    out: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, _, value = line.strip().partition(" ")
        out[int(key)] = value
    return out


def ensure_downloaded(root: Path) -> tuple[Path, Path]:
    """Fetch and extract both archives if absent.  Returns (image_dir, text_dir).

    Extraction is idempotent and keyed on a sentinel path, so re-running
    ``prepare_data`` after a crash costs nothing.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    image_root = root / "CUB_200_2011"
    if not (image_root / "images.txt").exists():
        archive = download_file(IMAGE_ARCHIVE_URLS, root / "CUB_200_2011.tgz")
        print(f"[cub] extracting {archive.name} ...")
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")

    text_root = root / "text"
    if not text_root.exists():
        archive = download_file(TEXT_ARCHIVE_URLS, root / "cub_text.zip")
        print(f"[cub] extracting {archive.name} ...")
        with zipfile.ZipFile(archive) as zf:
            # Skip macOS resource forks, which would otherwise double the file
            # count and produce empty captions.
            members = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
            zf.extractall(root, members=members)

    return image_root, text_root


def build_index(root: Path, require_captions: bool = True) -> list[CubRecord]:
    """Build the full list of records, joining images to captions and boxes."""
    image_root, text_root = ensure_downloaded(root)

    id_to_relpath = _read_pairs(image_root / "images.txt")
    id_to_class = {k: int(v) for k, v in _read_pairs(image_root / "image_class_labels.txt").items()}
    class_names = {int(k): v.split(".", 1)[1] for k, v in _read_pairs(image_root / "classes.txt").items()}

    boxes: dict[int, tuple[float, float, float, float]] = {}
    box_file = image_root / "bounding_boxes.txt"
    if box_file.exists():
        for line in box_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes[int(parts[0])] = tuple(float(p) for p in parts[1:])  # type: ignore[assignment]

    records: list[CubRecord] = []
    missing_captions = 0

    for image_id, relpath in sorted(id_to_relpath.items()):
        caption_file = text_root / Path(relpath).with_suffix(".txt")
        captions: list[str] = []
        if caption_file.exists():
            captions = [
                line.strip()
                for line in caption_file.read_text(encoding="utf8", errors="ignore").splitlines()
                if line.strip()
            ]
        else:
            missing_captions += 1

        if require_captions and not captions:
            continue

        class_id = id_to_class[image_id]
        records.append(
            CubRecord(
                image_id=image_id,
                path=image_root / "images" / relpath,
                class_id=class_id,
                class_name=class_names[class_id],
                captions=captions,
                bbox=boxes.get(image_id),
            )
        )

    if missing_captions:
        print(f"[cub] warning: {missing_captions} images had no caption file")
    print(
        f"[cub] indexed {len(records)} images, "
        f"{sum(len(r.captions) for r in records)} captions, "
        f"{len({r.class_id for r in records})} species"
    )
    return records


def split_records(def crop_box(
    bbox: tuple[float, float, float, float] | None,
    width: int,
    height: int,
    object_ratio: float = 0.75,
) -> tuple[int, int, int, int]:
    """Square crop centred on the bird, as (left, top, right, bottom).

    CUB birds frequently occupy a small part of a cluttered frame.  Training a
    small generative model on raw frames wastes most of its capacity modelling
    branches and sky.  Following the StackGAN/AttnGAN convention we crop to a
    square around the bounding box, sized so the bird fills roughly
    ``object_ratio`` of the frame, then clamp to the image.

    With no bounding box available this degrades to a centre crop.
    """
    if bbox is None:
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return left, top, left + side, top + side

    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) / max(object_ratio, 1e-3)
    side = min(side, float(min(width, height)))

    left = cx - side / 2.0
    top = cy - side / 2.0
    # Shift (rather than shrink) the window when it runs off an edge, so the
    # crop stays square and keeps the full requested context.
    left = min(max(left, 0.0), width - side)
    top = min(max(top, 0.0), height - side)

    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))
