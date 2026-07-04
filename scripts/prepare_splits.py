"""
Prepare train/val/test splits across multiple face datasets.

Scans dataset directories, filters subjects by minimum image count,
splits by subject (identity) to prevent data leakage, and outputs
a manifest CSV for PyTorch data loading.

Datasets:
    - Synthetic (subjects_0-1999_72_imgs): 2,000 subjects × 72 images
    - CASIA-WebFace: ~10,575 subjects × variable images

Output:
    - dataset/splits/manifest.csv  (path, subject_id, source, split)
    - dataset/splits/split_config.json (reproducibility metadata)

Usage:
    python scripts/prepare_splits.py
    python scripts/prepare_splits.py --seed 123 --min-images 10
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_RAW = PROJECT_ROOT / "dataset" / "raw"
SPLITS_DIR = PROJECT_ROOT / "dataset" / "splits"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass
class DatasetSource:
    """Configuration for a single dataset source directory."""

    name: str
    path: Path
    id_prefix: str

    def exists(self) -> bool:
        return self.path.is_dir()


@dataclass
class SplitStats:
    """Aggregated statistics for a single split."""

    subjects: int = 0
    images: int = 0
    per_source: dict = field(default_factory=dict)


def get_dataset_sources() -> list[DatasetSource]:
    """Define all dataset sources to scan."""
    return [
        DatasetSource(
            name="synthetic",
            path=DATASET_RAW / "subjects_0-1999_72_imgs",
            id_prefix="synthetic",
        ),
        DatasetSource(
            name="casia",
            path=DATASET_RAW / "casia_webface_extracted",
            id_prefix="casia",
        ),
    ]


def scan_source(source: DatasetSource) -> dict[str, list[Path]]:
    """
    Scan a dataset source directory for subject folders and their images.

    Returns:
        Mapping of prefixed subject_id -> list of image paths.
    """
    if not source.exists():
        logger.warning(
            "Source '%s' not found at %s — skipping", source.name, source.path
        )
        return {}

    subjects: dict[str, list[Path]] = {}

    subject_dirs = sorted(
        [d for d in source.path.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name,
    )

    if not subject_dirs:
        logger.warning("No subject directories found in %s", source.path)
        return {}

    for subject_dir in subject_dirs:
        images = sorted(
            f
            for f in subject_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            subject_id = f"{source.id_prefix}_{subject_dir.name}"
            subjects[subject_id] = images

    logger.info(
        "Source '%s': found %d subjects, %d total images",
        source.name,
        len(subjects),
        sum(len(imgs) for imgs in subjects.values()),
    )
    return subjects


def filter_by_min_images(
    subjects: dict[str, list[Path]], min_images: int
) -> dict[str, list[Path]]:
    """Remove subjects with fewer than min_images."""
    filtered = {sid: imgs for sid, imgs in subjects.items() if len(imgs) >= min_images}
    dropped = len(subjects) - len(filtered)
    if dropped > 0:
        dropped_images = sum(
            len(imgs) for sid, imgs in subjects.items() if len(imgs) < min_images
        )
        logger.info(
            "Filtered out %d subjects (%d images) below threshold of %d images",
            dropped,
            dropped_images,
            min_images,
        )
    return filtered


def split_subjects(
    subject_ids: list[str], rng: np.random.Generator
) -> dict[str, list[str]]:
    """
    Randomly assign subjects to train/val/test splits.

    Split ratios are 80/10/10.
    Shuffle is done in-place on a copy to avoid mutating the input.
    """
    ids = list(subject_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    # test gets the remainder to avoid rounding errors
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def build_manifest(
    sources: list[DatasetSource],
    min_images: int,
    seed: int,
) -> tuple[list[dict], dict[str, SplitStats], dict]:
    """
    Build the complete manifest across all sources.

    Returns:
        (manifest_rows, split_stats, config_metadata)
    """
    rng = np.random.default_rng(seed)
    manifest_rows: list[dict] = []
    overall_stats: dict[str, SplitStats] = {
        split: SplitStats() for split in SPLIT_RATIOS
    }
    config_sources = []

    for source in sources:
        subjects = scan_source(source)
        if not subjects:
            continue

        subjects = filter_by_min_images(subjects, min_images)
        if not subjects:
            logger.warning(
                "Source '%s' has no subjects after filtering — skipping", source.name
            )
            continue

        subject_ids = sorted(subjects.keys())
        splits = split_subjects(subject_ids, rng)

        source_stats = {split: {"subjects": 0, "images": 0} for split in SPLIT_RATIOS}

        for split_name, split_ids in splits.items():
            for subject_id in split_ids:
                images = subjects[subject_id]
                for img_path in images:
                    relative_path = img_path.relative_to(PROJECT_ROOT)
                    manifest_rows.append(
                        {
                            "path": str(relative_path),
                            "subject_id": subject_id,
                            "source": source.name,
                            "split": split_name,
                        }
                    )

                source_stats[split_name]["subjects"] += 1
                source_stats[split_name]["images"] += len(images)

            overall_stats[split_name].subjects += source_stats[split_name]["subjects"]
            overall_stats[split_name].images += source_stats[split_name]["images"]
            overall_stats[split_name].per_source[source.name] = source_stats[split_name]

        config_sources.append(
            {
                "name": source.name,
                "path": str(source.path.relative_to(PROJECT_ROOT)),
                "id_prefix": source.id_prefix,
                "subjects_total": len(subject_ids),
                "images_total": sum(len(subjects[sid]) for sid in subject_ids),
            }
        )

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "min_images_per_subject": min_images,
        "split_ratios": SPLIT_RATIOS,
        "sources": config_sources,
        "splits_summary": {
            name: {
                "subjects": stats.subjects,
                "images": stats.images,
                "per_source": stats.per_source,
            }
            for name, stats in overall_stats.items()
        },
    }

    return manifest_rows, overall_stats, config


def write_manifest(rows: list[dict], output_path: Path) -> None:
    """Write manifest rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["path", "subject_id", "source", "split"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Manifest written to %s (%d rows)", output_path, len(rows))


def write_config(config: dict, output_path: Path) -> None:
    """Write split configuration to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Config written to %s", output_path)


def print_summary(stats: dict[str, SplitStats]) -> None:
    """Print a formatted summary of split statistics."""
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)

    total_subjects = 0
    total_images = 0

    for split_name in ["train", "val", "test"]:
        s = stats[split_name]
        total_subjects += s.subjects
        total_images += s.images

        print(
            f"\n  {split_name.upper():>5}:  {s.subjects:>6} subjects  |  {s.images:>8} images"
        )
        for source_name, source_stats in s.per_source.items():
            print(
                f"         └─ {source_name:<12} "
                f"{source_stats['subjects']:>5} subjects  |  "
                f"{source_stats['images']:>7} images"
            )

    print(
        f"\n  {'TOTAL':>5}:  {total_subjects:>6} subjects  |  {total_images:>8} images"
    )
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test splits for face similarity training."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=5,
        help="Minimum images per subject to include (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SPLITS_DIR,
        help=f"Output directory for manifest and config (default: {SPLITS_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Seed: %d | Min images/subject: %d", args.seed, args.min_images)

    sources = get_dataset_sources()

    available = [s for s in sources if s.exists()]
    if not available:
        logger.error(
            "No dataset sources found. Expected at least one of:\n%s",
            "\n".join(f"  - {s.path}" for s in sources),
        )
        sys.exit(1)

    missing = [s for s in sources if not s.exists()]
    for s in missing:
        logger.warning("Dataset source '%s' not found at %s", s.name, s.path)

    manifest_rows, stats, config = build_manifest(available, args.min_images, args.seed)

    if not manifest_rows:
        logger.error(
            "No images found after filtering. Check dataset paths and thresholds."
        )
        sys.exit(1)

    manifest_path = args.output_dir / "manifest.csv"
    config_path = args.output_dir / "split_config.json"

    write_manifest(manifest_rows, manifest_path)
    write_config(config, config_path)
    print_summary(stats)


if __name__ == "__main__":
    main()
