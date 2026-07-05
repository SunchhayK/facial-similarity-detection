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
MAX_SUBJECTS = 600


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


def cap_images_per_subject(
    subjects: dict[str, list[Path]],
    max_images: int,
    rng: np.random.Generator,
) -> dict[str, list[Path]]:
    """Randomly subsample each subject down to max_images if they exceed it."""
    capped: dict[str, list[Path]] = {}
    total_dropped = 0
    for sid, imgs in subjects.items():
        if len(imgs) > max_images:
            indices = rng.choice(len(imgs), size=max_images, replace=False)
            indices.sort()
            capped[sid] = [imgs[i] for i in indices]
            total_dropped += len(imgs) - max_images
        else:
            capped[sid] = imgs
    if total_dropped > 0:
        logger.info(
            "Capped images per subject to %d (dropped %d images total)",
            max_images,
            total_dropped,
        )
    return capped


def sample_subjects(
    subject_ids: list[str], max_subjects: int, rng: np.random.Generator
) -> list[str]:
    """
    Randomly sample up to max_subjects from the full pool.

    If pool is smaller than max_subjects, use all subjects.
    """
    ids = list(subject_ids)
    rng.shuffle(ids)
    if len(ids) > max_subjects:
        ids = ids[:max_subjects]
        logger.info(
            "Sampled %d subjects from pool of %d", max_subjects, len(subject_ids)
        )
    else:
        logger.info("Pool has %d subjects (≤ %d), using all", len(ids), max_subjects)
    return ids


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


def _extract_source_name(subject_id: str) -> str:
    """Extract source name from prefixed subject_id (e.g. 'casia_001' -> 'casia')."""
    return subject_id.rsplit("_", 1)[0] if "_" in subject_id else subject_id


def build_manifest(
    sources: list[DatasetSource],
    min_images: int,
    seed: int,
    max_subjects: int = MAX_SUBJECTS,
    max_images_per_subject: int | None = None,
) -> tuple[list[dict], dict[str, SplitStats], dict]:
    """
    Build manifest by pooling all sources, sampling max_subjects, then splitting 80/10/10.

    Returns:
        (manifest_rows, split_stats, config_metadata)
    """
    rng = np.random.default_rng(seed)

    # Phase 1: Pool all subjects across sources
    all_subjects: dict[str, list[Path]] = {}
    subject_to_source: dict[str, str] = {}
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

        for sid, imgs in subjects.items():
            all_subjects[sid] = imgs
            subject_to_source[sid] = source.name

        config_sources.append(
            {
                "name": source.name,
                "path": str(source.path.relative_to(PROJECT_ROOT)),
                "id_prefix": source.id_prefix,
                "subjects_total": len(subjects),
                "images_total": sum(len(imgs) for imgs in subjects.values()),
            }
        )

    logger.info(
        "Pooled %d subjects across %d sources", len(all_subjects), len(config_sources)
    )

    # Phase 1.5: Cap images per subject if requested
    if max_images_per_subject is not None:
        all_subjects = cap_images_per_subject(all_subjects, max_images_per_subject, rng)

    # Phase 2: Sample max_subjects from the pool
    all_ids = sorted(all_subjects.keys())
    sampled_ids = sample_subjects(all_ids, max_subjects, rng)

    # Phase 3: Split sampled subjects 80/10/10
    splits = split_subjects(sampled_ids, rng)

    # Phase 4: Build manifest rows and stats
    manifest_rows: list[dict] = []
    overall_stats: dict[str, SplitStats] = {
        split: SplitStats() for split in SPLIT_RATIOS
    }

    for split_name, split_ids in splits.items():
        source_counts: dict[str, dict[str, int]] = {}

        for subject_id in split_ids:
            images = all_subjects[subject_id]
            source_name = subject_to_source[subject_id]

            for img_path in images:
                relative_path = img_path.relative_to(PROJECT_ROOT)
                manifest_rows.append(
                    {
                        "path": str(relative_path),
                        "subject_id": subject_id,
                        "source": source_name,
                        "split": split_name,
                    }
                )

            if source_name not in source_counts:
                source_counts[source_name] = {"subjects": 0, "images": 0}
            source_counts[source_name]["subjects"] += 1
            source_counts[source_name]["images"] += len(images)

        overall_stats[split_name].subjects = sum(
            sc["subjects"] for sc in source_counts.values()
        )
        overall_stats[split_name].images = sum(
            sc["images"] for sc in source_counts.values()
        )
        overall_stats[split_name].per_source = source_counts

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "min_images_per_subject": min_images,
        "max_subjects": max_subjects,
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
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Restrict to specific source names (e.g. --sources synthetic). Default: all available.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=MAX_SUBJECTS,
        help=f"Max subjects to sample from pool (default: {MAX_SUBJECTS})",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max images per subject (randomly subsampled if exceeded). Default: no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Seed: %d | Min images/subject: %d", args.seed, args.min_images)

    sources = get_dataset_sources()

    # Filter by --sources if specified
    if args.sources:
        sources = [s for s in sources if s.name in args.sources]
        unknown = set(args.sources) - {s.name for s in sources}
        if unknown:
            logger.warning("Unknown source names ignored: %s", unknown)

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

    manifest_rows, stats, config = build_manifest(
        available,
        args.min_images,
        args.seed,
        max_subjects=args.max_subjects,
        max_images_per_subject=args.max_images,
    )

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
