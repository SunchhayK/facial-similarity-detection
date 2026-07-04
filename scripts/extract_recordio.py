"""
Extract images from MXNet RecordIO format into folder-per-subject structure.

Reads CASIA-WebFace .rec/.idx files and extracts face images:
    casia-webface/train.rec  →  casia_webface_extracted/<label>/<index>.jpg

No MXNet dependency — parses the binary RecordIO format directly.

Key format insight: each image's data spans from the current idx offset
to the next idx offset. The RecordIO header's length field only covers
the first chunk, but the raw JPEG data continues beyond it unframed.

Usage:
    python scripts/extract_recordio.py
    python scripts/extract_recordio.py --max-records 100  # test run
"""

import argparse
import logging
import struct
import sys
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "raw" / "casia_webface" / "casia-webface"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "raw" / "casia_webface_extracted"

RECORDIO_MAGIC = 0xCED7230A
RECORDIO_HEADER_SIZE = 8  # 4 magic + 4 lenflag
INSIGHTFACE_HEADER_SIZE = 24  # 4 flag + 4 label + 8 id0 + 8 id1
TOTAL_HEADER_SIZE = RECORDIO_HEADER_SIZE + INSIGHTFACE_HEADER_SIZE  # 32 bytes


def read_idx(idx_path: Path) -> list[tuple[int, int]]:
    """
    Read text-format .idx → list of (key, byte_offset) sorted by key.

    Format: <key>\\t<offset>\\n
    """
    entries = []
    with open(idx_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            entries.append((int(parts[0]), int(parts[1])))

    entries.sort(key=lambda x: x[0])
    return entries


def is_jpeg(data: bytes) -> bool:
    """Check JPEG SOI marker (0xFFD8)."""
    return len(data) >= 2 and data[0:2] == b"\xff\xd8"


def extract_recordio(
    input_dir: Path,
    output_dir: Path,
    max_records: int | None = None,
) -> dict[str, int]:
    """
    Extract images from .rec using .idx offsets.

    Each image's full data spans from current idx offset to the next.
    The first 32 bytes at each offset are headers (8 RecordIO + 24 InsightFace).
    The InsightFace header contains the subject label as a float32.

    Returns:
        Stats dict with counts.
    """
    rec_path = input_dir / "train.rec"
    idx_path = input_dir / "train.idx"

    for required in [rec_path, idx_path]:
        if not required.exists():
            logger.error("Required file not found: %s", required)
            sys.exit(1)

    logger.info("Reading %s...", idx_path.name)
    idx_entries = read_idx(idx_path)
    total = min(len(idx_entries), max_records) if max_records else len(idx_entries)
    logger.info("Extracting %d / %d records", total, len(idx_entries))

    # Get file size for the last record's boundary
    rec_size = rec_path.stat().st_size

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"extracted": 0, "skipped": 0, "errors": 0}
    label_counts: dict[int, int] = {}

    with open(rec_path, "rb") as f:
        for i in tqdm(range(total), desc="Extracting", unit="img"):
            key, offset = idx_entries[i]

            # Boundary = next entry's offset, or EOF for the last record
            if i + 1 < len(idx_entries):
                next_offset = idx_entries[i + 1][1]
            else:
                next_offset = rec_size

            try:
                # Validate RecordIO magic
                f.seek(offset)
                header = f.read(RECORDIO_HEADER_SIZE)
                if len(header) < RECORDIO_HEADER_SIZE:
                    stats["errors"] += 1
                    continue

                magic = struct.unpack("<I", header[:4])[0]
                if magic != RECORDIO_MAGIC:
                    stats["errors"] += 1
                    continue

                # Read InsightFace header (immediately after RecordIO header)
                insight_hdr = f.read(INSIGHTFACE_HEADER_SIZE)
                if len(insight_hdr) < INSIGHTFACE_HEADER_SIZE:
                    stats["errors"] += 1
                    continue

                _flag, label_float = struct.unpack("<If", insight_hdr[:8])
                label = int(label_float)

                # Read full image data: from after headers to next record
                img_size = next_offset - offset - TOTAL_HEADER_SIZE
                if img_size <= 0:
                    stats["skipped"] += 1
                    continue

                image_data = f.read(img_size)

                if not is_jpeg(image_data):
                    stats["skipped"] += 1
                    continue

                # Write to subject folder
                subject_dir = output_dir / str(label)
                subject_dir.mkdir(exist_ok=True)

                img_idx = label_counts.get(label, 0)
                label_counts[label] = img_idx + 1

                img_path = subject_dir / f"{img_idx}.jpg"
                with open(img_path, "wb") as img_file:
                    img_file.write(image_data)

                stats["extracted"] += 1

            except (ValueError, struct.error) as e:
                logger.debug("Error key=%d offset=%d: %s", key, offset, e)
                stats["errors"] += 1

    stats["subjects"] = len(label_counts)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract images from MXNet RecordIO format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input dir with .rec/.idx files (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output dir for extracted images (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Max images to extract (for testing). Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Input:  %s", args.input)
    logger.info("Output: %s", args.output)

    if not args.input.is_dir():
        logger.error("Input directory not found: %s", args.input)
        sys.exit(1)

    stats = extract_recordio(args.input, args.output, args.max_records)

    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"  Extracted: {stats['extracted']:>8} images")
    print(f"  Subjects:  {stats['subjects']:>8}")
    print(f"  Skipped:   {stats['skipped']:>8}")
    print(f"  Errors:    {stats['errors']:>8}")
    print("=" * 50)


if __name__ == "__main__":
    main()
