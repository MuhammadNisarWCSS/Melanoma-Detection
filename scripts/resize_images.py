"""Pre-resize the raw ISIC JPEGs into a small training cache.

Roughly half the ISIC 2020 images are 6000x4000 (24 megapixels). Decoding one costs
~300ms of CPU, which starves the GPU when it happens on every epoch for every image.
Resizing once up front turns each read into a ~2ms operation.

The shorter edge is resized to --size (default 448), leaving headroom for the 384px
RandomResizedCrop used by EfficientNet-B4. Aspect ratio is preserved.

Reads:   data/raw/jpeg/train/*.jpg
Writes:  data/processed/jpeg_{size}/*.jpg  (e.g. jpeg_448)

Usage:
    python scripts/resize_images.py
    python scripts/resize_images.py --size 256          # B0 @ 224 cache
    python scripts/resize_images.py --size 448 --workers 8
    python scripts/resize_images.py --overwrite
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

SRC_DIR = Path("data/raw/jpeg/train")
PROCESSED_DIR = Path("data/processed")


def resize_one(job: tuple[Path, Path, int, bool]) -> str:
    """Resize a single JPEG so its shorter edge equals `size`.

    Returns one of 'ok', 'skipped', or 'failed:<reason>'.
    """
    src, dst, size, overwrite = job
    if dst.exists() and not overwrite:
        return "skipped"
    try:
        with Image.open(src) as img:
            # draft() lets libjpeg decode at a reduced DCT scale, so a 24MP source is
            # never fully materialised in memory just to produce a 256px output.
            img.draft("RGB", (size, size))
            img = img.convert("RGB")
            width, height = img.size
            scale = size / min(width, height)
            if scale < 1.0:
                img = img.resize(
                    (round(width * scale), round(height * scale)), Image.BICUBIC
                )
            img.save(dst, "JPEG", quality=92)
        return "ok"
    except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the batch
        return f"failed:{src.name}:{exc}"


def resize_all(src_dir: Path, dst_dir: Path, size: int, workers: int, overwrite: bool) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(
            f"{src_dir} not found. Download and extract the ISIC data first."
        )

    sources = sorted(src_dir.glob("*.jpg"))
    if not sources:
        raise FileNotFoundError(f"No .jpg files found in {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(src, dst_dir / src.name, size, overwrite) for src in sources]

    print(f"Resizing {len(jobs):,} images to {size}px shorter edge using {workers} workers …")
    start = time.perf_counter()
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []

    with Pool(processes=workers) as pool:
        for i, result in enumerate(pool.imap_unordered(resize_one, jobs, chunksize=16), 1):
            if result.startswith("failed"):
                counts["failed"] += 1
                failures.append(result)
            else:
                counts[result] += 1
            if i % 1000 == 0 or i == len(jobs):
                elapsed = time.perf_counter() - start
                rate = i / elapsed
                eta = (len(jobs) - i) / rate if rate else 0
                print(
                    f"  {i:6,}/{len(jobs):,}  {rate:5.0f} img/s  ETA {eta / 60:4.1f} min",
                    flush=True,
                )

    elapsed = time.perf_counter() - start
    print(
        f"\nDone in {elapsed / 60:.1f} min — "
        f"{counts['ok']:,} resized, {counts['skipped']:,} skipped, {counts['failed']:,} failed"
    )
    for failure in failures[:10]:
        print(f"  {failure}")
    if counts["failed"] > 10:
        print(f"  … and {counts['failed'] - 10} more failures")

    print(f"\nResized images written to {dst_dir}")
    print(f"Point training at them with: data.image_dir={dst_dir.as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-resize ISIC images for fast training")
    parser.add_argument("--src-dir", type=Path, default=SRC_DIR)
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/processed/jpeg_{size})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=448,
        help="Target length of the shorter edge (448 for B4@384, 256 for B0@224)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-resize images that already exist"
    )
    args = parser.parse_args()
    dst_dir = args.dst_dir or (PROCESSED_DIR / f"jpeg_{args.size}")

    resize_all(args.src_dir, dst_dir, args.size, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
