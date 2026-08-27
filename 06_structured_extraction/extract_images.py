# extract_images.py - Decode RVL-CDIP invoice images from parquet shards into PNG files

import sys
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

PARQUET_DIR = Path("data/data")
OUTPUT_DIR = Path("data/images")
BATCH_SIZE = 64


def list_shards():
    """Parquet shards in stable filename order."""
    paths = sorted(PARQUET_DIR.glob("train-*.parquet"))
    return paths


def shard_offsets(shard_paths):
    """Starting global index for each shard, so filenames stay unique across shards."""
    offsets = []
    running = 0
    for path in shard_paths:
        parquet_file = pq.ParquetFile(path)
        offsets.append(running)
        running += parquet_file.metadata.num_rows
    result = (offsets, running)
    return result


def extract_shard(task):
    """Decode one shard's image column to PNG. Returns (written, skipped)."""
    shard_path, offset = task
    parquet_file = pq.ParquetFile(shard_path)
    written = 0
    skipped = 0
    index = offset

    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["image"]):
        records = batch.column("image").to_pylist()
        for record in records:
            out_path = OUTPUT_DIR / f"invoice_{index:05d}.png"
            index += 1
            if out_path.exists():
                skipped += 1
                continue
            buffer = BytesIO(record["bytes"])
            image = Image.open(buffer)
            image.save(out_path, format="PNG", optimize=False)
            written += 1

    result = (written, skipped)
    return result


def main():
    shard_paths = list_shards()
    if not shard_paths:
        print(f"No parquet shards found in {PARQUET_DIR}", file=sys.stderr)
        return 1

    offsets, total_rows = shard_offsets(shard_paths)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(shard_paths)} shards, {total_rows} images -> {OUTPUT_DIR}", flush=True)

    tasks = list(zip(shard_paths, offsets))
    total_written = 0
    total_skipped = 0

    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        results = pool.map(extract_shard, tasks)
        for shard_path, (written, skipped) in zip(shard_paths, results):
            total_written += written
            total_skipped += skipped
            print(f"  {shard_path.name}: {written} written, {skipped} skipped", flush=True)

    print(f"Done: {total_written} written, {total_skipped} skipped", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
