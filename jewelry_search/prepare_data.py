"""Part I -- Offline pipeline: pre-compute catalog embeddings and build the index.

In production you never re-embed the whole catalog on every search. You do it
once, offline, and persist the results. This script:

    1. Finds every image in the dataset.
    2. Passes them (in batches) through MobileNetV2 with the head removed.
    3. L2-normalizes the resulting 1280-dim vectors.
    4. Saves the vectors, a FAISS inner-product index, and a metadata sidecar.

Usage:
    python prepare_data.py
    python prepare_data.py --data-dir ../dataset/Jewellery_Data --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image, ImageFile

from model_utils import (
    EMBEDDING_DIM,
    embed_batch,
    get_device,
    l2_normalize,
    load_model,
    preprocess_image,
)

# Some catalog JPEGs can be slightly truncated; allow Pillow to load them anyway.
ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Paths are resolved relative to this file so the script works from any CWD.
HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE.parent / "dataset" / "Jewellery_Data"
DEFAULT_OUT_DIR = HERE / "data"


def find_images(data_dir: Path) -> list[Path]:
    """Recursively collect image files under ``data_dir``, sorted for determinism."""
    return sorted(
        p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXT
    )


def batched(items: list, size: int):
    """Yield successive ``size``-length chunks from ``items``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root folder containing the jewelry images (searched recursively).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where to write embeddings.npy, index.faiss and metadata.json.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    image_paths = find_images(data_dir)
    if not image_paths:
        raise SystemExit(f"No images found under {data_dir}")

    print(f"Found {len(image_paths)} images under {data_dir}")

    device = get_device()
    print(f"Loading MobileNetV2 feature extractor on {device.type}...")
    model = load_model(device)

    all_feats: list[np.ndarray] = []
    metadata: list[dict] = []
    skipped: list[str] = []

    start_time = time.time()
    processed = 0

    for chunk in batched(image_paths, args.batch_size):
        tensors = []
        chunk_meta = []
        for path in chunk:
            try:
                with Image.open(path) as img:
                    tensors.append(preprocess_image(img))
            except Exception as exc:  # unreadable / corrupt file -> skip, don't crash
                skipped.append(f"{path.name}: {exc}")
                continue
            # The category is simply the immediate parent folder (necklace / ring).
            chunk_meta.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "label": path.parent.name,
                }
            )

        if not tensors:
            continue

        batch = torch.stack(tensors)
        feats = embed_batch(model, batch, device)
        all_feats.append(feats)
        metadata.extend(chunk_meta)

        processed += len(tensors)
        print(f"  embedded {processed}/{len(image_paths)}", end="\r", flush=True)

    print()  # newline after the progress line

    embeddings = np.vstack(all_feats).astype("float32")
    assert embeddings.shape[1] == EMBEDDING_DIM, (
        f"Expected {EMBEDDING_DIM}-dim features, got {embeddings.shape[1]}"
    )
    embeddings = l2_normalize(embeddings)

    # Inner product on unit vectors == cosine similarity.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    # Persist everything the app / evaluator needs.
    np.save(out_dir / "embeddings.npy", embeddings)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    config = {
        "model": "mobilenet_v2",
        "weights": "IMAGENET1K_V1",
        "embedding_dim": EMBEDDING_DIM,
        "metric": "cosine (inner product on L2-normalized vectors)",
        "num_images": len(metadata),
        "data_dir": str(data_dir),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    labels = [m["label"] for m in metadata]
    label_counts = {lbl: labels.count(lbl) for lbl in sorted(set(labels))}

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Indexed : {len(metadata)} images -> {embeddings.shape}")
    print(f"  Classes : {label_counts}")
    if skipped:
        print(f"  Skipped : {len(skipped)} unreadable file(s)")
        for line in skipped[:10]:
            print(f"    - {line}")
    print(f"  Output  : {out_dir}")


if __name__ == "__main__":
    main()
