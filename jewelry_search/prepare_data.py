"""Part I -- Offline pipeline: pre-compute catalog embeddings and build the index.

In production you never re-embed the whole catalog on every search. You do it
once, offline, and persist the results. This script:

    1. Finds every image in the dataset.
    2. Passes them (in batches) through MobileNetV2 with the head removed.
    3. L2-normalizes the resulting 1280-dim vectors.
    4. Saves the vectors, a FAISS inner-product index, and a metadata sidecar.

Image paths in the metadata are stored *relative to the repository root* so the
index is portable: it works the same on your laptop and on a Linux deploy host
(Streamlit Cloud), where absolute Windows paths would be meaningless.

Usage:
    python prepare_data.py
    python prepare_data.py --data-dir ../archive/Jewellery_Data --batch-size 32
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
REPO_ROOT = HERE.parent  # the repository root (contains jewelry_search/ + the images)
DEFAULT_OUT_DIR = HERE / "data"


def default_data_dir() -> Path:
    """Locate the image catalog, tolerant of the dataset/ -> archive/ rename."""
    for name in ("archive", "dataset"):
        candidate = REPO_ROOT / name / "Jewellery_Data"
        if candidate.exists():
            return candidate
    return REPO_ROOT / "archive" / "Jewellery_Data"


def rel_to_repo(path: Path) -> str:
    """Path relative to the repo root as a POSIX string (portable across OSes).

    Falls back to an absolute string if the image lives outside the repo, so the
    pipeline still works for ad-hoc catalogs.
    """
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def find_images(data_dir: Path) -> list[Path]:
    """Recursively collect image files under ``data_dir``, sorted for determinism."""
    return sorted(
        p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXT
    )


def batched(items: list, size: int):
    """Yield successive ``size``-length chunks from ``items``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_index(
    data_dir: Path,
    out_dir: Path,
    batch_size: int = 32,
    model=None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Embed every image under ``data_dir`` and write the index to ``out_dir``.

    Importable so the Streamlit app can rebuild the index at runtime if it was
    not shipped with the repo. Pass an already-loaded ``model``/``device`` to
    avoid loading MobileNetV2 twice.

    Returns a small summary dict.
    """
    data_dir = Path(data_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    image_paths = find_images(data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    log(f"Found {len(image_paths)} images under {data_dir}")

    device = device or get_device()
    if model is None:
        log(f"Loading MobileNetV2 feature extractor on {device.type}...")
        model = load_model(device)

    all_feats: list[np.ndarray] = []
    metadata: list[dict] = []
    skipped: list[str] = []
    start_time = time.time()
    processed = 0

    for chunk in batched(image_paths, batch_size):
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
                    "path": rel_to_repo(path),  # portable, repo-relative
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
        if verbose:
            print(f"  embedded {processed}/{len(image_paths)}", end="\r", flush=True)

    log()  # newline after the progress line

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
    labels = [m["label"] for m in metadata]
    label_counts = {lbl: labels.count(lbl) for lbl in sorted(set(labels))}
    config = {
        "model": "mobilenet_v2",
        "weights": "IMAGENET1K_V1",
        "embedding_dim": EMBEDDING_DIM,
        "metric": "cosine (inner product on L2-normalized vectors)",
        "num_images": len(metadata),
        "classes": label_counts,
        "data_dir": rel_to_repo(data_dir),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    log(f"\nDone in {elapsed:.1f}s")
    log(f"  Indexed : {len(metadata)} images -> {embeddings.shape}")
    log(f"  Classes : {label_counts}")
    if skipped:
        log(f"  Skipped : {len(skipped)} unreadable file(s)")
        for line in skipped[:10]:
            log(f"    - {line}")
    log(f"  Output  : {out_dir}")

    return {"num_images": len(metadata), "classes": label_counts, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
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

    try:
        build_index(args.data_dir, args.out_dir, args.batch_size)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
