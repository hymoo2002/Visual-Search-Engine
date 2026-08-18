"""Part III -- The Streamlit application: where the model meets the user.

Run the offline pipeline first (``python prepare_data.py``) to build the index,
then launch the app with:

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from PIL import Image

from model_utils import embed_pil_image, get_device, load_model

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
REPO_ROOT = HERE.parent  # image paths in metadata are stored relative to here


def resolve_image_path(path_str: str) -> Path:
    """Resolve a metadata path (repo-relative) to an absolute path on this host."""
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)

st.set_page_config(page_title="Jewelry Visual Search", page_icon="💎", layout="wide")


@st.cache_resource(show_spinner="Loading model and search index...")
def load_engine():
    """Load the model, FAISS index and metadata exactly once per session.

    ``@st.cache_resource`` is essential here: without it Streamlit would reload
    MobileNetV2 and re-read the index on every widget interaction, making the app
    painfully slow (and risking out-of-memory crashes).
    """
    device = get_device()
    model = load_model(device)

    # Self-heal: if the prebuilt index was not shipped with the repo (e.g. a
    # fresh deploy), build it now from the committed catalog images. Normally
    # the index is committed and this branch is skipped.
    if not (DATA_DIR / "index.faiss").exists():
        from prepare_data import build_index, default_data_dir

        build_index(default_data_dir(), DATA_DIR, model=model, device=device, verbose=False)

    index = faiss.read_index(str(DATA_DIR / "index.faiss"))
    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    return model, index, metadata, device


def search(query_vec: np.ndarray, index, metadata, top_k: int):
    """Return the ``top_k`` nearest catalog items to ``query_vec``.

    ``query_vec`` is a unit-length vector, so the inner-product scores FAISS
    returns are cosine similarities in roughly [0, 1].
    """
    scores, ids = index.search(query_vec.reshape(1, -1), top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:  # FAISS pads with -1 if fewer than top_k items exist
            continue
        item = dict(metadata[idx])
        item["score"] = float(score)
        results.append(item)
    return results


def render_results(results: list[dict], threshold: float, columns: int = 5) -> None:
    """Display matches as a grid, filtered by the similarity threshold."""
    kept = [r for r in results if r["score"] >= threshold]

    if not kept:
        st.warning(
            "No visually similar items found in the catalog above the current "
            "similarity threshold. This query may not be jewelry, or you can "
            "lower the threshold in the sidebar."
        )
        return

    st.success(f"Showing {len(kept)} match(es) above the threshold.")
    for row_start in range(0, len(kept), columns):
        cols = st.columns(columns)
        for col, item in zip(cols, kept[row_start : row_start + columns]):
            with col:
                img_path = resolve_image_path(item["path"])
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.caption("⚠️ image file missing")
                st.caption(f"**{item['label']}** · {item['score']:.3f}")


def main() -> None:
    st.title("💎 Jewelry Visual Search Engine")
    st.write(
        "Upload or snap a photo of a piece of jewelry and retrieve the most "
        "visually similar items from the catalog. Embeddings come from "
        "**MobileNetV2** and search runs on a **FAISS** cosine-similarity index."
    )

    try:
        model, index, metadata, device = load_engine()
    except FileNotFoundError as exc:
        st.error(
            f"Could not load or build the search index: {exc}\n\n"
            "Make sure the catalog images are present, then run "
            "`python prepare_data.py` to build the index."
        )
        st.stop()

    # ---- Sidebar controls -------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of matches", 5, 50, 25, step=5)
        threshold = st.slider(
            "Similarity threshold",
            0.0,
            1.0,
            0.50,
            step=0.01,
            help="Matches below this cosine similarity are hidden. Raise it for "
            "stricter matches; lower it if nothing shows up.",
        )
        st.caption(f"Catalog size: **{index.ntotal}** images")
        st.caption(f"Device: **{device.type}**")

    # ---- Query input ------------------------------------------------------
    tab_upload, tab_camera = st.tabs(["📤 Upload", "📷 Camera"])
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a jewelry image", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
    with tab_camera:
        snapshot = st.camera_input("Take a photo")

    query_file = uploaded or snapshot
    if query_file is None:
        st.info("👆 Upload an image or take a photo to start searching.")
        return

    query_img = Image.open(query_file).convert("RGB")

    left, right = st.columns([1, 3])
    with left:
        st.subheader("Your query")
        st.image(query_img, use_container_width=True)

    with right:
        st.subheader(f"Top {top_k} matches")
        with st.spinner("Searching..."):
            query_vec = embed_pil_image(model, query_img, device)
            results = search(query_vec, index, metadata, top_k)
        render_results(results, threshold)


if __name__ == "__main__":
    main()
