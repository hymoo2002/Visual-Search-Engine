# 💎 Jewelry Visual Search Engine

An end-to-end **visual similarity search** tool built on the Tanishq Jewellery
dataset. Upload (or snap) a photo of a piece of jewelry and the app returns the
most visually similar items from the catalog.

The pipeline uses **transfer learning** (MobileNetV2 pretrained on ImageNet, with
its classification head removed) to turn each image into a 1280-dimensional
embedding, then **FAISS** for fast cosine-similarity search, all wrapped in a
**Streamlit** UI.

## Project structure

```
jewelry_search/
├── data/              # Generated: embeddings.npy, index.faiss, metadata.json, config.json
├── model_utils.py     # Shared model + preprocessing (imported by every script)
├── prepare_data.py    # Part I  — offline pipeline: embed catalog, build index
├── evaluate.py        # Part II — retrieval quality check (Precision@K, Recall@K, mAP)
├── app.py             # Part III — the Streamlit web application
├── requirements.txt   # torch, torchvision, faiss-cpu, streamlit, numpy, pillow
└── README.md
```

The image catalog lives one level up, at `../dataset/Jewellery_Data/{necklace,ring}/`.

## Setup

```bash
pip install -r requirements.txt
```

If the CPU builds of torch/torchvision don't resolve on your platform, install
them from the PyTorch index first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## How to run

**1. Build the index (run once, offline):**

```bash
python prepare_data.py
```

This embeds all 490 catalog images and writes `data/embeddings.npy`,
`data/index.faiss`, and `data/metadata.json`. Takes ~20s on CPU.

**2. (Optional) Check retrieval quality:**

```bash
python evaluate.py
```

**3. Launch the app:**

```bash
streamlit run app.py
```

Then open http://localhost:8501, upload a jewelry photo, and browse the matches.
Use the sidebar to change the number of matches and the similarity threshold.

## How it works

| Stage | What happens |
|-------|--------------|
| **Preprocess** | Resize→256, center-crop→224, normalize with ImageNet mean/std. |
| **Embed** | MobileNetV2 (head removed) → 1280-dim feature vector. |
| **Normalize** | L2-normalize so inner product = cosine similarity. |
| **Index** | `faiss.IndexFlatIP` — exact inner-product search over the catalog. |
| **Search** | Embed the query the same way, retrieve the top-K nearest vectors. |
| **Threshold** | If the best match is below the similarity threshold, warn that nothing relevant was found (e.g. someone searches a *phone*). |

**Why cache?** `app.py` loads the model and index inside `@st.cache_resource`, so
MobileNetV2 is loaded exactly once instead of on every interaction — without it
the app would be unusably slow.

## Part II — Evaluation results

We use each image's category folder (`necklace` / `ring`) as ground-truth
relevance: a retrieved item counts as relevant if it shares the query's
category. Every catalog image is used as a query (excluding itself from its own
results).

| Metric | Score |
|--------|-------|
| Precision@1  | 0.994 |
| Precision@5  | 0.991 |
| Precision@10 | 0.987 |
| Precision@25 | 0.983 |
| mAP          | 0.923 |

Per-category Precision@5: **necklace 0.995**, **ring 0.985**.

Precision is very high — the off-the-shelf MobileNetV2 features already separate
necklaces from rings almost perfectly. (Recall@K is low simply because there are
hundreds of relevant items per class but only K slots.)

## Discussion questions

**The semantic gap.** If a user queries a *silver* ring and the system returns a
*gold* ring with the identical diamond setting, is that "relevant"? It depends on
the user's intent. Our embeddings capture **visual form** (shape, setting,
texture, composition) far more strongly than fine-grained **material/color**, so
the system will happily match on setting while ignoring metal color. For a
*"find this exact product"* intent that's a miss; for a *"find items in this
style"* intent it's a good result. Category-based Precision@K can't see this
distinction at all — it would score both as relevant — which is exactly the
semantic gap between what we measure and what the user means.

**Improving precision.** If Precision@K were low, options include:

- **Stronger backbone** — swap MobileNetV2 for a Vision Transformer (ViT/DINOv2)
  or a CLIP image encoder for richer, more discriminative embeddings.
- **Metric fine-tuning** — train with a triplet / contrastive loss so
  same-style items are pulled together and different styles pushed apart.
- **Attribute conditioning** — add color/material features (e.g. color
  histograms) to close the silver-vs-gold gap.
- **Re-ranking** — retrieve a broad candidate set with FAISS, then re-rank the
  top candidates with a heavier, more precise model.
