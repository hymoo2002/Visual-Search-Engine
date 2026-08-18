"""Part II (Bonus) -- Quality check: how good are the retrieved matches?

Returning 25 images is easy; returning *relevant* ones is the hard part. We use
each image's category folder (necklace / ring) as a ground-truth relevance
label: a retrieved item is "relevant" if it shares the query's category.

For every catalog image we treat it as a query, retrieve its nearest neighbors
(excluding itself), and compute standard retrieval metrics:

    * Precision@K  -- fraction of the top-K that share the query's label
    * Recall@K     -- fraction of all same-label items found in the top-K
    * mAP          -- mean average precision over the full ranking

Usage:
    python evaluate.py
    python evaluate.py --k 5 10 25
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def average_precision(relevant: np.ndarray) -> float:
    """Average precision for one ranked list of boolean relevance flags."""
    if not relevant.any():
        return 0.0
    hits = np.cumsum(relevant)
    ranks = np.arange(1, len(relevant) + 1)
    precision_at_hits = (hits / ranks)[relevant]
    return float(precision_at_hits.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1, 5, 10, 25], help="K values to report."
    )
    args = parser.parse_args()

    emb_path = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "metadata.json"
    if not emb_path.exists() or not meta_path.exists():
        raise SystemExit("Missing embeddings/metadata. Run prepare_data.py first.")

    embeddings = np.load(emb_path).astype("float32")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    labels = np.array([m["label"] for m in metadata])
    n = len(labels)
    print(f"Evaluating retrieval over {n} images...\n")

    # Full cosine-similarity matrix (vectors are already L2-normalized).
    # 490x490 is tiny, so a dense matmul is the simplest correct approach.
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)  # never let an image retrieve itself

    # Rank neighbors once (descending similarity) and reuse for every K.
    ranking = np.argsort(-sims, axis=1)
    ranked_labels = labels[ranking]  # shape (n, n-1 usable)
    query_labels = labels[:, None]
    relevant_matrix = ranked_labels == query_labels  # boolean relevance grid

    # Total number of relevant items available per query (excludes the query).
    total_relevant = np.array([(labels == lbl).sum() - 1 for lbl in labels])

    max_k = min(max(args.k), n - 1)
    print(f"{'Metric':<14}{'Score':>8}")
    print("-" * 22)

    per_class_p = defaultdict(list)
    for k in args.k:
        if k > n - 1:
            continue
        topk = relevant_matrix[:, :k]
        precision_at_k = topk.sum(axis=1) / k
        recall_at_k = topk.sum(axis=1) / np.clip(total_relevant, 1, None)
        print(f"P@{k:<11}{precision_at_k.mean():>8.3f}")
        print(f"R@{k:<11}{recall_at_k.mean():>8.3f}")
        if k == min(args.k, key=lambda x: abs(x - 5)):  # nearest K to 5
            for lbl, p in zip(labels, precision_at_k):
                per_class_p[lbl].append(p)

    # Mean average precision over the entire ranking.
    ap = np.array([average_precision(relevant_matrix[i]) for i in range(n)])
    print(f"{'mAP':<14}{ap.mean():>8.3f}")

    # Per-category Precision@~5 breakdown.
    if per_class_p:
        k_used = min(args.k, key=lambda x: abs(x - 5))
        print(f"\nPer-category Precision@{k_used}:")
        for lbl in sorted(per_class_p):
            vals = per_class_p[lbl]
            print(f"  {lbl:<12}{np.mean(vals):.3f}  (n={len(vals)})")

    print(
        "\nNote: relevance here means 'same category'. It rewards category "
        "consistency but not fine-grained style match -- see the README's "
        "discussion of the semantic gap."
    )


if __name__ == "__main__":
    main()
