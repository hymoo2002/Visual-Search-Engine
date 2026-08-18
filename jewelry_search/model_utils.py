"""Shared model and preprocessing utilities for the jewelry visual search engine.

Both the offline pipeline (``prepare_data.py``) and the Streamlit app
(``app.py``) import from here so that the *exact same* preprocessing and
feature extractor are used at index time and at query time. If these two ever
drift apart, query embeddings stop living in the same space as the catalog
embeddings and search quality silently collapses -- so we keep it in one place.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

# MobileNetV2 outputs a 1280-dim vector once the classification head is removed.
EMBEDDING_DIM = 1280

# Standard ImageNet preprocessing expected by the pretrained MobileNetV2 weights.
# Resize the short side to 256, center-crop to 224, then normalize with the
# ImageNet channel statistics.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device | None = None) -> nn.Module:
    """Load MobileNetV2 pretrained on ImageNet with the classifier head removed.

    Replacing ``model.classifier`` with ``nn.Identity`` turns the network into a
    pure feature extractor: the forward pass still runs the conv backbone and the
    global average pool, but returns the 1280-dim pooled feature instead of the
    1000 ImageNet class logits.
    """
    device = device or get_device()
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Identity()
    model.eval()
    model.to(device)
    return model


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image into a normalized CHW float tensor for the model."""
    return _preprocess(img.convert("RGB"))


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row so that inner product == cosine similarity.

    Once every vector has unit length, a FAISS inner-product search returns
    cosine similarity directly, which is the scale-invariant metric we want for
    comparing visual embeddings.
    """
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12  # guard against divide-by-zero
    return vectors / norms


@torch.inference_mode()
def embed_batch(
    model: nn.Module, batch: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Run a preprocessed NCHW batch through the model and return float32 features."""
    batch = batch.to(device)
    feats = model(batch)
    return feats.detach().cpu().numpy().astype("float32")


@torch.inference_mode()
def embed_pil_image(
    model: nn.Module, img: Image.Image, device: torch.device
) -> np.ndarray:
    """Embed a single PIL image and return a unit-length 1-D vector.

    This is the query-time counterpart to the batched catalog embedding used in
    ``prepare_data.py``.
    """
    tensor = preprocess_image(img).unsqueeze(0)  # add batch dimension
    vec = embed_batch(model, tensor, device)
    return l2_normalize(vec)[0]
