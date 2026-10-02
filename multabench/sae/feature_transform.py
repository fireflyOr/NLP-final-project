"""Per-column transforms that replace MulTaBench's PCA(30) step for text embeddings.

All transforms are fitted on the training fold's embeddings only and never see the target,
so models using them stay in the Joint Frozen regime.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from multabench.sae.topk_sae import TopKSAE, encode_numpy

DEFAULT_SAE_DIR = os.getenv("MULTABENCH_SAE_DIR", "sae_checkpoints")
SAE_MIN_COUNT = int(os.getenv("MULTABENCH_SAE_MIN_COUNT", "5"))      # latent must fire on >= this many train rows
SAE_MAX_LATENTS = int(os.getenv("MULTABENCH_SAE_MAX_LATENTS", "1024"))  # keep the most frequent latents


class SAEFeatureTransform:
    """E5 embedding -> TopK SAE code -> keep latents that fire often enough on the train fold.

    Latent selection uses activation frequency only (unsupervised): it drops latents that are dead on
    this dataset and caps the column count, so the tabular learner never sees thousands of empty columns.
    """

    def __init__(self, sae: TopKSAE, binary: bool, min_count: int = SAE_MIN_COUNT,
                 max_latents: Optional[int] = SAE_MAX_LATENTS):
        self.sae = sae
        self.binary = binary
        self.min_count = min_count
        self.max_latents = max_latents
        self.keep: Optional[np.ndarray] = None
        self.n_components: Optional[int] = None

    def fit(self, embeddings: np.ndarray) -> "SAEFeatureTransform":
        counts = (encode_numpy(self.sae, embeddings) > 0).sum(axis=0)
        order = np.argsort(-counts, kind="stable")
        order = order[counts[order] >= self.min_count]
        if self.max_latents is not None:
            order = order[: self.max_latents]
        if len(order) == 0:  # degenerate column (e.g. constant text): keep the single most frequent latent
            order = np.argsort(-counts, kind="stable")[:1]
        self.keep = np.sort(order)
        self.n_components = len(self.keep)
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.keep is not None, "call fit() first"
        codes = encode_numpy(self.sae, embeddings)[:, self.keep]
        return (codes > 0).astype(np.float32) if self.binary else codes


class IdentityTransform:
    """Control: the full 384-d E5 embedding with no compression."""

    def __init__(self):
        self.n_components: Optional[int] = None

    def fit(self, embeddings: np.ndarray) -> "IdentityTransform":
        self.n_components = embeddings.shape[1]
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings


_SAE_CACHE: dict = {}


def load_sae_checkpoint(name: str, sae_dir: str = DEFAULT_SAE_DIR) -> TopKSAE:
    path = os.path.join(sae_dir, f"{name}.pt")
    if path not in _SAE_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"SAE checkpoint not found: {path}. Run train_sae.py first "
                                    f"(or set MULTABENCH_SAE_DIR).")
        _SAE_CACHE[path] = TopKSAE.load(path)
    return _SAE_CACHE[path]
