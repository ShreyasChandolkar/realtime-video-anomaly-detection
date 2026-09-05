"""Rotate frame embeddings toward the prompt embeddings.

CLIP-family encoders leave a modality gap: image and text embeddings occupy
different regions of the shared space, so a cosine between them is dominated by
that offset rather than by content. Measured on 484 held-out train clips, naming
a clip by its nearest class prompt scores 37.2%, while naming it by the nearest
*image* centroid scores 56.2% - the space separates these classes far better
than the text side can reach.

A single orthogonal rotation recovers most of that: 52.1% held out, against a
56.2% ceiling. Deliberately a rotation and not a general linear map:

  - It cannot rescale or collapse directions, so it cannot manufacture
    separation that is not there, and it barely overfits (train 68.0% against
    held-out 52.1%, where an unconstrained ridge fit reaches 69.1%/52.9% with
    far more freedom to go wrong).
  - It preserves all distances and angles, so the space keeps its structure and
    a class the rotation never saw still lands somewhere sensible. That matters
    here: the label set is meant to be extensible, and a fitted projection that
    only works for eleven known classes would quietly close that door.

Solved by SVD (Procrustes): R = UV' for U,S,V' = svd(X'Y).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_rotation(frames: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Least-squares rotation carrying `frames` onto `targets`.

    frames:  (N, D) L2-normalised image embeddings
    targets: (N, D) the prompt embedding each frame should sit near
    """
    X = np.asarray(frames, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    return (U @ Vt).astype(np.float32)


def apply_rotation(emb: np.ndarray, R: np.ndarray) -> np.ndarray:
    z = np.asarray(emb, dtype=np.float32) @ R
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def save(R: np.ndarray, path: str | Path, meta: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, R=R, meta=json.dumps(meta or {}))
    return p


def load(path: str | Path) -> tuple[np.ndarray, dict]:
    z = np.load(Path(path), allow_pickle=False)
    return z["R"], json.loads(str(z["meta"]))
