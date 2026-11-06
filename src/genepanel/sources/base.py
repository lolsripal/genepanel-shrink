"""Shared container and download helper for every data source."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests


@dataclass
class Dataset:
    cancer: str              # dataset id (kept as "cancer" for v1 compatibility), e.g. BRCA, GSE89843
    task: str
    X: np.ndarray            # samples x genes, float32, log2 scale
    y: np.ndarray            # binary: 1 = cancer; multiclass: 0..C-1 indexing `classes`
    groups: np.ndarray       # patient id, for grouped CV
    genes: np.ndarray
    samples: np.ndarray
    stage: np.ndarray | None = None
    classes: list[str] = field(default_factory=lambda: ["normal", "tumor"])
    meta: pd.DataFrame | None = None   # per-sample covariates (hospital, batch, age, ...)
    source: str = "tcga"

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def fetch(url: str, dest: Path, force: bool = False) -> Path:
    """Stream `url` to `dest` (atomic rename), skipping if it already exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.rename(dest)
    return dest


def encode(labels: pd.Series, order: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """String labels -> (int codes, class names)."""
    classes = order or sorted(labels.unique())
    lut = {c: i for i, c in enumerate(classes)}
    return labels.map(lut).to_numpy(int), list(classes)
