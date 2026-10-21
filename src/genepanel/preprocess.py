"""Label-free gene filtering. Safe to apply before CV because it never looks at y."""
from __future__ import annotations

import numpy as np

from .data import Dataset


def filter_genes(ds: Dataset, min_expr: float = 1.0, min_frac_expressed: float = 0.2,
                 min_std: float = 0.1) -> Dataset:
    X = ds.X
    expressed = (X > min_expr).mean(axis=0) >= min_frac_expressed
    variable = X.std(axis=0) >= min_std
    keep = expressed & variable & np.isfinite(X).all(axis=0)
    ds.X = np.ascontiguousarray(X[:, keep])
    ds.genes = ds.genes[keep]
    return ds
