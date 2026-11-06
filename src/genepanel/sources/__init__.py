"""Data-source registry. A config's `datasets:` entries are dicts with a `source` key;
each source module turns one entry into a `Dataset`."""
from __future__ import annotations

import numpy as np

from .base import Dataset

__all__ = ["Dataset", "load_spec", "specs_from_config"]


def _synthetic(spec: dict, data_dir) -> Dataset:
    """Downloads-free data for CI: `n_informative` genes shifted with the label."""
    rng = np.random.default_rng(spec.get("seed", 0))
    n, p, k = spec.get("n", 300), spec.get("p", 400), spec.get("n_informative", 5)
    n_classes = spec.get("n_classes", 2)
    y = rng.integers(0, n_classes, n)
    X = rng.normal(4, 1, size=(n, p)).astype(np.float32)
    for c in range(1, n_classes):
        X[y == c, (c - 1) * k % p:(c - 1) * k % p + k] += 1.5
    classes = ["normal", "tumor"] if n_classes == 2 else [f"class{c}" for c in range(n_classes)]
    return Dataset(cancer=spec.get("name", "SYNTH"), task=spec["task"], X=X, y=y,
                   groups=np.array([f"P{i}" for i in range(n)]),
                   genes=np.array([f"G{j}" for j in range(p)]),
                   samples=np.array([f"S{i}" for i in range(n)]), classes=classes,
                   source="synthetic")


def load_spec(spec: dict, data_dir) -> Dataset:
    src = spec.get("source", "tcga")
    if src == "tcga":
        from . import tcga
        return tcga.from_spec(spec, data_dir)
    if src == "platelets":
        from . import platelets
        return platelets.load(spec, data_dir)
    if src == "toil":
        from . import toil
        return toil.load(spec, data_dir)
    if src == "synthetic":
        return _synthetic(spec, data_dir)
    raise ValueError(f"unknown source {src!r}")


def specs_from_config(cfg: dict) -> list[dict]:
    """v2 configs list `datasets:`; v1 configs give `cancers` x `tasks` (still supported)."""
    if "datasets" in cfg:
        return [dict(s) for s in cfg["datasets"]]
    return [{"source": "tcga", "kind": "tumor_vs_normal", "cancer": c, "task": t, **(opts or {})}
            for c in cfg["cancers"] for t, opts in cfg["tasks"].items()]
