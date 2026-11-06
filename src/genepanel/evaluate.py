"""Repeated, patient-grouped outer CV with feature selection nested inside each fold."""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold

from .models import fit_predict
from .selectors import FULL_SET, SELECTORS, univariate
from .stats import brier


def metrics(y, p, n_classes: int | None = None) -> dict:
    """Binary (p = P(positive), 1-D) or multiclass (p = n x C probabilities)."""
    p = np.asarray(p)
    if p.ndim == 1:
        fpr, tpr, _ = roc_curve(y, p)
        pred = (p >= 0.5).astype(int)
        return {
            "auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "bal_acc": balanced_accuracy_score(y, pred),
            "sens_at_95spec": tpr[fpr <= 0.05].max(),
            "brier": brier(y, p),
            "f1_macro": f1_score(y, pred, average="macro"),
        }
    labels = np.arange(n_classes or p.shape[1])
    pred = p.argmax(axis=1)
    return {
        "auc": roc_auc_score(y, p, multi_class="ovr", average="macro", labels=labels),
        "pr_auc": np.nan,
        "bal_acc": balanced_accuracy_score(y, pred),
        "sens_at_95spec": np.nan,
        "brier": brier(y, p),
        "f1_macro": f1_score(y, pred, average="macro", labels=labels),
    }


def splits(y, groups, n_splits, n_repeats, seed):
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            yield r, f, tr, te


def rank_all(Xtr, ytr, cfg, seed, selectors=SELECTORS) -> dict[str, np.ndarray]:
    sel_cfg = cfg["selection"]
    pref = univariate(Xtr, ytr)[: sel_cfg["prefilter_k"]]
    out = {}
    for name in sel_cfg["selectors"]:
        fn = selectors[name]
        kw = {}
        if name == "rfe":
            kw["drop_frac"] = sel_cfg["rfe_drop_frac"]
        if name == "stability":
            kw["n_subsamples"] = sel_cfg.get("stability_subsamples", 50)
        if name in FULL_SET:
            out[name] = fn(Xtr, ytr, seed=seed, **kw)
        else:
            out[name] = pref[fn(Xtr[:, pref], ytr, seed=seed, **kw)]
    return out


def run_fold(X, y, genes, tr, te, cfg, seed, meta: dict, selectors=SELECTORS):
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    n_classes = int(y.max()) + 1
    binary = n_classes == 2
    keep_pred = set(cfg.get("prediction_models", ["logreg"])) if binary else set()
    t0 = time.time()
    rankings = rank_all(Xtr, ytr, cfg, seed, selectors)
    t_sel = time.time() - t0
    rows, preds = [], []

    def ev(selector, k, model, idx, draw=-1):
        p = fit_predict(model, Xtr[:, idx], ytr, Xte[:, idx], seed=seed)
        rows.append({**meta, "selector": selector, "draw": draw, "model": model, "k": k,
                     **metrics(yte, p, n_classes)})
        if model in keep_pred and draw < 0:
            preds.append(pd.DataFrame({**meta, "selector": selector, "model": model, "k": k,
                                       "sample": te, "y": yte, "p": p.astype(np.float32)}))

    all_idx = np.arange(X.shape[1])
    for model in cfg.get("all_genes_models", cfg["models"]):
        ev("all_genes", X.shape[1], model, all_idx)
    for sel, ranking in rankings.items():
        for k in cfg["selection"]["panel_sizes"]:
            for model in cfg["models"]:
                ev(sel, k, model, ranking[:k])
    rng = np.random.default_rng(seed)
    rb = cfg["random_baseline"]
    for d in range(rb["n_draws"]):
        for k in cfg["selection"]["panel_sizes"]:
            idx = rng.choice(X.shape[1], size=k, replace=False)
            for model in rb["models"]:
                ev("random", k, model, idx, draw=d)

    kmax = max(cfg["selection"]["panel_sizes"])
    panels = [{**meta, "selector": s, "rank": i, "gene": genes[j]}
              for s, r in rankings.items() for i, j in enumerate(r[:kmax])]
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    timing = {**meta, "selection_seconds": t_sel, "total_seconds": time.time() - t0}
    return rows, panels, timing, pred_df


def _job(X, y, groups, genes, tr, te, cfg, seed, meta, cache: Path):
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    out = run_fold(X, y, genes, tr, te, cfg, seed, meta)
    tmp = cache.with_suffix(".part")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh)
    tmp.rename(cache)
    return out


def config_hash(cfg: dict, spec: dict | None = None) -> str:
    """Hash of everything that changes a fold's results, so stale caches are never reused."""
    keys = ["seed", "preprocess", "cv", "selection", "models", "all_genes_models",
            "random_baseline", "prediction_models"]
    blob = {k: cfg.get(k) for k in keys}
    blob["cv"] = {k: v for k, v in (blob["cv"] or {}).items() if k != "n_jobs"}
    blob["spec"] = spec or {}
    return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:8]


def run_experiment(items, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`items`: list of (Dataset, per-dataset config, spec). Writes metrics/panels/
    predictions/timing to results_dir."""
    res = Path(cfg["results_dir"])
    cache_dir = res / "cache" / "v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ds, dcfg, spec in items:
        cv = dcfg["cv"]
        h = config_hash(dcfg, spec)
        for r, f, tr, te in splits(ds.y, ds.groups, cv["n_splits"], cv["n_repeats"], dcfg["seed"]):
            meta = {"cancer": ds.cancer, "task": ds.task, "repeat": r, "fold": f}
            seed = dcfg["seed"] + 1000 * r + f
            cache = cache_dir / f"{ds.cancer}_{ds.task}_{h}_r{r}_f{f}.pkl"
            jobs.append(delayed(_job)(ds.X, ds.y, ds.groups, ds.genes, tr, te, dcfg, seed, meta, cache))
    out = Parallel(n_jobs=cfg["cv"]["n_jobs"], verbose=5)(jobs)
    metrics_df = pd.DataFrame([row for o in out for row in o[0]])
    panels_df = pd.DataFrame([p for o in out for p in o[1]])
    timing_df = pd.DataFrame([o[2] for o in out])
    preds = [o[3] for o in out if len(o) > 3 and len(o[3])]
    metrics_df.to_parquet(res / "metrics.parquet")
    panels_df.to_parquet(res / "panels.parquet")
    timing_df.to_csv(res / "timing.csv", index=False)
    if preds:
        pd.concat(preds, ignore_index=True).to_parquet(res / "predictions.parquet")
    return metrics_df, panels_df, timing_df
