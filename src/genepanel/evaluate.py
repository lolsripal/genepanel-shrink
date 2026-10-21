"""Repeated, patient-grouped outer CV with feature selection nested inside each fold."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from .models import fit_predict
from .selectors import FULL_SET, SELECTORS, univariate


def metrics(y, p) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "bal_acc": balanced_accuracy_score(y, p >= 0.5),
        "sens_at_95spec": tpr[fpr <= 0.05].max(),
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
        kw = {"drop_frac": sel_cfg["rfe_drop_frac"]} if name == "rfe" else {}
        if name in FULL_SET:
            out[name] = fn(Xtr, ytr, seed=seed, **kw)
        else:
            out[name] = pref[fn(Xtr[:, pref], ytr, seed=seed, **kw)]
    return out


def run_fold(X, y, genes, tr, te, cfg, seed, meta: dict, selectors=SELECTORS):
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    t0 = time.time()
    rankings = rank_all(Xtr, ytr, cfg, seed, selectors)
    t_sel = time.time() - t0
    rows = []

    def ev(selector, k, model, idx, draw=-1):
        p = fit_predict(model, Xtr[:, idx], ytr, Xte[:, idx], seed=seed)
        rows.append({**meta, "selector": selector, "draw": draw, "model": model, "k": k,
                     **metrics(yte, p)})

    all_idx = np.arange(X.shape[1])
    for model in cfg["models"]:
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
    return rows, panels, {**meta, "selection_seconds": t_sel, "total_seconds": time.time() - t0}


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


def run_experiment(datasets, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    res = Path(cfg["results_dir"])
    cache_dir = res / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cv = cfg["cv"]
    jobs = []
    for ds in datasets:
        for r, f, tr, te in splits(ds.y, ds.groups, cv["n_splits"], cv["n_repeats"], cfg["seed"]):
            meta = {"cancer": ds.cancer, "task": ds.task, "repeat": r, "fold": f}
            seed = cfg["seed"] + 1000 * r + f
            cache = cache_dir / f"{ds.cancer}_{ds.task}_r{r}_f{f}.pkl"
            jobs.append(delayed(_job)(ds.X, ds.y, ds.groups, ds.genes, tr, te, cfg, seed, meta, cache))
    out = Parallel(n_jobs=cv["n_jobs"], verbose=10)(jobs)
    metrics_df = pd.DataFrame([row for rows, _, _ in out for row in rows])
    panels_df = pd.DataFrame([p for _, panels, _ in out for p in panels])
    timing_df = pd.DataFrame([t for _, _, t in out])
    metrics_df.to_parquet(res / "metrics.parquet")
    panels_df.to_parquet(res / "panels.parquet")
    timing_df.to_csv(res / "timing.csv", index=False)
    return metrics_df, panels_df, timing_df
