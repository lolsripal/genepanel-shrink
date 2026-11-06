"""Is the blood-test signal cancer, or where and from whom the blood was drawn?

Platelet RNA studies have known site and batch effects. In GSE89843 only one hospital
(VUMC) contributed both cancer and control samples, and cancer patients are ~17 years
older. In GSE68086 the healthy controls come from three of six processing batches.

Outputs (results/blood/tables/):
  confounders.csv      CV AUC of models that see ONLY metadata (hospital / batch / age / sex /
                       smoking), next to the gene model, on the same patient-grouped splits
  site_signature.csv   can platelet RNA tell which hospital a non-cancer sample came from?
  age_restricted.csv   gene-model AUC re-scored on samples inside the overlapping age range
  benchmark.csv        the original authors' training -> validation split, panel sizes 1..all
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .evaluate import splits
from .models import fit_predict
from .selectors import SELECTORS, univariate
from .stats import bootstrap_ci, corrected_ttest


def _onehot(meta: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = []
    for c in cols:
        v = meta[c]
        if c == "age":
            parts.append(v.fillna(v.median()).to_numpy(float)[:, None])
        else:
            parts.append(pd.get_dummies(v.fillna("NA").astype(str)).to_numpy(float))
    return np.hstack(parts)


def _cv_auc(X, y, groups, cv, seed) -> list[float]:
    out = []
    for _, _, tr, te in splits(y, groups, cv["n_splits"], cv["n_repeats"], seed):
        if len(np.unique(y[te])) < 2:
            continue
        p = fit_predict("logreg", X[tr], y[tr], X[te], seed=seed)
        out.append(roc_auc_score(y[te], p))
    return out


FEATURE_SETS = {
    "GSE89843": {"hospital only": ["hospital"], "age only": ["age"],
                 "age + sex + smoking": ["age", "sex", "smoking"],
                 "hospital + age + sex + smoking": ["hospital", "age", "sex", "smoking"]},
    "GSE68086": {"batch only": ["batch"]},
}


def confounders(items, cfg) -> pd.DataFrame:
    rows = []
    for ds, dcfg, spec in items:
        if ds.source != "platelets" or ds.n_classes != 2:
            continue
        study = spec["study"]
        cv = dcfg["cv"]
        for name, cols in FEATURE_SETS[study].items():
            # a single-valued feature (e.g. hospital inside VUMC) carries no information
            if all(ds.meta[c].nunique(dropna=False) <= 1 for c in cols):
                continue
            aucs = _cv_auc(_onehot(ds.meta, cols), ds.y, ds.groups, cv, dcfg["seed"])
            r = corrected_ttest(np.array(aucs) - 0.5, 1 / cv["n_splits"])
            rows.append({"cancer": ds.cancer, "task": ds.task, "features": name,
                         "auc_mean": np.mean(aucs), "auc_lo": 0.5 + r["lo"], "auc_hi": 0.5 + r["hi"]})
    return pd.DataFrame(rows)


def site_signature(items, cfg) -> pd.DataFrame:
    """Non-cancer samples only: VUMC vs every other hospital, from platelet RNA alone.
    No cancer in either class, so a high AUC here is a pure site/batch effect."""
    rows = []
    for ds, dcfg, spec in items:
        if spec.get("study") != "GSE89843" or ds.task != "nsclc_vs_noncancer" or spec.get("hospital") or spec.get("cohort"):
            continue
        ctrl = ds.y == 0
        m = ds.meta[ctrl]
        y = (m["hospital"] != "VUMC").to_numpy(int)
        X = ds.X[ctrl]
        for k in [None, 10]:
            aucs = []
            for _, _, tr, te in splits(y, ds.groups[ctrl], 5, 2, dcfg["seed"]):
                idx = np.arange(X.shape[1]) if k is None else univariate(X[tr], y[tr])[:k]
                aucs.append(roc_auc_score(y[te], fit_predict("logreg", X[tr][:, idx], y[tr], X[te][:, idx])))
            rows.append({"cancer": ds.cancer, "contrast": "non-cancer: VUMC vs other hospitals",
                         "n_vumc": int((y == 0).sum()), "n_other": int(y.sum()),
                         "genes": "all" if k is None else k, "auc_mean": np.mean(aucs),
                         "auc_min": np.min(aucs), "auc_max": np.max(aucs)})
    return pd.DataFrame(rows)


def age_restricted(cfg, items, lo=45, hi=75) -> pd.DataFrame:
    """Re-score repeat-0 out-of-fold gene-model predictions on patients aged lo..hi, where
    both classes are well represented, so age cannot drive the AUC as much."""
    p = Path(cfg["results_dir"]) / "predictions.parquet"
    if not p.exists():
        return pd.DataFrame()
    pred = pd.read_parquet(p)
    pred = pred[(pred.repeat == 0) & (pred.model == "logreg")]
    rows = []
    for ds, _, spec in items:
        if spec.get("study") != "GSE89843":
            continue
        age = ds.meta["age"].to_numpy()
        for (sel, k), d in pred[(pred.cancer == ds.cancer) & (pred.task == ds.task)].groupby(["selector", "k"]):
            if sel not in ("all_genes", "univariate", "lasso", "stability") or k not in (1, 5, 10, 50, len(ds.genes)):
                continue
            a = age[d["sample"].to_numpy()]
            inside = (a >= lo) & (a <= hi)
            yy, pp = d.y.to_numpy()[inside], d.p.to_numpy()[inside]
            if len(np.unique(yy)) < 2:
                continue
            rows.append({"cancer": ds.cancer, "task": ds.task, "selector": sel, "k": k,
                         "auc_all_ages": roc_auc_score(d.y, d.p), "auc_age_45_75": roc_auc_score(yy, pp),
                         "n_inside": int(inside.sum()), "age_auc_inside": roc_auc_score(yy, a[inside])})
    return pd.DataFrame(rows)


def benchmark(items, cfg, ks=(1, 2, 5, 10, 20, 50, 100, 500)) -> pd.DataFrame:
    """Replay the authors' split: select genes and fit on their training (+ evaluation)
    samples only, then score once on their held-out validation samples."""
    rows = []
    for ds, dcfg, spec in items:
        if spec.get("study") != "GSE89843" or ds.task != "nsclc_vs_noncancer" or spec.get("hospital"):
            continue
        col = "matched_cohort" if spec.get("cohort") else "full_cohort"
        split = ds.meta[col].fillna("NA").to_numpy()
        tr = np.flatnonzero(np.isin(split, ["training", "evaluation"]))
        te = np.flatnonzero(split == "validation")
        Xtr, ytr, Xte, yte = ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]
        pref = univariate(Xtr, ytr)[: dcfg["selection"]["prefilter_k"]]
        rankings = {"univariate": univariate(Xtr, ytr),
                    "lasso": pref[SELECTORS["lasso"](Xtr[:, pref], ytr, seed=0)],
                    "stability": pref[SELECTORS["stability"](Xtr[:, pref], ytr, seed=0, n_subsamples=50)]}
        combos = [("all_genes", np.arange(ds.X.shape[1]))] + [
            (s, r[:k]) for s, r in rankings.items() for k in ks]
        hosp = ds.meta["hospital"].to_numpy()[te]
        train_sites = set(ds.meta["hospital"].to_numpy()[tr][ytr == 1])
        # validation cancers from hospitals that sent cancers to training vs from unseen ones
        subsets = {"all validation": np.ones(len(te), bool),
                   "cancers from training hospitals": (yte == 0) | np.isin(hosp, list(train_sites)),
                   "cancers from unseen hospitals": (yte == 0) | ~np.isin(hosp, list(train_sites))}
        for sel, idx in combos:
            p = fit_predict("logreg", Xtr[:, idx], ytr, Xte[:, idx], seed=0)
            for sub, mask in subsets.items():
                if yte[mask].sum() == 0:
                    continue
                ci = bootstrap_ci(yte[mask], p[mask], roc_auc_score, n_boot=1000)
                rows.append({"cancer": ds.cancer, "cohort": col, "subset": sub, "n_train": len(tr),
                             "n_validation": int(mask.sum()), "n_cancer": int(yte[mask].sum()),
                             "selector": sel, "k": len(idx), "auc": ci["est"],
                             "auc_lo": ci["lo"], "auc_hi": ci["hi"]})
    return pd.DataFrame(rows)


def run(cfg, items):
    out = Path(cfg["results_dir"]) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in [("confounders", lambda: confounders(items, cfg)),
                     ("site_signature", lambda: site_signature(items, cfg)),
                     ("age_restricted", lambda: age_restricted(cfg, items)),
                     ("benchmark", lambda: benchmark(items, cfg))]:
        df = fn()
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows")
