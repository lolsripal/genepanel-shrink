"""Turn raw fold-level metrics into the answers to the research questions.

Outputs (results/tables/):
  summary.csv        mean/sd/CI of every metric per cancer, task, selector, model, k
  retention.csv      smallest panel that is non-inferior to the all-genes model
  vs_random.csv      selected panels vs same-size random panels (logreg)
  stability.csv      mean pairwise Jaccard of fold panels per selector and k
  gene_frequency.csv how often each gene is picked, per selector and k
  consensus.csv      consensus panels (per selector, plus cross-selector top 10)
  transfer.csv       cross-cancer reuse of the consensus 10-gene panels (v1 TCGA only)
  minimal_panel.csv  per task: smallest retained panel, smallest panel reaching 90%
                     sensitivity at 95% specificity, and the all-genes AUC
  calibration.csv    reliability bins + Brier for selected panels (repeat 0 predictions)
  decision_curve.csv net benefit vs threshold for the same panels
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import metrics, splits
from .models import fit_predict
from .stats import brier, calibration_bins, corrected_ttest, net_benefit

ALL_METRICS = ["auc", "pr_auc", "bal_acc", "sens_at_95spec", "brier", "f1_macro"]
KEYS = ["cancer", "task", "selector", "model", "k"]
FOLD = ["repeat", "fold"]
SUPERVISED = ["univariate", "lasso", "mutual_info", "rfe", "shap", "stability"]


def _metrics(m):
    return [c for c in ALL_METRICS if c in m.columns]


def summarize(m: pd.DataFrame) -> pd.DataFrame:
    METRICS = _metrics(m)
    # random: average the draws within a fold first, so n = number of folds everywhere
    per_fold = m.groupby(KEYS + FOLD, as_index=False)[METRICS].mean()
    g = per_fold.groupby(KEYS)[METRICS]
    mean, sd, n = g.mean(), g.std(), g.size()
    out = mean.add_suffix("_mean").join(sd.add_suffix("_sd"))
    for c in METRICS:  # naive, for reference; see retention() for the corrected test
        half = 1.96 * out[f"{c}_sd"] / np.sqrt(n)
        out[f"{c}_lo"], out[f"{c}_hi"] = out[f"{c}_mean"] - half, out[f"{c}_mean"] + half
    out["n_folds"] = n
    return out.reset_index()


def retention(m: pd.DataFrame, delta: float, test_frac=0.2) -> pd.DataFrame:
    """Paired, fold-wise comparison of each panel against all genes (same model, same fold).

    A panel 'retains performance' when the lower 95% bound of the mean AUC difference
    (panel - all genes) is above -delta. The bound uses the Nadeau-Bengio corrected
    resampled t-test, because repeated-CV folds overlap; the naive bound (`diff_lo_naive`)
    is kept for comparison. `test_frac` is 1/n_splits, either one number or a
    {(cancer, task): frac} dict. Reports the smallest retained k.
    """
    m = m[m.selector != "random"]
    ref = (m[m.selector == "all_genes"].set_index(["cancer", "task", "model"] + FOLD)["auc"]
           .rename("auc_all"))
    d = m[m.selector != "all_genes"].join(ref, on=["cancer", "task", "model"] + FOLD)
    d["diff"] = d["auc"] - d["auc_all"]
    g = d.groupby(KEYS)["diff"]
    t = pd.DataFrame({"diff_mean": g.mean(), "diff_sd": g.std(), "n": g.size()})
    t["diff_lo_naive"] = t.diff_mean - 1.96 * t.diff_sd / np.sqrt(t.n)
    frac = (lambda c, tk: test_frac.get((c, tk), 0.2)) if isinstance(test_frac, dict) else (lambda c, tk: test_frac)
    nb = {key: corrected_ttest(grp.to_numpy(), frac(key[0], key[1])) for key, grp in g}
    t["diff_lo"] = [nb[i]["lo"] for i in t.index]
    t["p_corrected"] = [nb[i]["p"] for i in t.index]
    t["retained"] = t.diff_lo > -delta
    t = t.reset_index()
    rows = []
    for key, grp in t.groupby(["cancer", "task", "selector", "model"]):
        ok = grp[grp.retained]
        auc_all = ref.xs(key[:2] + (key[3],), level=["cancer", "task", "model"]).mean()
        rows.append(dict(zip(["cancer", "task", "selector", "model"], key),
                         auc_all_genes=auc_all,
                         smallest_retained_k=int(ok.k.min()) if len(ok) else np.nan,
                         diff_at_10=grp.set_index("k").diff_mean.get(10, np.nan)))
    return pd.DataFrame(rows), t


def vs_random(m: pd.DataFrame) -> pd.DataFrame:
    """Where does each selected panel fall in the null distribution of random panels?

    Null = AUC of each random draw averaged over folds (n_draws values per k).
    """
    rnd = m[m.selector == "random"]
    models = rnd.model.unique()
    null = rnd.groupby(["cancer", "task", "model", "k", "draw"]).auc.mean()
    sel = (m[(m.selector != "random") & m.model.isin(models) & (m.selector != "all_genes")]
           .groupby(KEYS).auc.mean())
    rows = []
    for (c, t, s, mod, k), auc in sel.items():
        nd = null.loc[(c, t, mod, k)].to_numpy()
        rows.append({"cancer": c, "task": t, "selector": s, "model": mod, "k": k,
                     "auc": auc, "random_auc_mean": nd.mean(),
                     "random_auc_q05": np.quantile(nd, 0.05), "random_auc_q95": np.quantile(nd, 0.95),
                     # empirical p with +1 correction
                     "p_random_ge": (1 + (nd >= auc).sum()) / (1 + len(nd))})
    return pd.DataFrame(rows)


def stability(panels: pd.DataFrame, sizes) -> tuple[pd.DataFrame, pd.DataFrame]:
    stab, freq = [], []
    for (c, t, s), grp in panels.groupby(["cancer", "task", "selector"]):
        n_folds = grp.groupby(FOLD).ngroups
        for k in sizes:
            sets = [set(g.gene) for _, g in grp[grp["rank"] < k].groupby(FOLD)]
            jac = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
            stab.append({"cancer": c, "task": t, "selector": s, "k": k,
                         "jaccard_mean": np.mean(jac), "jaccard_sd": np.std(jac)})
            cnt = pd.Series([g for st in sets for g in st]).value_counts() / n_folds
            freq += [{"cancer": c, "task": t, "selector": s, "k": k, "gene": g, "freq": f}
                     for g, f in cnt.items()]
    return pd.DataFrame(stab), pd.DataFrame(freq)


def consensus(freq: pd.DataFrame, min_freq: float) -> pd.DataFrame:
    """Per-selector consensus (genes in >= min_freq of fold panels) and a cross-selector
    10-gene panel: genes ranked by mean selection frequency at k=10 across supervised selectors."""
    per = freq[freq.freq >= min_freq].assign(panel=lambda d: d.selector + "@" + d.k.astype(str))
    per = per[["cancer", "task", "panel", "selector", "k", "gene", "freq"]]
    f10 = freq[(freq.k == 10) & freq.selector.isin(SUPERVISED)]
    n_sel = f10.selector.nunique()
    cross = (f10.groupby(["cancer", "task", "gene"]).freq.sum() / n_sel).rename("freq").reset_index()
    cross = (cross.sort_values(["cancer", "task", "freq"], ascending=[True, True, False])
             .groupby(["cancer", "task"]).head(10)
             .assign(panel="consensus10", selector="consensus", k=10))
    return pd.concat([per, cross[per.columns]], ignore_index=True)


def transfer(datasets, cons: pd.DataFrame, cfg) -> pd.DataFrame:
    """Reuse cancer A's consensus10 panel on cancer B.

    within_cv: retrain logreg on B with A's genes (B's grouped repeated CV).
    direct:    train on all of A, predict B unchanged (same TCGA pipeline/units).
    """
    by = {(d.cancer, d.task): d for d in datasets}
    cv = cfg["cv"]
    rows = []
    for task in {d.task for d in datasets}:
        cancers = sorted(c for c, t in by if t == task)
        for a in cancers:
            for b in cancers:
                da, db = by[(a, task)], by[(b, task)]
                genes = cons[(cons.cancer == a) & (cons.task == task) & (cons.panel == "consensus10")].gene
                genes = [g for g in genes if g in set(db.genes) and g in set(da.genes)]
                ia = [list(da.genes).index(g) for g in genes]
                ib = [list(db.genes).index(g) for g in genes]
                if a != b:
                    p = fit_predict("logreg", da.X[:, ia], da.y, db.X[:, ib], seed=cfg["seed"])
                    rows.append({"task": task, "panel_from": a, "evaluated_on": b, "mode": "direct",
                                 "n_genes": len(genes), **metrics(db.y, p)})
                # a == b is in-sample selection (panel chosen on all of A): optimistic, for reference only
                res = [metrics(db.y[te], fit_predict("logreg", db.X[tr][:, ib], db.y[tr],
                                                     db.X[te][:, ib], seed=cfg["seed"]))
                       for _, _, tr, te in splits(db.y, db.groups, cv["n_splits"], cv["n_repeats"],
                                                  cfg["seed"])]
                rows.append({"task": task, "panel_from": a, "evaluated_on": b,
                             "mode": "within_cv" if a != b else "within_cv_in_sample",
                             "n_genes": len(genes), **pd.DataFrame(res).mean().to_dict()})
    return pd.DataFrame(rows)


def minimal_panel(summary: pd.DataFrame, ret: pd.DataFrame, model="logreg",
                  sens_target=0.9) -> pd.DataFrame:
    """One row per task: how small can the panel get (best supervised selector)?"""
    rows = []
    s = summary[summary.model == model]
    r = ret[(ret.model == model) & ret.selector.isin(SUPERVISED)]
    for (c, t), g in s.groupby(["cancer", "task"]):
        allg = g[g.selector == "all_genes"]
        sel = g[g.selector.isin(SUPERVISED)]
        best_k = sel.groupby("k").auc_mean.max()
        rr = r[(r.cancer == c) & (r.task == t)]
        row = {"cancer": c, "task": t, "model": model,
               "n_genes_all": int(allg.k.iloc[0]) if len(allg) else np.nan,
               "auc_all_genes": allg.auc_mean.iloc[0] if len(allg) else np.nan,
               "auc_best_1": best_k.get(1, np.nan), "auc_best_10": best_k.get(10, np.nan),
               "smallest_retained_k": rr.smallest_retained_k.min() if len(rr) else np.nan}
        rnd = g[g.selector == "random"].set_index("k").auc_mean
        row["auc_random_10"] = rnd.get(10, np.nan)
        if "sens_at_95spec_mean" in sel and sel.sens_at_95spec_mean.notna().any():
            best_sens = sel.groupby("k").sens_at_95spec_mean.max()
            hit = best_sens[best_sens >= sens_target]
            row["smallest_k_sens90"] = hit.index.min() if len(hit) else np.nan
            row["sens_all_genes"] = allg.sens_at_95spec_mean.iloc[0] if len(allg) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_tables(pred: pd.DataFrame, summary: pd.DataFrame, ks=(1, 5, 10, 50),
                       model="logreg") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration and decision curves from repeat-0 out-of-fold predictions (each sample
    predicted exactly once) for the all-genes model and the best selector at each k."""
    pred = pred[(pred.repeat == 0) & (pred.model == model)]
    cal, dca = [], []
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for (c, t), g in pred.groupby(["cancer", "task"]):
        s = summary[(summary.cancer == c) & (summary.task == t) & (summary.model == model)
                    & summary.selector.isin(SUPERVISED)]
        combos = [("all_genes", int(g[g.selector == "all_genes"].k.iloc[0]))] if (g.selector == "all_genes").any() else []
        for k in ks:
            sk = s[s.k == k]
            if len(sk):
                combos.append((sk.loc[sk.auc_mean.idxmax(), "selector"], k))
        for sel, k in combos:
            d = g[(g.selector == sel) & (g.k == k)]
            if d.empty:
                continue
            base = {"cancer": c, "task": t, "selector": sel, "k": k, "brier": brier(d.y, d.p),
                    "prevalence": d.y.mean()}
            cal += [{**base, **b} for b in calibration_bins(d.y, d.p)]
            dca += [{**base, **b} for b in net_benefit(d.y.to_numpy(), d.p.to_numpy(), thresholds)]
    return pd.DataFrame(cal), pd.DataFrame(dca)


def _is_v1(items) -> bool:
    return (len({ds.cancer for ds, _, _ in items}) > 1
            and all(spec.get("source", "tcga") == "tcga" and spec.get("kind", "tumor_vs_normal") == "tumor_vs_normal"
                    for _, _, spec in items))


def run(cfg, items):
    """`items`: [(Dataset, per-dataset config, spec)] from cli.load_items (a plain list of
    Datasets is accepted too, for v1 callers)."""
    if items and not isinstance(items[0], tuple):
        items = [(ds, cfg, {"source": "tcga"}) for ds in items]
    datasets = [ds for ds, _, _ in items]
    res = Path(cfg["results_dir"])
    out = res / "tables"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(res / "metrics.parquet")
    panels = pd.read_parquet(res / "panels.parquet")
    a = cfg["analysis"]

    summary = summarize(m)
    summary.to_csv(out / "summary.csv", index=False)
    test_frac = {(ds.cancer, ds.task): 1 / dcfg["cv"]["n_splits"] for ds, dcfg, _ in items}
    ret, ret_detail = retention(m, a["delta_auc"], test_frac)
    ret.to_csv(out / "retention.csv", index=False)
    ret_detail.to_csv(out / "retention_detail.csv", index=False)
    vs_random(m).to_csv(out / "vs_random.csv", index=False)
    sizes = sorted({k for _, dcfg, _ in items for k in dcfg["selection"]["panel_sizes"]})
    stab, freq = stability(panels, sizes)
    stab.to_csv(out / "stability.csv", index=False)
    freq.to_csv(out / "gene_frequency.csv", index=False)
    cons = consensus(freq, a["consensus_freq"])
    cons.to_csv(out / "consensus.csv", index=False)
    minimal_panel(summary, ret, sens_target=a.get("sens_target", 0.9)).to_csv(
        out / "minimal_panel.csv", index=False)
    if (res / "predictions.parquet").exists():
        cal, dca = calibration_tables(pd.read_parquet(res / "predictions.parquet"), summary)
        cal.to_csv(out / "calibration.csv", index=False)
        dca.to_csv(out / "decision_curve.csv", index=False)
    if _is_v1(items):
        transfer(datasets, cons, cfg).to_csv(out / "transfer.csv", index=False)
    print(f"wrote tables to {out}")
