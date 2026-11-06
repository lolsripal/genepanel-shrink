"""External validation: do small panels survive a change of pipeline and of controls?

Train exactly as in chapter 1 (Xena legacy RSEM, tumor vs *adjacent* normal), then test
the frozen model on the UCSC Toil recompute: TCGA tumors from patients never used in
training vs *healthy* GTEx tissue. Patients are split 5 ways, so no test tumor was seen.

Pipelines differ in units (RSEM normalized counts vs TPM), so each gene is put back on
the training scale with a per-gene mean/SD match fitted on samples that exist in BOTH
pipelines from TRAINING patients only (same tissue, two pipelines). This uses no test
labels. The unharmonized result is reported too.

Output: results/external/tables/external.csv
  setting = internal (Xena held-out patients, adjacent normal: the chapter-1 setting)
            external_harmonized / external_raw (Toil tumors + GTEx healthy)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from . import preprocess
from .models import fit, proba
from .selectors import SELECTORS, univariate
from .sources import tcga, toil


def _sens_spec(y, p, t=0.5):
    pred = p >= t
    return pred[y == 1].mean(), (~pred[y == 0]).mean()


def run_pair(xena: str, tissue: str, cfg) -> list[dict]:
    ecfg, data_dir = cfg["external"], cfg["data_dir"]
    xd = preprocess.filter_genes(tcga.load(xena, "tumor_vs_normal", data_dir), **cfg["preprocess"])
    tumor_t = toil.load({"kind": "tumor_vs_gtex", "tissue": tissue, "task": "x"}, data_dir)
    normal_t = toil.load({"kind": "normal_vs_gtex", "tissue": tissue, "task": "x"}, data_dir)
    genes = sorted(set(xd.genes) & set(tumor_t.genes))
    gx = {g: i for i, g in enumerate(xd.genes)}
    gt = {g: i for i, g in enumerate(tumor_t.genes)}
    Xx = xd.X[:, [gx[g] for g in genes]]
    # Toil samples of the TCGA cohort (tumors + adjacent normals) for harmonization
    tcga_t = np.concatenate([tumor_t.samples[tumor_t.y == 1], normal_t.samples[normal_t.y == 1]])
    Xt_tcga = np.vstack([tumor_t.X[tumor_t.y == 1], normal_t.X[normal_t.y == 1]])[:, [gt[g] for g in genes]]
    t_index = {s: i for i, s in enumerate(tcga_t)}
    # test set: Toil TCGA tumors (positives) and GTEx healthy (negatives)
    Xt = tumor_t.X[:, [gt[g] for g in genes]]
    yt, pt = tumor_t.y, toil.patient(pd.Index(tumor_t.samples))
    gtex = yt == 0

    rows = []
    gkf = GroupKFold(n_splits=ecfg["n_splits"])
    for f, (tr, te) in enumerate(gkf.split(Xx, xd.y, xd.groups)):
        train_pat, test_pat = set(xd.groups[tr]), set(xd.groups[te])
        # harmonization: same samples in both pipelines, training patients only
        x_pos = {s: i for i, s in enumerate(xd.samples)}
        paired = [s for s in xd.samples[tr] if s in t_index]
        A = Xx[[x_pos[s] for s in paired]]
        B = Xt_tcga[[t_index[s] for s in paired]]
        mu_a, sd_a = A.mean(0), A.std(0) + 1e-6
        mu_b, sd_b = B.mean(0), B.std(0) + 1e-6

        def harmonize(X):
            return (X - mu_b) / sd_b * sd_a + mu_a

        # GTEx healthy samples are never trained on; tumors only from held-out patients
        test_mask = gtex | (np.isin(pt, list(test_pat)) & ~np.isin(pt, list(train_pat)))
        Xte_raw, yte = Xt[test_mask], yt[test_mask]
        Xte_h = harmonize(Xte_raw)
        Xtr, ytr = Xx[tr], xd.y[tr]
        pref = univariate(Xtr, ytr)[:cfg["selection"]["prefilter_k"]]
        rankings = {}
        for s in ecfg["selectors"]:
            fn = SELECTORS[s]
            kw = {"drop_frac": cfg["selection"]["rfe_drop_frac"]} if s == "rfe" else {}
            rankings[s] = fn(Xtr, ytr, seed=f) if s == "univariate" else pref[fn(Xtr[:, pref], ytr, seed=f, **kw)]
        combos = [("all_genes", np.arange(len(genes)))] + [
            (s, r[:k]) for s, r in rankings.items() for k in ecfg["panel_sizes"]]
        for sel, idx in combos:
            m = fit("logreg", Xtr[:, idx], ytr, seed=f)
            for setting, Xe, ye in [("internal", Xx[te][:, idx], xd.y[te]),
                                    ("external_harmonized", Xte_h[:, idx], yte),
                                    ("external_raw", Xte_raw[:, idx], yte)]:
                p = proba(m, Xe)
                sens, spec = _sens_spec(ye, p)
                rows.append({"tissue": tissue, "xena": xena, "fold": f, "selector": sel, "k": len(idx),
                             "setting": setting, "auc": roc_auc_score(ye, p), "sens": sens, "spec": spec,
                             "n_pos": int(ye.sum()), "n_neg": int((1 - ye).sum())})
    return rows


def run(cfg):
    out = Path(cfg["results_dir"]) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in cfg["external"]["pairs"]:
        rows += run_pair(pair["xena"], pair["toil_tissue"], cfg)
        print(f"external: {pair['xena']} -> Toil {pair['toil_tissue']} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "external_folds.csv", index=False)
    g = df.groupby(["tissue", "xena", "selector", "k", "setting"])
    summ = g[["auc", "sens", "spec"]].mean().join(g.auc.std().rename("auc_sd")).join(
        g.auc.min().rename("auc_min")).join(g[["n_pos", "n_neg"]].mean()).reset_index()
    summ.to_csv(out / "external.csv", index=False)
    print(f"wrote {out / 'external.csv'}")
