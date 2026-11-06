"""Tests for the v2 additions: multiclass, stability selection, count normalization,
corrected statistics, decision curves, config overrides."""
import numpy as np
import pandas as pd
import pytest

from genepanel import selectors, stats
from genepanel.cli import dataset_config
from genepanel.evaluate import config_hash, metrics, run_fold, splits
from genepanel.sources import load_spec
from genepanel.sources.platelets import log_cpm

CFG = {
    "selection": {"prefilter_k": 60, "selectors": ["univariate", "lasso", "rfe", "shap", "stability"],
                  "panel_sizes": [10, 3], "rfe_drop_frac": 0.2, "stability_subsamples": 8},
    "models": ["logreg", "xgb", "mlp", "rf"],
    "random_baseline": {"n_draws": 2, "models": ["logreg"]},
}


@pytest.fixture(scope="module")
def multi():
    ds = load_spec({"source": "synthetic", "task": "mc", "n": 360, "p": 120, "n_classes": 4,
                    "n_informative": 3}, None)
    return ds


@pytest.mark.parametrize("name", ["univariate", "lasso", "mutual_info", "rfe", "shap", "stability"])
def test_multiclass_selectors_find_class_genes(multi, name):
    r = selectors.SELECTORS[name](multi.X, multi.y, seed=0, **({"n_subsamples": 8} if name == "stability" else {}))
    assert sorted(r) == list(range(multi.X.shape[1]))
    informative = set(range(9))  # classes 1..3 shift genes 0-2, 3-5, 6-8
    assert len(set(r[:15]) & informative) >= 7


def test_multiclass_fold_runs_all_models(multi):
    _, _, tr, te = next(splits(multi.y, multi.groups, 3, 1, 0))
    rows, _, _, pred = run_fold(multi.X, multi.y, multi.genes, tr, te, CFG, 0, {})
    df = pd.DataFrame(rows)
    assert set(df.model) == set(CFG["models"])
    assert df.auc.between(0, 1).all() and df.sens_at_95spec.isna().all()
    assert df[(df.selector == "lasso") & (df.k == 10)].auc.min() > 0.85
    assert pred.empty  # predictions are stored for binary tasks only


def test_binary_fold_stores_predictions():
    ds = load_spec({"source": "synthetic", "task": "b", "n": 200, "p": 80}, None)
    _, _, tr, te = next(splits(ds.y, ds.groups, 4, 1, 0))
    _, _, _, pred = run_fold(ds.X, ds.y, ds.genes, tr, te, {**CFG, "models": ["logreg"]}, 0, {"repeat": 0})
    assert set(pred["sample"]) == set(te) and pred.p.between(0, 1).all()


def test_metrics_multiclass_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    P = np.eye(3)[y] * 0.9 + 0.1 / 3
    m = metrics(y, P, 3)
    assert m["auc"] == 1 and m["bal_acc"] == 1 and np.isnan(m["pr_auc"])


def test_log_cpm_is_per_sample_and_label_free():
    counts = pd.DataFrame([[10, 30, 60], [1, 3, 6], [0, 0, 0]], columns=list("abc"))
    out = log_cpm(counts)
    # first two samples differ only in depth -> identical after CPM
    assert np.allclose(out.iloc[0], out.iloc[1])
    assert (out.iloc[2] == 0).all()
    # adding a sample never changes the others
    assert np.allclose(log_cpm(counts.iloc[:2]).to_numpy(), out.iloc[:2].to_numpy())


def test_corrected_ttest_wider_than_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.02, 25)
    nb = stats.corrected_ttest(d, test_frac=0.2)
    naive_half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    assert (nb["mean"] - nb["lo"]) > naive_half
    # known value: correction factor 1/25 + 0.25 = 0.29 on the variance
    assert np.isclose(nb["se"], np.sqrt(0.29 * d.var(ddof=1)))


def test_corrected_ttest_handles_missing_reference():
    r = stats.corrected_ttest([np.nan, np.nan], test_frac=0.2)
    assert r["n"] == 0 and np.isnan(r["lo"])
    r = stats.corrected_ttest([0.1], test_frac=0.2)
    assert r["n"] == 1 and np.isnan(r["lo"])


def test_net_benefit_known_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.1])
    nb = stats.net_benefit(y, p, [0.5])[0]
    assert np.isclose(nb["model"], 0.5)          # 2 TP / 4, no FP
    assert np.isclose(nb["treat_all"], 0.5 - 0.5)  # prevalence 0.5, odds 1


def test_bh_qvalues_known_values():
    q = stats.bh_qvalues([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5])


def test_bootstrap_ci_contains_estimate():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    ci = stats.bootstrap_ci(y, p, roc_auc_score, n_boot=200)
    assert ci["lo"] <= ci["est"] <= ci["hi"]


def test_dataset_overrides_merge_and_change_cache_key():
    cfg = {"seed": 0, "cv": {"n_splits": 5, "n_repeats": 5, "n_jobs": 8},
           "selection": {"prefilter_k": 2000, "panel_sizes": [10]}, "models": ["logreg"]}
    spec = {"source": "synthetic", "task": "t", "overrides": {"cv": {"n_repeats": 2}}}
    d = dataset_config(cfg, spec)
    assert d["cv"] == {"n_splits": 5, "n_repeats": 2, "n_jobs": 8} and cfg["cv"]["n_repeats"] == 5
    assert config_hash(cfg) != config_hash(d)
    assert config_hash(cfg) == config_hash({**cfg, "cv": {**cfg["cv"], "n_jobs": 1}})
    with pytest.raises(ValueError):
        dataset_config(cfg, {"overrides": {"results_dir": "x"}})
