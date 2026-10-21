import numpy as np
import pandas as pd
import pytest

from genepanel import selectors
from genepanel.data import collapse_stage
from genepanel.evaluate import metrics, rank_all, run_fold, splits

CFG = {
    "selection": {"prefilter_k": 60, "selectors": list(selectors.SELECTORS),
                  "panel_sizes": [20, 5, 1], "rfe_drop_frac": 0.2},
    "models": ["logreg"],
    "random_baseline": {"n_draws": 2, "models": ["logreg"]},
}
INFORMATIVE = np.arange(5)


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(0)
    n, p = 300, 200
    y = (rng.random(n) < 0.7).astype(int)
    X = rng.normal(size=(n, p)).astype(np.float32)
    X[:, INFORMATIVE] += 1.5 * y[:, None]          # 5 planted informative genes
    groups = np.array([f"P{i // 2}" for i in range(n)])  # paired samples per patient
    return X, y, groups


@pytest.mark.parametrize("name", list(selectors.SELECTORS))
def test_rankings_are_permutations(synth, name):
    X, y, _ = synth
    r = selectors.SELECTORS[name](X, y, seed=0)
    assert sorted(r) == list(range(X.shape[1]))


@pytest.mark.parametrize("name", ["univariate", "lasso", "mutual_info", "rfe", "shap"])
def test_supervised_selectors_recover_planted_genes(synth, name):
    X, y, _ = synth
    top = set(selectors.SELECTORS[name](X, y, seed=0)[:10])
    assert len(top & set(INFORMATIVE)) >= 4


def test_rank_all_panels_have_k_unique_genes(synth):
    X, y, _ = synth
    for name, r in rank_all(X, y, CFG, seed=0).items():
        for k in CFG["selection"]["panel_sizes"]:
            assert len(set(r[:k])) == k, name


def test_selectors_never_see_test_rows(synth):
    """Feature selection must happen inside the fold: wrap every selector and check
    the rows it receives are exactly the training rows."""
    X, y, groups = synth
    X = X.copy()
    X[:, -1] = np.arange(len(y))  # tag column: row id
    _, _, tr, te = next(splits(y, groups, 3, 1, 0))
    seen = []

    def spy(fn):
        def wrapped(Xs, ys, seed=0, **kw):
            seen.append(Xs.shape[0])
            return fn(Xs, ys, seed=seed, **kw)
        return wrapped

    spies = {k: spy(v) for k, v in selectors.SELECTORS.items()}
    # prefiltered selectors drop the tag column, so check row counts plus tag for full-set ones
    tag_rows = []
    spies["variance"] = lambda Xs, ys=None, seed=0: (tag_rows.append(Xs[:, -1].copy()),
                                                     selectors.variance(Xs))[1]
    rows, panels, _ = run_fold(X, y, np.array([f"g{i}" for i in range(X.shape[1])]),
                               tr, te, CFG, 0, {}, selectors=spies)
    assert seen and all(n == len(tr) for n in seen)
    assert set(tag_rows[0].astype(int)) == set(tr)
    assert not set(tag_rows[0].astype(int)) & set(te)


def test_grouped_splits_keep_patients_together(synth):
    _, y, groups = synth
    for _, _, tr, te in splits(y, groups, 5, 2, 0):
        assert not set(groups[tr]) & set(groups[te])


def test_run_fold_rows(synth):
    X, y, groups = synth
    _, _, tr, te = next(splits(y, groups, 3, 1, 0))
    rows, panels, _ = run_fold(X, y, np.array([f"g{i}" for i in range(X.shape[1])]),
                               tr, te, CFG, 0, {"fold": 0})
    df = pd.DataFrame(rows)
    n_sel = len(CFG["selection"]["selectors"]) * len(CFG["selection"]["panel_sizes"])
    n_rand = CFG["random_baseline"]["n_draws"] * len(CFG["selection"]["panel_sizes"])
    assert len(df) == 1 + n_sel + n_rand
    # planted signal is easy: selected 5-gene panels should beat random 5-gene panels
    sel5 = df[(df.selector == "lasso") & (df.k == 5)].auc.mean()
    rnd5 = df[(df.selector == "random") & (df.k == 5)].auc.mean()
    assert sel5 > 0.9 and sel5 > rnd5 + 0.1


def test_metrics_perfect():
    m = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc"] == 1 and m["sens_at_95spec"] == 1 and m["bal_acc"] == 1


def test_collapse_stage():
    s = pd.Series(["Stage IA", "Stage IIIC", "Stage IV", "Stage X", "[Discrepancy]", None, "Stage II"])
    assert collapse_stage(s).tolist() == ["I", "III", "IV", "NA", "NA", "NA", "II"]
