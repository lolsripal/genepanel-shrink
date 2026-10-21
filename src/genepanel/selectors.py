"""Feature-ranking methods.

Every selector has the signature ``rank(X, y, seed) -> indices`` and returns column
indices of X ordered from most to least important, so a k-gene panel is just
``ranking[:k]``. Selectors are only ever called on a training split.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def univariate(X, y, seed=0):
    """ANOVA F (equivalent to a two-sample t-test for binary y)."""
    F, _ = f_classif(X, y)
    return np.argsort(-np.nan_to_num(F, nan=-1.0), kind="stable")


def variance(X, y=None, seed=0):
    """Label-free baseline: most variable genes."""
    return np.argsort(-X.var(axis=0), kind="stable")


def lasso(X, y, seed=0, Cs=np.logspace(-3, 1, 40)):
    """Order genes by when they enter the L1-logistic regularization path.

    Genes entering at stronger regularization (smaller C) carry more signal.
    Ties are broken by |coef| at the weakest regularization; genes that never
    enter are appended in univariate order.
    """
    Xs = StandardScaler().fit_transform(X)
    entry = np.full(X.shape[1], len(Cs), dtype=int)
    coef = np.zeros(X.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i, C in enumerate(Cs):
            m = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C,
                                   class_weight="balanced", max_iter=2000, random_state=seed)
            coef = np.abs(m.fit(Xs, y).coef_.ravel())
            entry[(coef > 0) & (entry == len(Cs))] = i
    uni_pos = np.empty(X.shape[1], dtype=int)
    uni_pos[univariate(X, y)] = np.arange(X.shape[1])
    # lexsort: last key is primary
    return np.lexsort((uni_pos, -coef, entry))


def mutual_info(X, y, seed=0):
    mi = mutual_info_classif(X, y, random_state=seed)
    return np.argsort(-mi, kind="stable")


def rfe(X, y, seed=0, drop_frac=0.1):
    """Recursive feature elimination with an L2 logistic regression.

    Drops ``drop_frac`` of the *remaining* genes each round (at least one), so the
    ranking stays fine-grained at small panel sizes, unlike sklearn's RFE whose
    fractional step is relative to the starting feature count.
    """
    Xs = StandardScaler().fit_transform(X)
    remaining = np.arange(X.shape[1])
    eliminated: list[np.ndarray] = []
    while len(remaining) > 1:
        m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        imp = np.abs(m.fit(Xs[:, remaining], y).coef_.ravel())
        n_drop = max(1, int(drop_frac * len(remaining)))
        order = np.argsort(imp, kind="stable")
        eliminated.append(remaining[order[:n_drop]])
        remaining = remaining[order[n_drop:]]
    # last survivor first, then batches in reverse elimination order
    # (within a batch, the gene dropped last, i.e. most important, comes first)
    return np.concatenate([remaining] + [b[::-1] for b in reversed(eliminated)])


def shap_xgb(X, y, seed=0):
    """Mean |SHAP value| from a gradient-boosted tree model."""
    import shap
    from xgboost import XGBClassifier

    pos = y.sum()
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.5, scale_pos_weight=(len(y) - pos) / max(pos, 1),
                      tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)
    m.fit(X, y)
    sv = shap.TreeExplainer(m).shap_values(X)
    imp = np.abs(sv).mean(axis=0)
    # genes never used by the trees all get 0; order those by univariate F
    uni_pos = np.empty(X.shape[1], dtype=int)
    uni_pos[univariate(X, y)] = np.arange(X.shape[1])
    return np.lexsort((uni_pos, -imp))


SELECTORS = {
    "univariate": univariate,
    "variance": variance,
    "lasso": lasso,
    "mutual_info": mutual_info,
    "rfe": rfe,
    "shap": shap_xgb,
}
# Selectors that rank the whole filtered gene set; the rest rank within the
# univariate prefilter (top `prefilter_k` genes), which keeps RFE/SHAP tractable.
FULL_SET = {"univariate", "variance"}
