"""Statistics used across all chapters.

- Nadeau & Bengio (2003) corrected resampled t-test. Repeated-CV folds share training
  data, so the naive standard error is too small; the correction inflates the variance
  by (1/J + n_test/n_train).
- Bootstrap CIs for metrics on a single external test set.
- Calibration (Brier score, reliability bins) and decision-curve net benefit
  (Vickers & Elkin 2006) for out-of-fold predicted probabilities.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def corrected_ttest(diffs, test_frac: float, alpha: float = 0.05) -> dict:
    """Mean of fold-wise differences with a Nadeau-Bengio corrected CI and p-value.

    `test_frac` = n_test / n for one fold (1 / n_splits), so n_test/n_train =
    test_frac / (1 - test_frac).
    """
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    J = len(d)
    if J < 2:  # e.g. no all-genes reference for this model: nothing to test
        nan = float("nan")
        return {"mean": d.mean() if J else nan, "se": nan, "lo": nan, "hi": nan,
                "t": nan, "p": nan, "n": J}
    mean = d.mean()
    var = d.var(ddof=1) if J > 1 else 0.0
    corr_var = (1 / J + test_frac / (1 - test_frac)) * var
    se = np.sqrt(corr_var)
    tcrit = st.t.ppf(1 - alpha / 2, J - 1) if J > 1 else np.nan
    t = mean / se if se > 0 else (np.inf if mean > 0 else -np.inf if mean < 0 else 0.0)
    p = 2 * st.t.sf(abs(t), J - 1) if J > 1 and np.isfinite(t) else (0.0 if mean != 0 else 1.0)
    return {"mean": mean, "se": se, "lo": mean - tcrit * se, "hi": mean + tcrit * se,
            "t": t, "p": p, "n": J}


def bh_qvalues(p) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate q-values."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def bootstrap_ci(y, p, metric, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI, resampling within each class (keeps prevalence fixed)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    idx_by = [np.flatnonzero(y == c) for c in np.unique(y)]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by])
        vals.append(metric(y[idx], p[idx]))
    vals = np.asarray(vals)
    return {"est": metric(y, p), "lo": np.quantile(vals, alpha / 2),
            "hi": np.quantile(vals, 1 - alpha / 2)}


def brier(y, p) -> float:
    """Binary: mean (p - y)^2. Multiclass (p is n x C): mean squared distance to one-hot."""
    y, p = np.asarray(y), np.asarray(p, float)
    if p.ndim == 1:
        return float(np.mean((p - y) ** 2))
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def calibration_bins(y, p, n_bins: int = 10) -> list[dict]:
    """Reliability diagram data: mean predicted vs observed rate per probability bin."""
    y, p = np.asarray(y), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) | (hi == 1))
        if m.sum():
            out.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                        "mean_pred": float(p[m].mean()), "observed": float(y[m].mean())})
    return out


def net_benefit(y, p, thresholds) -> list[dict]:
    """Decision curve: net benefit of acting on p >= t, vs act-on-all and act-on-none.

    NB(t) = TP/n - FP/n * t/(1-t). It is in units of true positives per patient,
    with false positives weighted by the odds of the threshold.
    """
    y, p = np.asarray(y), np.asarray(p, float)
    n, prev = len(y), y.mean()
    out = []
    for t in thresholds:
        w = t / (1 - t)
        pos = p >= t
        tp, fp = np.sum(pos & (y == 1)), np.sum(pos & (y == 0))
        out.append({"threshold": float(t), "model": tp / n - fp / n * w,
                    "treat_all": prev - (1 - prev) * w, "treat_none": 0.0})
    return out
