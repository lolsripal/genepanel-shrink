"""The four classifiers, with fixed modest hyperparameters and class-imbalance handling."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def make_model(name: str, y: np.ndarray, seed: int = 0):
    if name == "logreg":
        # only model with tuning: C by inner 3-fold CV on the training split
        clf = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=3, scoring="roc_auc", l1_ratios=(0.0,),
                                   class_weight="balanced", max_iter=5000, n_jobs=1,
                                   use_legacy_attributes=False)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                                     class_weight="balanced_subsample", n_jobs=1,
                                     random_state=seed)
    elif name == "xgb":
        from xgboost import XGBClassifier

        pos = y.sum()
        clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, scale_pos_weight=(len(y) - pos) / max(pos, 1),
                            tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)
    elif name == "mlp":
        # "simple neural network": two small hidden layers, L2 + early stopping
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                            early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                            max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"unknown model {name!r}")
    return make_pipeline(StandardScaler(), clf)


def fit_predict(name: str, Xtr, ytr, Xte, seed: int = 0) -> np.ndarray:
    """Fit on train, return P(tumor) on test."""
    model = make_model(name, ytr, seed)
    kw = {}
    if name == "mlp":  # MLP has no class_weight; use balanced sample weights instead
        kw["mlpclassifier__sample_weight"] = compute_sample_weight("balanced", ytr)
    model.fit(Xtr, ytr, **kw)
    return model.predict_proba(Xte)[:, 1]
