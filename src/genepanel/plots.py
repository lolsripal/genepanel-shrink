"""Static report figures (PNG) from results/tables. Palette: validated 6-slot categorical
order; selectors also get distinct markers so identity never rests on color alone."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SELECTOR_STYLE = {  # fixed order -> fixed hue, never re-assigned when filtered
    "univariate": ("#2a78d6", "o"),
    "lasso": ("#eb6834", "s"),
    "mutual_info": ("#1baf7a", "^"),
    "rfe": ("#eda100", "D"),
    "shap": ("#e87ba4", "v"),
    "variance": ("#008300", "P"),
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
MODEL_LABEL = {"logreg": "Logistic regression", "rf": "Random forest", "xgb": "XGBoost",
               "mlp": "Neural net (MLP)"}
TASK_LABEL = {"tumor_vs_normal": "tumor vs normal", "stage1_vs_normal": "Stage I tumor vs normal"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def _ks(df):
    return sorted(df.k.unique())


def auc_vs_k(summary: pd.DataFrame, vsr: pd.DataFrame, cancer: str, task: str, out: Path):
    s = summary[(summary.cancer == cancer) & (summary.task == task)]
    models = [m for m in MODEL_LABEL if m in s.model.unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    ks = _ks(s[s.selector.isin(SELECTOR_STYLE)])
    lo = s[s.selector.isin(SELECTOR_STYLE)].auc_mean.min()
    for ax, model in zip(axes, models):
        ref = s[(s.selector == "all_genes") & (s.model == model)]
        if len(ref):
            ax.axhline(ref.auc_mean.iloc[0], color=INK, lw=1, ls="--")
            ax.annotate(f"all {int(ref.k.iloc[0]):,} genes", (ks[-1], ref.auc_mean.iloc[0]),
                        xytext=(0, -12), textcoords="offset points", fontsize=8, color=MUTED,
                        ha="right")
        r = vsr[(vsr.cancer == cancer) & (vsr.task == task) & (vsr.model == model)]
        if len(r):
            band = r.groupby("k")[["random_auc_q05", "random_auc_q95", "random_auc_mean"]].first()
            ax.fill_between(band.index, band.random_auc_q05, band.random_auc_q95,
                            color="#c3c2b7", alpha=0.5, lw=0, label="random genes (5–95%)")
            ax.plot(band.index, band.random_auc_mean, color=MUTED, lw=1)
            lo = min(lo, band.random_auc_q05.min())
        for sel, (col, mk) in SELECTOR_STYLE.items():
            d = s[(s.selector == sel) & (s.model == model)].sort_values("k")
            if len(d):
                ax.plot(d.k, d.auc_mean, color=col, marker=mk, ms=5, lw=2, label=sel)
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_title(MODEL_LABEL[model])
        ax.set_xlabel("genes in panel")
    axes[0].set_ylabel("ROC-AUC (mean over 25 folds)")
    axes[0].set_ylim(max(0.5, lo - 0.01), 1.002)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{cancer}: {TASK_LABEL.get(task, task)}. AUC as the gene panel shrinks",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def heatmap_k(summary: pd.DataFrame, k: int, out: Path, metric="auc_mean"):
    s = summary[(summary.k == k) & summary.selector.isin(SELECTOR_STYLE)]
    combos = s[["cancer", "task"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(combos), figsize=(3.2 * len(combos), 3.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)
    vmin = s[metric].min()
    for ax, (c, t) in zip(axes, combos):
        p = (s[(s.cancer == c) & (s.task == t)].pivot(index="selector", columns="model", values=metric)
             .reindex(index=[x for x in SELECTOR_STYLE if x in s.selector.unique()],
                      columns=[m for m in MODEL_LABEL if m in s.model.unique()]))
        im = ax.imshow(p.values, cmap="Blues", vmin=vmin, vmax=1.0, aspect="auto")
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                v = p.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if v > vmin + 0.6 * (1 - vmin) else INK)
        ax.set_xticks(range(p.shape[1]), p.columns)
        ax.set_yticks(range(p.shape[0]), p.index)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.set_title(f"{c}: {TASK_LABEL.get(t, t)}", fontsize=10)
    fig.suptitle(f"ROC-AUC with a {k}-gene panel (selector × model)", x=0.01, y=1.04, ha="left")
    fig.colorbar(im, ax=axes, shrink=0.8, label="ROC-AUC")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def error_vs_k(summary: pd.DataFrame, vsr: pd.DataFrame, out: Path, model="logreg"):
    """1 - AUC on a log axis: near the ceiling, this is where the differences live."""
    s = summary[summary.model == model]
    combos = s[["cancer", "task"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(combos), figsize=(3.4 * len(combos), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    floor = 1e-4
    for ax, (c, t) in zip(axes, combos):
        d0 = s[(s.cancer == c) & (s.task == t)]
        ks = _ks(d0[d0.selector.isin(SELECTOR_STYLE)])
        r = vsr[(vsr.cancer == c) & (vsr.task == t) & (vsr.model == model)]
        if len(r):
            band = r.groupby("k")[["random_auc_q05", "random_auc_q95", "random_auc_mean"]].first()
            ax.fill_between(band.index, np.maximum(1 - band.random_auc_q95, floor),
                            np.maximum(1 - band.random_auc_q05, floor), color="#c3c2b7", alpha=0.5,
                            lw=0, label="random genes (5–95%)")
            ax.plot(band.index, np.maximum(1 - band.random_auc_mean, floor), color=MUTED, lw=1)
        ref = d0[d0.selector == "all_genes"]
        if len(ref):
            ax.axhline(max(1 - ref.auc_mean.iloc[0], floor), color=INK, lw=1, ls="--",
                       label="all genes")
        for sel, (col, mk) in SELECTOR_STYLE.items():
            d = d0[d0.selector == sel].sort_values("k")
            if len(d):
                ax.plot(d.k, np.maximum(1 - d.auc_mean, floor), color=col, marker=mk, ms=5, lw=2,
                        label=sel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_title(f"{c}: {TASK_LABEL.get(t, t)}", fontsize=10)
        ax.set_xlabel("genes in panel")
    axes[0].set_ylabel(f"1 − ROC-AUC (log; floor {floor:g})")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Error as the panel shrinks ({MODEL_LABEL[model]}). Lower is better",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stability_plot(stab: pd.DataFrame, out: Path):
    combos = stab[["cancer", "task"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(1, len(combos), figsize=(3.4 * len(combos), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (c, t) in zip(axes, combos):
        for sel, (col, mk) in SELECTOR_STYLE.items():
            d = stab[(stab.cancer == c) & (stab.task == t) & (stab.selector == sel)].sort_values("k")
            if len(d):
                ax.plot(d.k, d.jaccard_mean, color=col, marker=mk, ms=5, lw=2, label=sel)
        ax.set_xscale("log")
        ax.set_xticks(_ks(stab), [str(k) for k in _ks(stab)])
        ax.minorticks_off()
        ax.set_title(f"{c}: {TASK_LABEL.get(t, t)}", fontsize=10)
        ax.set_xlabel("genes in panel")
    axes[0].set_ylabel("mean pairwise Jaccard\nacross CV folds")
    axes[0].set_ylim(0, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Panel stability: do different training folds pick the same genes?",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_all(cfg):
    res = Path(cfg["results_dir"])
    tab, fig = res / "tables", res / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(tab / "summary.csv")
    vsr = pd.read_csv(tab / "vs_random.csv")
    for c, t in summary[["cancer", "task"]].drop_duplicates().values:
        auc_vs_k(summary, vsr, c, t, fig / f"auc_vs_k_{c}_{t}.png")
    error_vs_k(summary, vsr, fig / "error_vs_k_logreg.png")
    for k in (10, 5, 1):
        if k in summary.k.values:
            heatmap_k(summary, k, fig / f"heatmap_k{k}.png")
    stability_plot(pd.read_csv(tab / "stability.csv"), fig / "stability.png")
    print(f"wrote figures to {fig}")
