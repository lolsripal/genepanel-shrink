"""Generate the analysis notebooks (run from repo root): python scripts/make_notebooks.py"""
import nbformat as nbf

SETUP = """\
import os, sys
os.chdir('..') if os.path.basename(os.getcwd()) == 'notebooks' else None
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from IPython.display import Image, display
from genepanel.cli import load_config, load_datasets
cfg = load_config('config/default.yaml')
TAB = f"{cfg['results_dir']}/tables"; FIG = f"{cfg['results_dir']}/figures"
pd.set_option('display.precision', 4)"""


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c)
               for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


eda = nb([
    "md:# 01 · Exploring the data\nTCGA BRCA and LUAD RNA-seq (UCSC Xena, log2(RSEM+1)). Which samples are there, and how separable are tumor and normal before any modelling?",
    SETUP,
    "datasets = load_datasets(cfg)\n"
    "pd.DataFrame([{'cancer': d.cancer, 'task': d.task, 'samples': len(d.y), 'tumor': int(d.y.sum()),\n"
    "               'normal': int((1-d.y).sum()), 'patients': len(set(d.groups)), 'genes_after_filter': len(d.genes)}\n"
    "              for d in datasets])",
    "md:Stage distribution of tumors in the tumor-vs-normal task (Stage I tumors form the early-detection task).",
    "pd.concat({d.cancer: pd.Series(d.stage[d.y == 1]).value_counts() for d in datasets if d.task == 'tumor_vs_normal'}, axis=1)",
    "md:## PCA: is the tumor/normal split visible without labels?",
    "from sklearn.decomposition import PCA\nfrom sklearn.preprocessing import StandardScaler\n"
    "tvn = [d for d in datasets if d.task == 'tumor_vs_normal']\n"
    "fig, axes = plt.subplots(1, len(tvn), figsize=(5 * len(tvn), 4))\n"
    "for ax, d in zip(np.atleast_1d(axes), tvn):\n"
    "    pc = PCA(2).fit(StandardScaler().fit_transform(d.X))\n"
    "    Z = pc.transform(StandardScaler().fit_transform(d.X))\n"
    "    for lab, col, name in [(1, '#2a78d6', 'tumor'), (0, '#eb6834', 'normal')]:\n"
    "        ax.scatter(Z[d.y == lab, 0], Z[d.y == lab, 1], s=8, c=col, label=name, alpha=0.7)\n"
    "    ax.set(title=d.cancer, xlabel=f'PC1 ({pc.explained_variance_ratio_[0]:.0%})', ylabel=f'PC2 ({pc.explained_variance_ratio_[1]:.0%})')\n"
    "    ax.legend()",
    "md:## How strong is the best single gene?\nIf one gene already separates tumor from normal, the 'small panel' question is mostly answered by the data, not the method.",
    "from sklearn.metrics import roc_auc_score\nrows = []\n"
    "for d in datasets:\n"
    "    auc = np.array([roc_auc_score(d.y, d.X[:, j]) for j in range(d.X.shape[1])])\n"
    "    auc = np.maximum(auc, 1 - auc)  # direction-free\n"
    "    top = np.argsort(-auc)[:5]\n"
    "    rows.append({'cancer': d.cancer, 'task': d.task, 'genes_auc_ge_0.95': int((auc >= 0.95).sum()),\n"
    "                 'genes_auc_ge_0.90': int((auc >= 0.90).sum()),\n"
    "                 'top5 (in-sample AUC)': ', '.join(f'{d.genes[j]} {auc[j]:.3f}' for j in top)})\n"
    "pd.DataFrame(rows)",
    "md:*Note:* these single-gene AUCs are in-sample (gene chosen on all data), so they are optimistic. The CV results in notebook 02 select genes inside each fold.",
])

results = nb([
    "md:# 02 · How small can the panel get?\nAll numbers come from 5×5 repeated, patient-grouped CV, with gene selection redone inside every training fold.",
    SETUP,
    "summary = pd.read_csv(f'{TAB}/summary.csv')\nret = pd.read_csv(f'{TAB}/retention.csv')\nvsr = pd.read_csv(f'{TAB}/vs_random.csv')",
    "md:## AUC vs panel size\nDashed line: all genes. Gray band: 5–95% range of random gene panels of the same size (logistic regression).",
    "import glob\nfor f in sorted(glob.glob(f'{FIG}/auc_vs_k_*.png')):\n    display(Image(f))",
    "md:## Smallest panel that retains performance\nNon-inferiority: the lower 95% bound of the paired fold-wise AUC difference (panel − all genes) must stay above −0.01.",
    "ret.pivot_table(index=['cancer', 'task', 'selector'], columns='model', values='smallest_retained_k')",
    "md:## Do selected panels beat random genes?",
    "vsr[vsr.k.isin([1, 2, 5, 10])].pivot_table(index=['cancer', 'task', 'k'], columns='selector', values='auc')\n"
    "  .join(vsr.groupby(['cancer', 'task', 'k']).random_auc_mean.first().rename('RANDOM'))",
    "md:## Beyond AUC: sensitivity at 95% specificity\nThe clinically relevant operating point for screening.",
    "summary[summary.k.isin([1, 5, 10, 50]) & (summary.model == 'logreg') & (summary.selector != 'random')]\\\n"
    "  .pivot_table(index=['cancer', 'task', 'k'], columns='selector', values='sens_at_95spec_mean')",
    "md:## Selector × model at k = 10",
    "display(Image(f'{FIG}/heatmap_k10.png'))",
    "md:## Stability",
    "display(Image(f'{FIG}/stability.png'))",
    "md:## Cross-cancer transfer of the consensus 10-gene panel",
    "pd.read_csv(f'{TAB}/transfer.csv')",
])

bio = nb([
    "md:# 03 · Are the selected genes biologically meaningful?",
    SETUP,
    "cons = pd.read_csv(f'{TAB}/consensus.csv')\n"
    "cons[cons.panel == 'consensus10'][['cancer', 'task', 'gene', 'freq']]",
    "md:## Overlap with the OncoKB cancer gene list\nHypergeometric test against the filtered gene universe.",
    "pd.read_csv(f'{TAB}/oncokb_overlap.csv').query('n_panel >= 5')",
    "md:## Pathway enrichment (Enrichr; background = filtered genes)",
    "enr = pd.read_csv(f'{TAB}/enrichment.csv')\n"
    "enr.sort_values('Adjusted P-value').groupby(['cancer', 'task']).head(8)[['cancer', 'task', 'panel', 'Gene_set', 'Term', 'Adjusted P-value', 'Genes']]",
    "md:## Literature support: PubMed co-mentions vs random genes",
    "display(pd.read_csv(f'{TAB}/pubmed_summary.csv'))\n"
    "pc = pd.read_csv(f'{TAB}/pubmed_counts.csv')\n"
    "pc[pc.set == 'selected'].sort_values(['cancer', 'task', 'pubmed'], ascending=[True, True, False])",
])

for name, n in [("01_eda", eda), ("02_results", results), ("03_biology", bio)]:
    nbf.write(n, f"notebooks/{name}.ipynb")
    print("wrote", name)
