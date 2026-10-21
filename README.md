# genepanel-shrink

**Can a small gene signature detect cancer as well as the whole transcriptome?
How much biological information can we remove before detection degrades?**

TCGA RNA-seq for breast (BRCA) and lung adenocarcinoma (LUAD). We rank genes with six
methods, cut the ranking into panels of 500 → 100 → 50 → 20 → 10 → 5 → 2 → 1 genes,
train four classifiers on each, and compare them with the all-genes model and with
random panels of the same size. Findings are in [REPORT.md](REPORT.md).

## Design in brief
| | |
|---|---|
| Data | UCSC Xena TCGA `HiSeqV2` (log2 RSEM+1), clinical matrix |
| Tasks | primary tumor vs solid-tissue normal; **Stage I** tumor vs normal |
| Gene filter (label-free) | expressed (>1) in ≥20% of samples, SD ≥ 0.1 → ~17.7k genes |
| Selectors | univariate F-test, LASSO path, mutual information, RFE (L2 logreg), SHAP (XGBoost); baselines: top-variance, **random** (20 draws) |
| Models | logistic regression (C tuned by inner CV), random forest, XGBoost, MLP (64-32) |
| Validation | 5×5 repeated `StratifiedGroupKFold`, grouped by patient; **selection happens inside each training fold** |
| Metrics | ROC-AUC, PR-AUC, balanced accuracy, sensitivity at 95% specificity |
| "Retains performance" | lower 95% bound of paired fold-wise ΔAUC (panel − all genes) > −0.01 |
| Biology | OncoKB overlap (hypergeometric), Enrichr (GO/KEGG/Hallmark), PubMed co-mentions vs random genes |

## Run it
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
./scripts/fix_xgboost_macos.sh      # macOS without Homebrew libomp only
.venv/bin/pytest                    # synthetic-data tests, incl. a leakage check
.venv/bin/genepanel all --config config/smoke.yaml   # ~1 min end-to-end check
.venv/bin/genepanel all             # full run: download, CV (~20 min on 8 cores), analyze, biology, figures
python scripts/make_notebooks.py && jupyter nbconvert --execute --inplace notebooks/*.ipynb
```
Steps can be run separately: `download`, `run`, `analyze`, `biology` (`--no-pubmed` to skip
NCBI queries), `report`. Fold results are cached in `results/cache/`; **delete it after
changing the config**.

## Layout
```
config/            default.yaml (full), smoke.yaml (quick)
src/genepanel/     data, preprocess, selectors, models, evaluate, analysis, biology, plots, cli
notebooks/         01_eda, 02_results, 03_biology
results/tables/    every number in the report (CSV)
results/figures/   report figures
tests/             pytest suite
```

## Notes on choices
- **MLP is scikit-learn's**, not PyTorch: a 2-layer network is all the question needs, and it
  avoids loading two OpenMP runtimes (torch + XGBoost) in the same worker processes on macOS.
- **LASSO ranks genes by entry order on the regularization path** rather than taking one
  tuned C, so every panel size gets a well-defined top-k.
- **RFE drops 10% of the *remaining* genes per round** (custom loop), which keeps the ranking
  fine-grained at k ≤ 10. sklearn's fractional step is relative to the starting count.
- LASSO, MI, RFE and SHAP rank within the top 2,000 univariate genes of each training fold.
- OncoKB replaces COSMIC CGC as the cancer-gene reference because COSMIC requires a login.
