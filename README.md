# genepanel-shrink

[![CI](https://github.com/lolsripal/genepanel-shrink/actions/workflows/ci.yml/badge.svg)](https://github.com/lolsripal/genepanel-shrink/actions/workflows/ci.yml)
[![Report](https://img.shields.io/badge/report-interactive-1f6f8b)](https://lolsripal.github.io/genepanel-shrink/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**How small can a cancer test get before it stops working?**

I trained models to detect cancer from gene expression, then kept shrinking the set of genes
they could use, from ~17,700 down to one, across 19 tasks of increasing difficulty: tumor
tissue, a blood test, healthy controls from another study, and telling cancer types apart.

**→ Interactive report: https://lolsripal.github.io/genepanel-shrink/** · Full write-up: [REPORT.md](REPORT.md)

## The 60-second version

| Setting | What happens as the panel shrinks |
|---|---|
| **1 · Tumor tissue** (TCGA breast, lung) | 1–2 genes match all ~17,700 (AUC ≈ 0.99). The task is easy: hundreds of genes each separate tumor from normal. |
| **2 · Blood test** (tumor-educated platelets, 1,064 people) | Panel size finally matters: ~0.84 AUC with 1 gene, ~0.90 with 10, ~0.96 with all. Hospital plus age, sex and smoking alone (no genes) reach AUC 0.93, so confounding checks are essential. A real signal survives in an age/sex/smoking-matched cohort. |
| **3 · Healthy controls + new data** (GTEx, UCSC Toil) | TCGA "normal" vs GTEx healthy separates perfectly (a study effect). Frozen tissue models mostly keep their AUC on a new pipeline, but a 1-gene lung panel misclassifies 70% of healthy lungs. Bigger panels are more robust. |
| **4 · Harder tasks** (histology, PAM50, 12 cancer types) | 1 gene names 1 of 12 cancer types only 33% of the time (balanced accuracy); ~10 genes reach 95%. Breast subtypes, which shade into each other, need ~50. Selected genes are textbook lineage markers. |

**Takeaway:** small panels look great when the signal is huge and redundant. They lose
accuracy first when the signal is weak, and they break first when the data change.

## What's inside

| | |
|---|---|
| Data | TCGA (UCSC Xena), TCGA + GTEx through one pipeline (UCSC Toil), platelet RNA-seq from GEO (GSE89843, GSE68086) |
| Gene selection | t-test, LASSO path, mutual information, recursive feature elimination, SHAP (XGBoost), **stability selection**; baselines: most-variable and **random** genes |
| Models | logistic regression, random forest, XGBoost, neural network (MLP), all class-balanced |
| Validation | repeated 5-fold CV **grouped by patient**, selection redone **inside every training fold**, corrected repeated-CV test (Nadeau–Bengio) |
| Confounding checks | metadata-only models, hospital fingerprint test, one-hospital and matched-cohort reruns, the original authors' split |
| External validation | train on one pipeline, test frozen on another with truly healthy controls; bootstrap CIs |
| Clinical-style metrics | sensitivity at 95% specificity, calibration (Brier), decision curves |
| Biology | OncoKB overlap, Enrichr pathways, PubMed co-mention counts vs random genes |

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
./scripts/fix_xgboost_macos.sh                       # macOS without Homebrew libomp only
.venv/bin/pytest                                     # 35 tests on synthetic data, incl. a leakage check
.venv/bin/genepanel all --config config/ci.yaml      # < 1 min end-to-end smoke run, no downloads

# the four chapters (downloads public data on first run; times are on 8 cores)
.venv/bin/genepanel all --config config/default.yaml   # 1 · tissue            ~30 min
.venv/bin/genepanel all --config config/blood.yaml     # 2 · blood test        ~1 h
.venv/bin/genepanel all --config config/external.yaml  # 3 · healthy/external  ~30 min, 1.3 GB download
.venv/bin/genepanel all --config config/harder.yaml    # 4 · harder tasks      ~1.7 h

python scripts/build_report_page.py                  # rebuild docs/ (the GitHub Pages site)
python scripts/make_notebooks.py && jupyter nbconvert --execute --inplace notebooks/*.ipynb
```

Each command runs `download → run → analyze → … → report` as listed in the config's
`pipeline:`. Steps can also be run one by one (`genepanel run --config …`). Fold results are
cached under `results/**/cache/v2/`, keyed by a hash of the config, so changing a setting
never reuses stale results.

## Layout

```
config/               one YAML per chapter (+ ci.yaml, smoke.yaml)
src/genepanel/
  sources/            data loaders: tcga, platelets, toil (+ synthetic for CI)
  selectors.py        the six gene-ranking methods
  models.py           the four classifiers
  evaluate.py         nested, patient-grouped repeated CV
  analysis.py         summaries, minimal panels, corrected tests, calibration, decision curves
  confounding.py      blood-test confounding checks and the authors' split
  external.py         frozen-model validation on a second pipeline
  stats.py            Nadeau–Bengio test, bootstrap CIs, Brier, net benefit
  biology.py          OncoKB, Enrichr, PubMed checks
notebooks/            01–03 tissue, 04 blood, 05 external, 06 harder
results/              tables/ and figures/ per chapter (blood/, external/, harder/)
docs/                 the GitHub Pages report (generated)
scripts/              page builder, notebook generator, macOS XGBoost fix
tests/                pytest suite
```

## Notes on choices

- **The neural network is scikit-learn's MLP**, not PyTorch. Two small hidden layers are all
  the question needs, and this avoids loading two OpenMP runtimes in one process on macOS.
- **LASSO ranks genes by when they enter the regularization path**, so every panel size has a
  well-defined top-k. For multiclass tasks it runs one-vs-rest per class.
- **RFE drops 10% of the remaining genes per round**, which keeps the ranking fine-grained at
  k ≤ 10.
- **External validation harmonizes each gene** with a mean/SD match fitted on samples present
  in both pipelines, from training patients only. Unharmonized results are reported too.
- **OncoKB replaces COSMIC** as the cancer-gene reference, because COSMIC requires a login.

## Skills this project uses

Machine learning (feature selection, model comparison, nested cross-validation, calibration),
statistics (corrected resampled tests, bootstrap, decision-curve analysis), genomics data
handling (TCGA, GTEx, GEO, count normalization, gene-ID mapping), study design (confounding,
external validation, leakage tests), and software engineering (a tested, config-driven Python
package with continuous integration and a generated, data-driven website).

## Citation and license

See [CITATION.cff](CITATION.cff). Code is MIT-licensed ([LICENSE](LICENSE)). The data come
from TCGA, GTEx and GEO under their own terms.
