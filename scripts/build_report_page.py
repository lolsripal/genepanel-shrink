"""Build docs/index.html (GitHub Pages) and docs/report.html (claude.ai artifact) from the
result tables of all four chapters. Run from the repo root after the analyses:

    python scripts/build_report_page.py

Every number in the page text is a "fact" computed here from results/*/tables and filled
into the page by key, so the prose can never drift from the tables.
"""
import json
import math
from pathlib import Path

import pandas as pd
import yaml

SUP = ["univariate", "lasso", "mutual_info", "rfe", "shap", "stability"]
CHAPTERS = {
    "tissue": {"dir": "results/tables", "title": "Tissue"},
    "blood": {"dir": "results/blood/tables", "title": "Blood test"},
    "external": {"dir": "results/external/tables", "title": "Healthy controls"},
    "harder": {"dir": "results/harder/tables", "title": "Harder tasks"},
}
LABELS = {
    ("BRCA", "tumor_vs_normal"): "Breast tissue: tumor vs normal",
    ("BRCA", "stage1_vs_normal"): "Breast tissue: Stage I vs normal",
    ("LUAD", "tumor_vs_normal"): "Lung tissue: tumor vs normal",
    ("LUAD", "stage1_vs_normal"): "Lung tissue: Stage I vs normal",
    ("GSE89843", "nsclc_vs_healthy"): "Blood: lung cancer vs healthy",
    ("GSE89843", "nsclc_vs_noncancer"): "Blood: lung cancer vs all non-cancer",
    ("GSE89843", "nonmetastatic_vs_noncancer"): "Blood: non-metastatic lung cancer vs non-cancer",
    ("GSE89843_VUMC", "nsclc_vs_noncancer"): "Blood: lung cancer, one hospital only",
    ("GSE89843_matched", "nsclc_vs_noncancer"): "Blood: lung cancer, age/sex/smoking-matched",
    ("GSE68086", "pancancer_vs_healthy"): "Blood: any of 6 cancers vs healthy",
    ("GSE68086_batch234", "pancancer_vs_healthy"): "Blood: any cancer vs healthy, matched batches",
    ("GSE68086", "cancer_type"): "Blood: which of 6 cancer types",
    ("BRCA_GTEx", "tumor_vs_healthy"): "Breast: TCGA tumor vs GTEx healthy",
    ("LUAD_GTEx", "tumor_vs_healthy"): "Lung: TCGA tumor vs GTEx healthy",
    ("BRCA_GTEx", "normal_vs_healthy"): "Breast: TCGA normal vs GTEx healthy (control)",
    ("LUAD_GTEx", "normal_vs_healthy"): "Lung: TCGA normal vs GTEx healthy (control)",
    ("LUAD_LUSC", "adeno_vs_squamous"): "Lung tumors: adenocarcinoma vs squamous",
    ("BRCA", "pam50_subtype"): "Breast tumors: 5 PAM50 subtypes",
    ("TCGA12", "tissue_of_origin"): "Tumors: which of 12 cancer types",
}


def read(ch, name):
    p = Path(CHAPTERS[ch]["dir"]) / name
    return pd.read_csv(p) if p.exists() else None


def clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def records(df):
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]


facts = {}


def fact(key, value, fmt="{:.2f}"):
    facts[key] = "–" if value is None or (isinstance(value, float) and math.isnan(value)) else (
        fmt.format(value) if fmt else str(value))


# ---------------------------------------------------------------- curves + ladder
curves, ladder = [], []
for ch in CHAPTERS:
    s = read(ch, "summary.csv")
    if s is None:
        continue
    vr = read(ch, "vs_random.csv")
    mp = read(ch, "minimal_panel.csv")
    ret = read(ch, "retention.csv")
    l = s[s.model == "logreg"]
    for (c, t), g in l.groupby(["cancer", "task"]):
        allg = g[g.selector == "all_genes"]
        if allg.empty:
            continue
        sel = g[g.selector.isin(SUP)].pivot(index="k", columns="selector", values="auc_mean")
        var = g[g.selector == "variance"].set_index("k").auc_mean
        rnd = vr[(vr.cancer == c) & (vr.task == t) & (vr.model == "logreg")].groupby("k")[
            ["random_auc_mean", "random_auc_q05", "random_auc_q95"]].first()
        pts = []
        for k, r in sel.iterrows():
            pts.append({"k": int(k), "best": r.max(), "worst": r.min(), "best_sel": r.idxmax(),
                        "variance": clean(var.get(k, float("nan"))),
                        "rand": clean(rnd.random_auc_mean.get(k, float("nan"))),
                        "rand_lo": clean(rnd.random_auc_q05.get(k, float("nan"))),
                        "rand_hi": clean(rnd.random_auc_q95.get(k, float("nan")))})
        label = LABELS.get((c, t), f"{c}: {t}")
        curves.append({"chapter": ch, "cancer": c, "task": t, "label": label,
                       "all": float(allg.auc_mean.iloc[0]), "n_genes": int(allg.k.iloc[0]), "points": pts})
        # ladder: smallest retained panel using the best selector (corrected test)
        rr = ret[(ret.cancer == c) & (ret.task == t) & (ret.model == "logreg") & ret.selector.isin(SUP)]
        best_k = sel.max(axis=1)
        ladder.append({"chapter": ch, "label": label, "cancer": c, "task": t,
                       "auc_all": float(allg.auc_mean.iloc[0]),
                       "auc_1": clean(best_k.get(1, float("nan"))), "auc_10": clean(best_k.get(10, float("nan"))),
                       "rand_10": clean(rnd.random_auc_mean.get(10, float("nan"))),
                       "min_k": clean(float(rr.smallest_retained_k.min())) if len(rr) else None})


def curve(c, t):
    return next((x for x in curves if x["cancer"] == c and x["task"] == t), None)


def at(c, t, k, key="best"):
    cv = curve(c, t)
    if cv is None:
        return None
    if k == "all":
        return cv["all"]
    return next((p[key] for p in cv["points"] if p["k"] == k), None)


# ---------------------------------------------------------------- chapter 1 facts
s1 = read("tissue", "summary.csv")
fact("ch1_genes", int(s1[s1.selector == "all_genes"].k.min()), "{:,}")
fact("ch1_brca_1", at("BRCA", "tumor_vs_normal", 1), "{:.3f}")
fact("ch1_luad_1", at("LUAD", "tumor_vs_normal", 1), "{:.3f}")
fact("ch1_rand_1", at("BRCA", "tumor_vs_normal", 1, "rand"), "{:.2f}")
fact("ch1_brca_10", at("BRCA", "tumor_vs_normal", 10), "{:.3f}")
fact("ch1_brca_all", at("BRCA", "tumor_vs_normal", "all"), "{:.3f}")
fact("ch1_brca_rand10", at("BRCA", "tumor_vs_normal", 10, "rand"), "{:.2f}")
fact("ch1_all", min(at("BRCA", "tumor_vs_normal", "all"), at("LUAD", "tumor_vs_normal", "all")), "{:.3f}")
ch1_min = [r["min_k"] for r in ladder if r["chapter"] == "tissue" and r["min_k"] is not None]
fact("ch1_min_k", int(max(ch1_min)) if ch1_min else None, "{}")

# ---------------------------------------------------------------- chapter 2 facts
conf = read("blood", "confounders.csv")
site = read("blood", "site_signature.csv")
bench = read("blood", "benchmark.csv")
dca = read("blood", "decision_curve.csv")
B = ("GSE89843", "nsclc_vs_noncancer")
fact("ch2_all", at(*B, "all"), "{:.2f}")
fact("ch2_1", at(*B, 1), "{:.2f}")
fact("ch2_10", at(*B, 10), "{:.2f}")
fact("ch2_100", at(*B, 100), "{:.2f}")
fact("ch2_rand10", at(*B, 10, "rand"), "{:.2f}")
fact("ch2_matched_all", at("GSE89843_matched", "nsclc_vs_noncancer", "all"), "{:.2f}")
fact("ch2_matched_10", at("GSE89843_matched", "nsclc_vs_noncancer", 10), "{:.2f}")
fact("ch2_vumc_all", at("GSE89843_VUMC", "nsclc_vs_noncancer", "all"), "{:.2f}")
fact("ch2_nonmet_all", at("GSE89843", "nonmetastatic_vs_noncancer", "all"), "{:.2f}")
fact("ch2_nonmet_10", at("GSE89843", "nonmetastatic_vs_noncancer", 10), "{:.2f}")
fact("ch2_pan_all", at("GSE68086", "pancancer_vs_healthy", "all"), "{:.2f}")
fact("ch2_pan_10", at("GSE68086", "pancancer_vs_healthy", 10), "{:.2f}")
fact("ch2_type_all", at("GSE68086", "cancer_type", "all"), "{:.2f}")
fact("ch2_type_10", at("GSE68086", "cancer_type", 10), "{:.2f}")
s2 = read("blood", "summary.csv")
ba = s2[(s2.cancer == "GSE68086") & (s2.task == "cancer_type") & (s2.selector == "all_genes")]
fact("ch2_type_balacc", ba[ba.model == "logreg"].bal_acc_mean.iloc[0] * 100 if len(ba) else None, "{:.0f}%")


def conf_auc(c, t, f):
    d = conf[(conf.cancer == c) & (conf.task == t) & (conf.features == f)]
    return float(d.auc_mean.iloc[0]) if len(d) else None


fact("ch2_hosp", conf_auc(*B, "hospital only"))
fact("ch2_age", conf_auc(*B, "age only"))
fact("ch2_meta", conf_auc(*B, "hospital + age + sex + smoking"))
fact("ch2_nonmet_demo", conf_auc("GSE89843", "nonmetastatic_vs_noncancer", "age + sex + smoking"))
fact("ch2_matched_meta", conf_auc("GSE89843_matched", "nsclc_vs_noncancer", "hospital + age + sex + smoking"))
fact("ch2_site_all", float(site[site.genes == "all"].auc_mean.iloc[0]))
fact("ch2_site_n_other", int(site.n_other.iloc[0]), "{}")


def bench_auc(cohort, subset, sel, k=None):
    d = bench[(bench.cohort == cohort) & (bench.subset == subset) & (bench.selector == sel)]
    if k is not None:
        d = d[d.k == k]
    return float(d.auc.iloc[0]) if len(d) else None


fact("ch2_bench_seen", bench_auc("full_cohort", "cancers from training hospitals", "all_genes"))
fact("ch2_bench_unseen", bench_auc("full_cohort", "cancers from unseen hospitals", "all_genes"))
fact("ch2_bench_unseen_10", bench_auc("full_cohort", "cancers from unseen hospitals", "lasso", 10))
fact("ch2_bench_matched", bench_auc("matched_cohort", "all validation", "all_genes"))
fact("ch2_bench_matched_10", bench_auc("matched_cohort", "all validation", "lasso", 10))
fact("ch2_bench_ntrain", int(bench[bench.cohort == "full_cohort"].n_train.iloc[0]), "{}")
pub = yaml.safe_load(open("data/ref/benchmarks.yaml"))

# ---------------------------------------------------------------- chapter 3 facts
ext = read("external", "external.csv")
fact("ch3_study_2", max(at("BRCA_GTEx", "normal_vs_healthy", 2), at("LUAD_GTEx", "normal_vs_healthy", 2)), "{:.3f}")
fact("ch3_study_rand1", at("BRCA_GTEx", "normal_vs_healthy", 1, "rand"), "{:.2f}")


def ext_v(tissue, sel, k, setting, col):
    d = ext[(ext.tissue == tissue) & (ext.selector == sel) & (ext.setting == setting)]
    d = d[d.k == k] if k != "all" else d
    return float(d[col].iloc[0]) if len(d) else None


for tis in ["lung", "breast"]:
    for k in [1, 10, 100]:
        for setting, tag in [("internal", "in"), ("external_harmonized", "ex")]:
            fact(f"ch3_{tis}_{k}_{tag}_auc", ext_v(tis, "lasso", k, setting, "auc"), "{:.2f}")
            fact(f"ch3_{tis}_{k}_{tag}_spec", (ext_v(tis, "lasso", k, setting, "spec") or 0) * 100, "{:.0f}%")
ext_chart = []
if ext is not None:
    d = ext[(ext.selector.isin(["lasso", "all_genes"])) & ext.setting.isin(["internal", "external_harmonized"])]
    for r in d.to_dict("records"):
        ext_chart.append({k: clean(r[k]) for k in ["tissue", "selector", "k", "setting", "auc", "sens", "spec"]})

# ---------------------------------------------------------------- chapter 4 facts
for c, t, tag in [("LUAD_LUSC", "adeno_vs_squamous", "hist"), ("BRCA", "pam50_subtype", "pam50"),
                  ("TCGA12", "tissue_of_origin", "too")]:
    fact(f"ch4_{tag}_all", at(c, t, "all"), "{:.3f}")
    fact(f"ch4_{tag}_1", at(c, t, 1), "{:.3f}")
    fact(f"ch4_{tag}_10", at(c, t, 10), "{:.3f}")
    fact(f"ch4_{tag}_rand10", at(c, t, 10, "rand"), "{:.2f}")
    lr = next((r for r in ladder if r["cancer"] == c and r["task"] == t), None)
    fact(f"ch4_{tag}_mink", (int(lr["min_k"]) if lr and lr["min_k"] else None), "{}")
s4 = read("harder", "summary.csv")
for tag in ["pam50", "too"]:  # placeholders until chapter 4 has run
    for k in ["all", "10", "1"]:
        fact(f"ch4_{tag}_balacc_{k}", None)
if s4 is not None:
    for c, t, tag in [("BRCA", "pam50_subtype", "pam50"), ("TCGA12", "tissue_of_origin", "too")]:
        d = s4[(s4.cancer == c) & (s4.task == t) & (s4.model == "logreg")]
        allb = d[d.selector == "all_genes"].bal_acc_mean
        b10 = d[d.selector.isin(SUP) & (d.k == 10)].bal_acc_mean.max()
        b1 = d[d.selector.isin(SUP) & (d.k == 1)].bal_acc_mean.max()
        fact(f"ch4_{tag}_balacc_all", allb.iloc[0] * 100 if len(allb) else None, "{:.0f}%")
        fact(f"ch4_{tag}_balacc_10", b10 * 100, "{:.0f}%")
        fact(f"ch4_{tag}_balacc_1", b1 * 100, "{:.0f}%")

# ---------------------------------------------------------------- tables for the page
conf_tbl = records(conf) if conf is not None else []
bench_tbl = []
if bench is not None:
    d = bench[bench.selector.isin(["all_genes", "lasso"]) & bench.k.isin([1, 10, 50, 100])
              | (bench.selector == "all_genes")]
    bench_tbl = records(d.drop_duplicates(["cohort", "subset", "selector", "k"]))
dca_tbl = []
if dca is not None:
    d = dca[(dca.cancer == B[0]) & (dca.task == B[1])]
    dca_tbl = records(d[["selector", "k", "threshold", "model", "treat_all"]])
cons_tbl = []
for ch in CHAPTERS:
    c = read(ch, "consensus.csv")
    if c is not None:
        c = c[c.panel == "consensus10"]
        for (cc, t), g in c.groupby(["cancer", "task"]):
            cons_tbl.append({"chapter": ch, "label": LABELS.get((cc, t), f"{cc}: {t}"),
                             "genes": g.sort_values("freq", ascending=False).gene.tolist()})
models_tbl = []
for ch in CHAPTERS:
    s = read(ch, "summary.csv")
    if s is None:
        continue
    for (c, t), g in s.groupby(["cancer", "task"]):
        row = {"chapter": ch, "label": LABELS.get((c, t), f"{c}: {t}")}
        for m in ["logreg", "rf", "xgb", "mlp"]:
            a = g[(g.model == m) & (g.selector == "all_genes")].auc_mean
            b = g[(g.model == m) & g.selector.isin(SUP) & (g.k == 10)].auc_mean
            row[f"{m}_all"] = clean(float(a.iloc[0])) if len(a) else None
            row[f"{m}_10"] = clean(float(b.max())) if len(b) else None
        models_tbl.append(row)

data = {"facts": facts, "curves": curves, "ladder": ladder, "confounders": conf_tbl,
        "benchmark": bench_tbl, "published": pub, "dca": dca_tbl, "external": ext_chart,
        "consensus": cons_tbl, "models": models_tbl,
        "site": records(site) if site is not None else []}

tpl = Path("scripts/report_template.html").read_text()
missing = [k for k in set(__import__("re").findall(r'data-f="([^"]+)"', tpl)) if k not in facts]
if missing:
    raise SystemExit(f"template references unknown facts: {sorted(missing)}")
html = tpl.replace("/*__DATA__*/null", json.dumps(data, allow_nan=False))
split = html.index('<div class="wrap">')
head, body = html[:split], html[split:]
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n'
        + head + '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
Path("docs").mkdir(exist_ok=True)
Path("docs/report.html").write_text(html)
Path("docs/index.html").write_text(page)
Path("docs/.nojekyll").touch()
md_tpl = Path("scripts/report_template.md").read_text()
md_keys = set(__import__("re").findall(r"\{(\w+)\}", md_tpl))
missing_md = sorted(md_keys - set(facts))
if missing_md:
    raise SystemExit(f"report_template.md references unknown facts: {missing_md}")
Path("REPORT.md").write_text(md_tpl.format_map(facts))
print(f"wrote docs/report.html, docs/index.html, REPORT.md ({len(curves)} curves, {len(facts)} facts)")
