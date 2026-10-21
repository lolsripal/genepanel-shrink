"""Build docs/report.html from results/tables (run from repo root after `genepanel analyze`)."""
import json
from pathlib import Path

import pandas as pd

TAB = Path("results/tables")
SUP = ["univariate", "lasso", "mutual_info", "rfe", "shap"]
s = pd.read_csv(TAB / "summary.csv")
vr = pd.read_csv(TAB / "vs_random.csv")
ret = pd.read_csv(TAB / "retention.csv")
st = pd.read_csv(TAB / "stability.csv")

curves = []
for (c, t), g in s[s.model == "logreg"].groupby(["cancer", "task"]):
    sel = g[g.selector.isin(SUP)].pivot(index="k", columns="selector", values="auc_mean")
    var = g[g.selector == "variance"].set_index("k").auc_mean
    rnd = vr[(vr.cancer == c) & (vr.task == t)].groupby("k")[["random_auc_mean", "random_auc_q05", "random_auc_q95"]].first()
    curves.append({
        "cancer": c, "task": t,
        "all": float(g[g.selector == "all_genes"].auc_mean.iloc[0]),
        "n_genes": int(g[g.selector == "all_genes"].k.iloc[0]),
        "points": [{"k": int(k), "best": float(r.max()), "worst": float(r.min()),
                    "best_sel": r.idxmax(), "variance": float(var[k]),
                    "rand": float(rnd.loc[k].random_auc_mean), "rand_lo": float(rnd.loc[k].random_auc_q05),
                    "rand_hi": float(rnd.loc[k].random_auc_q95)} for k, r in sel.iterrows()],
    })

def best_by_model(k, metric):
    d = s[(s.k == k) & s.selector.isin(SUP)]
    return d.groupby(["cancer", "task", "model"])[metric].max().unstack().round(4).reset_index().to_dict("records")

models = {"k1": best_by_model(1, "auc_mean"), "k10": best_by_model(10, "auc_mean"),
          "all": s[s.selector == "all_genes"].pivot_table(index=["cancer", "task"], columns="model",
                                                          values="auc_mean").round(4).reset_index().to_dict("records")}
selectors = (s[(s.model == "logreg") & s.k.isin([1, 2, 10]) & s.selector.isin(SUP + ["variance"])]
             .pivot_table(index=["cancer", "task", "k"], columns="selector", values="auc_mean").round(4)
             .reset_index().to_dict("records"))
retention = (ret[ret.selector.isin(SUP)].groupby(["cancer", "task", "model"]).smallest_retained_k.min()
             .unstack().reset_index().to_dict("records"))
retention_var = ret[ret.selector == "variance"].pivot_table(index=["cancer", "task"], columns="model",
                                                            values="smallest_retained_k").reset_index().to_dict("records")
sens = (s[(s.model == "logreg") & s.k.isin([1, 2, 5]) & s.selector.isin(SUP)]
        .groupby(["cancer", "task", "k"]).sens_at_95spec_mean.max().unstack().round(4).reset_index().to_dict("records"))
stab = st[st.k == 10].pivot_table(index=["cancer", "task"], columns="selector", values="jaccard_mean").round(2).reset_index().to_dict("records")
data = {"curves": curves, "models": models, "selectors": selectors, "retention": retention,
        "retention_var": retention_var, "sens": sens, "stability": stab,
        "transfer": pd.read_csv(TAB / "transfer.csv").round(4).to_dict("records"),
        "pubmed": pd.read_csv(TAB / "pubmed_summary.csv").round(4).to_dict("records")}

tpl = Path("scripts/report_template.html").read_text()
Path("docs").mkdir(exist_ok=True)
Path("docs/report.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(data)))
print("wrote docs/report.html")
