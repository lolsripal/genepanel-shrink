"""Is there known biology behind the selected genes?

Three complementary checks on the consensus panels:
1. Overlap with the OncoKB cancer gene list (public; COSMIC CGC needs a login),
   tested against the filtered gene universe with a hypergeometric test.
2. Pathway enrichment (Enrichr via gseapy) of the larger per-selector consensus
   panels, with the filtered gene universe as background.
3. Literature support: PubMed co-mention counts ("GENE"[tiab] AND cancer terms),
   compared with same-size sets of random genes from the universe.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import hypergeom, mannwhitneyu

ONCOKB_URL = "https://www.oncokb.org/api/v1/utils/cancerGeneList.txt"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
CANCER_TERMS = {"BRCA": "breast cancer", "LUAD": "lung adenocarcinoma"}
LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human", "MSigDB_Hallmark_2020"]


def oncokb_genes(ref_dir: Path) -> set[str]:
    path = ref_dir / "oncokb_cancerGeneList.tsv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(ONCOKB_URL, timeout=60)
        r.raise_for_status()
        path.write_text(r.text)
    df = pd.read_csv(path, sep="\t")
    return set(df["Hugo Symbol"])


def overlap_test(panel: set, reference: set, universe: set) -> dict:
    ref = reference & universe
    hits = panel & ref
    # P(X >= hits) for a random panel of this size drawn from the universe
    p = hypergeom.sf(len(hits) - 1, len(universe), len(ref), len(panel))
    return {"n_panel": len(panel), "n_hits": len(hits), "expected": len(panel) * len(ref) / len(universe),
            "p_value": p, "hits": ",".join(sorted(hits))}


def enrich(genes: list[str], background: list[str]) -> pd.DataFrame:
    import gseapy as gp

    res = gp.enrichr(gene_list=list(genes), gene_sets=LIBRARIES, background=list(background),
                     outdir=None, no_plot=True)
    return res.results


def pubmed_count(gene: str, cancer: str, session: requests.Session, cache: dict) -> int:
    key = f"{gene}|{cancer}"
    if key not in cache:
        term = f'"{gene}"[tiab] AND "{CANCER_TERMS[cancer]}"[tiab]'
        for attempt in range(5):
            r = session.get(EUTILS, params={"db": "pubmed", "term": term, "retmode": "json",
                                            "rettype": "count"}, timeout=30)
            if r.status_code == 200:
                cache[key] = int(r.json()["esearchresult"]["count"])
                break
            time.sleep(2 ** attempt)
        time.sleep(0.35)  # NCBI limit without API key: 3 requests/second
    return cache.get(key, -1)


def literature(panels: pd.DataFrame, universe: dict, cache_path: Path, n_random=50, seed=0):
    cache = pd.read_csv(cache_path).set_index("key")["count"].to_dict() if cache_path.exists() else {}
    s = requests.Session()
    rng = np.random.default_rng(seed)
    rows = []
    try:
        for (c, t), grp in panels.groupby(["cancer", "task"]):
            sel = grp.gene.unique()
            rnd = rng.choice(sorted(universe[(c, t)] - set(sel)), size=n_random, replace=False)
            for g in sel:
                rows.append({"cancer": c, "task": t, "gene": g, "set": "selected",
                             "pubmed": pubmed_count(g, c, s, cache)})
            for g in rnd:
                rows.append({"cancer": c, "task": t, "gene": g, "set": "random",
                             "pubmed": pubmed_count(g, c, s, cache)})
    finally:
        pd.DataFrame({"key": list(cache), "count": list(cache.values())}).to_csv(cache_path, index=False)
    return pd.DataFrame(rows)


def run(cfg, pubmed: bool = True):
    from .cli import load_datasets

    res = Path(cfg["results_dir"])
    tab = res / "tables"
    cons = pd.read_csv(tab / "consensus.csv")
    universe = {(d.cancer, d.task): set(d.genes) for d in load_datasets(cfg)}
    onco = oncokb_genes(Path(cfg["data_dir"]) / "ref")

    # 1. OncoKB overlap for every consensus panel
    ov = []
    for (c, t, panel), grp in cons.groupby(["cancer", "task", "panel"]):
        ov.append({"cancer": c, "task": t, "panel": panel,
                   **overlap_test(set(grp.gene), onco, universe[(c, t)])})
    pd.DataFrame(ov).to_csv(tab / "oncokb_overlap.csv", index=False)

    # 2. Enrichment of the larger panels (enrichment on <20 genes has little power)
    enr = []
    big = cons[cons.k.isin([50, 100]) & (cons.selector != "consensus")]
    for (c, t, panel), grp in big.groupby(["cancer", "task", "panel"]):
        if len(grp) < 10:
            continue
        try:
            r = enrich(grp.gene.tolist(), sorted(universe[(c, t)]))
        except Exception as e:  # network / Enrichr hiccups shouldn't kill the run
            print(f"enrichment failed for {c}/{t}/{panel}: {e}")
            continue
        r = r[r["Adjusted P-value"] < 0.05].assign(cancer=c, task=t, panel=panel)
        enr.append(r)
    if enr:
        pd.concat(enr).to_csv(tab / "enrichment.csv", index=False)

    # 3. Literature support for the 10-gene consensus panels
    if pubmed:
        small = cons[cons.panel == "consensus10"]
        lit = literature(small, universe, Path(cfg["data_dir"]) / "ref" / "pubmed_cache.csv",
                         seed=cfg["seed"])
        lit.to_csv(tab / "pubmed_counts.csv", index=False)
        summ = []
        for (c, t), grp in lit.groupby(["cancer", "task"]):
            a = grp[grp.set == "selected"].pubmed
            b = grp[grp.set == "random"].pubmed
            summ.append({"cancer": c, "task": t, "median_selected": a.median(),
                         "median_random": b.median(), "frac_selected_with_any": (a > 0).mean(),
                         "frac_random_with_any": (b > 0).mean(),
                         "mannwhitney_p": mannwhitneyu(a, b, alternative="greater").pvalue})
        pd.DataFrame(summ).to_csv(tab / "pubmed_summary.csv", index=False)
    print(f"wrote biology tables to {tab}")
