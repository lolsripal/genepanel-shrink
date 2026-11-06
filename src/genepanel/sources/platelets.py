"""Tumor-educated platelet (TEP) RNA-seq from a blood draw (Best et al., GEO).

GSE89843 (Best 2017): 779 people: NSCLC, healthy controls, and other non-cancer diseases.
GSE68086 (Best 2015): 285 people: healthy controls and six cancer types.

Raw read counts -> log2(CPM + 1). CPM is computed per sample, so it never sees labels.
Ensembl IDs are mapped to gene symbols with the GENCODE v23 probemap (UCSC Toil).

Confounding is a known issue here and is kept in `meta`:
  GSE89843: only VUMC contributed both classes; MGH/NKI sent only cancer, UMCU/AMC/UMEA
            only controls; cancer patients are ~17 years older on average.
  GSE68086: healthy controls come from batches 02-04 only.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Dataset, encode, fetch

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"
FILES = {
    "GSE89843": {
        "counts.txt.gz": f"{GEO}/GSE89nnn/GSE89843/suppl/GSE89843_TEP_Count_Matrix.txt.gz",
        "tableS1.xlsx": f"{GEO}/GSE89nnn/GSE89843/suppl/GSE89843_Best_et_al_Cancer_Cell_2017_Table_S1_conversion_to_GEO.xlsx",
    },
    "GSE68086": {
        "counts.txt.gz": f"{GEO}/GSE68nnn/GSE68086/suppl/GSE68086_TEP_data_matrix.txt.gz",
        "series_matrix.txt.gz": f"{GEO}/GSE68nnn/GSE68086/matrix/GSE68086_series_matrix.txt.gz",
    },
}
PROBEMAP_URL = ("https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/"
                "probeMap%2Fgencode.v23.annotation.gene.probemap")
GSE68086_TYPES = ["Breast", "CRC", "GBM", "Hepatobiliary", "Lung", "Pancreas"]


def download(study: str, data_dir, force: bool = False) -> Path:
    out = Path(data_dir) / "raw" / study
    for name, url in FILES[study].items():
        fetch(url, out / name, force=force)
    fetch(PROBEMAP_URL, Path(data_dir) / "raw" / "ref" / "gencode.v23.probemap", force=force)
    return out


def ensembl_to_symbol(data_dir) -> pd.Series:
    pm = pd.read_csv(Path(data_dir) / "raw" / "ref" / "gencode.v23.probemap", sep="\t")
    return pd.Series(pm["gene"].to_numpy(), index=pm["id"].str.split(".").str[0])


def log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """samples x genes raw counts -> log2(counts per million + 1), per sample."""
    lib = counts.sum(axis=1).replace(0, np.nan)
    return np.log2(counts.div(lib, axis=0) * 1e6 + 1).fillna(0).astype(np.float32)


def _to_symbols(expr: pd.DataFrame, data_dir) -> pd.DataFrame:
    """Rename Ensembl columns to symbols; for duplicated symbols keep the most expressed."""
    sym = ensembl_to_symbol(data_dir)
    expr = expr.loc[:, expr.columns.isin(sym.index)]
    names = sym.reindex(expr.columns).to_numpy()
    order = np.argsort(-expr.mean(axis=0).to_numpy(), kind="stable")
    keep = order[~pd.Series(names[order]).duplicated().to_numpy()]
    keep.sort()
    out = expr.iloc[:, keep]
    out.columns = names[keep]
    return out


def _load_counts(study: str, data_dir, sep: str) -> pd.DataFrame:
    cache = Path(data_dir) / "processed" / f"{study}_logcpm.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raw = Path(data_dir) / "raw" / study
    if not (raw / "counts.txt.gz").exists():
        download(study, data_dir)
    counts = pd.read_csv(raw / "counts.txt.gz", sep=sep, index_col=0).T
    counts.columns = counts.columns.astype(str).str.split(".").str[0]
    expr = _to_symbols(log_cpm(counts), data_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    expr.to_parquet(cache)
    return expr


def meta_gse89843(data_dir) -> pd.DataFrame:
    raw = Path(data_dir) / "raw" / "GSE89843"
    if not (raw / "tableS1.xlsx").exists():
        download("GSE89843", data_dir)
    t = pd.read_excel(raw / "tableS1.xlsx", header=2).dropna(subset=["Sample name"])
    t = t.loc[:, ~t.columns.astype(str).str.startswith("Unnamed")].drop(columns=["Legend"], errors="ignore")
    t = t.rename(columns={"GEO ID (GSE89843)": "sample", "Patient group": "group",
                          "Classification group": "label", "Hospital": "hospital", "Age": "age",
                          "Gender": "sex", "Smoking": "smoking", "Metastasis": "metastasis",
                          "Matched cohort": "matched_cohort", "Full cohort": "full_cohort",
                          "Storage time": "storage"})
    t["age"] = pd.to_numeric(t["age"], errors="coerce")
    return t.set_index("sample")


def meta_gse68086(data_dir) -> pd.DataFrame:
    """Series-matrix characteristics are misaligned across samples, so parse them by key."""
    raw = Path(data_dir) / "raw" / "GSE68086"
    if not (raw / "series_matrix.txt.gz").exists():
        download("GSE68086", data_dir)
    source, chars = None, []
    with gzip.open(raw / "series_matrix.txt.gz", "rt") as fh:
        for line in fh:
            vals = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            if line.startswith("!Sample_source_name_ch1"):
                source = vals
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append(vals)
    recs = [{} for _ in source]
    for ch in chars:
        for i, v in enumerate(ch):
            if ": " in v:
                k, val = v.split(": ", 1)
                recs[i][k] = val
    m = pd.DataFrame(recs, index=source).rename(columns={
        "patient id": "patient", "cancer type": "cancer_type", "mutational subclass": "subclass"})
    return m[["patient", "cancer_type", "batch", "subclass"]]


def load(spec: dict, data_dir) -> Dataset:
    study, task = spec["study"], spec["task"]
    if study == "GSE89843":
        expr = _load_counts(study, data_dir, sep=" ")
        meta = meta_gse89843(data_dir).reindex(expr.index)
        cancer = meta["label"] == "NSCLC"
        if task == "nsclc_vs_healthy":
            keep = cancer | (meta["group"] == "Healthy Control")
        elif task == "nsclc_vs_noncancer":
            keep = cancer | (meta["label"] == "Non-cancer")
        elif task == "nonmetastatic_vs_noncancer":
            keep = (cancer & (meta["metastasis"] == "N")) | (meta["label"] == "Non-cancer")
        else:
            raise ValueError(f"unknown GSE89843 task {task!r}")
        if spec.get("hospital"):
            keep &= meta["hospital"] == spec["hospital"]
        if spec.get("cohort"):  # e.g. matched_cohort: restrict to the authors' matched subset
            keep &= meta[spec["cohort"]].notna()
        y = cancer[keep].to_numpy(int)
        classes = ["non-cancer", "NSCLC"]
        groups = expr.index[keep].to_numpy()  # one sample per person
    elif study == "GSE68086":
        expr = _load_counts(study, data_dir, sep="\t")
        meta = meta_gse68086(data_dir).reindex(expr.index)
        is_hc = meta["cancer_type"] == "HC"
        keep = pd.Series(True, index=expr.index)
        if spec.get("batches"):
            keep &= meta["batch"].isin(spec["batches"])
        if task == "pancancer_vs_healthy":
            y = (~is_hc)[keep].to_numpy(int)
            classes = ["healthy", "cancer"]
        elif task == "cancer_type":
            keep &= ~is_hc
            y, classes = encode(meta.loc[keep, "cancer_type"], order=GSE68086_TYPES)
        else:
            raise ValueError(f"unknown GSE68086 task {task!r}")
        groups = meta.loc[keep, "patient"].to_numpy()
    else:
        raise ValueError(f"unknown platelet study {study!r}")
    e = expr.loc[keep]
    return Dataset(cancer=spec.get("name", study), task=task, X=e.to_numpy(np.float32), y=y,
                   groups=groups, genes=e.columns.to_numpy(), samples=e.index.to_numpy(),
                   classes=classes, meta=meta.loc[keep].reset_index(drop=True), source="platelets")
