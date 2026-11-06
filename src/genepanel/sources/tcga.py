"""TCGA RNA-seq from UCSC Xena (legacy HiSeqV2, log2(RSEM norm count + 1)).

Tasks:
  tumor_vs_normal  primary tumor vs solid tissue normal (optionally one stage only)
  histology        which of several cancer types a primary tumor is (e.g. LUAD vs LUSC)
  pam50            BRCA intrinsic subtype (PAM50 call from RNA-seq)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import Dataset, encode, fetch

XENA_BASE = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"
FILES = {
    "HiSeqV2.gz": "TCGA.{c}.sampleMap/HiSeqV2.gz",  # log2(RSEM norm count + 1)
    "clinicalMatrix.tsv": "TCGA.{c}.sampleMap/{c}_clinicalMatrix",
}
TUMOR, NORMAL = "Primary Tumor", "Solid Tissue Normal"
PAM50_ORDER = ["LumA", "LumB", "Her2", "Basal", "Normal"]


def download(cancer: str, data_dir: str | Path, force: bool = False) -> Path:
    out = Path(data_dir) / "raw" / cancer
    for name, remote in FILES.items():
        fetch(f"{XENA_BASE}/{remote.format(c=cancer)}", out / name, force=force)
    return out


def collapse_stage(s: pd.Series) -> pd.Series:
    """'Stage IIIA' -> 'III'; anything unparseable (Stage X, [Discrepancy], NaN) -> 'NA'."""
    return s.astype(str).str.extract(r"^Stage (IV|I{1,3})")[0].fillna("NA")


def _load_raw(cancer: str, data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expression (samples x genes) and clinical table, cached as parquet."""
    raw = Path(data_dir) / "raw" / cancer
    cache = Path(data_dir) / "processed" / f"{cancer}_expr.parquet"
    if cache.exists():
        expr = pd.read_parquet(cache)
    else:
        if not (raw / "HiSeqV2.gz").exists():
            download(cancer, data_dir)
        expr = pd.read_csv(raw / "HiSeqV2.gz", sep="\t", index_col=0).T.astype(np.float32)
        expr = expr.loc[:, ~expr.columns.duplicated()]
        cache.parent.mkdir(parents=True, exist_ok=True)
        expr.to_parquet(cache)
    if not (raw / "clinicalMatrix.tsv").exists():
        download(cancer, data_dir)
    clin = pd.read_csv(raw / "clinicalMatrix.tsv", sep="\t", index_col=0, low_memory=False)
    return expr, clin


def _sample_masks(expr: pd.DataFrame, clin: pd.DataFrame):
    # Clinical sample_type is authoritative (it separates e.g. FFPE from frozen tumor);
    # the barcode type code must agree with it.
    code = expr.index.str[13:15]
    stype = clin["sample_type"]
    return (stype == TUMOR) & (code == "01"), (stype == NORMAL) & (code == "11")


def load(cancer: str, task: str, data_dir: str | Path, stage: str | None = None) -> Dataset:
    """Tumor vs normal (the v1 task)."""
    expr, clin = _load_raw(cancer, data_dir)
    clin = clin.reindex(expr.index)
    is_tumor, is_normal = _sample_masks(expr, clin)
    stg = collapse_stage(clin["pathologic_stage"])
    if stage is not None:
        is_tumor &= stg == stage
    keep = is_tumor | is_normal
    e = expr.loc[keep]
    return Dataset(
        cancer=cancer,
        task=task,
        X=e.to_numpy(np.float32),
        y=is_tumor[keep].to_numpy(int),
        groups=e.index.str[:12].to_numpy(),
        genes=e.columns.to_numpy(),
        samples=e.index.to_numpy(),
        stage=stg[keep].to_numpy(),
    )


def load_histology(cancers: list[str], name: str, task: str, data_dir) -> Dataset:
    """Primary tumors of several cancer types; label = cancer type. Genes = intersection."""
    parts = []
    for c in cancers:
        expr, clin = _load_raw(c, data_dir)
        is_tumor, _ = _sample_masks(expr, clin.reindex(expr.index))
        parts.append(expr.loc[is_tumor].assign(_label=c))
    genes = sorted(set.intersection(*(set(p.columns) - {"_label"} for p in parts)))
    e = pd.concat([p[genes + ["_label"]] for p in parts])
    e = e[~e.index.duplicated()]
    # with two cancers this is binary and the second listed one is the positive class
    y, classes = encode(e.pop("_label"), order=cancers)
    return Dataset(cancer=name, task=task, X=e.to_numpy(np.float32), y=y,
                   groups=e.index.str[:12].to_numpy(), genes=np.array(genes),
                   samples=e.index.to_numpy(), classes=classes)


def load_pam50(task: str, data_dir) -> Dataset:
    expr, clin = _load_raw("BRCA", data_dir)
    clin = clin.reindex(expr.index)
    is_tumor, _ = _sample_masks(expr, clin)
    lab = clin["PAM50Call_RNAseq"]
    keep = is_tumor & lab.isin(PAM50_ORDER)
    e = expr.loc[keep]
    y, classes = encode(lab[keep], order=PAM50_ORDER)
    return Dataset(cancer="BRCA", task=task, X=e.to_numpy(np.float32), y=y,
                   groups=e.index.str[:12].to_numpy(), genes=e.columns.to_numpy(),
                   samples=e.index.to_numpy(), classes=classes,
                   stage=collapse_stage(clin.loc[keep, "pathologic_stage"]).to_numpy())


def from_spec(spec: dict, data_dir) -> Dataset:
    kind = spec.get("kind", "tumor_vs_normal")
    if kind == "tumor_vs_normal":
        return load(spec["cancer"], spec["task"], data_dir, stage=spec.get("stage"))
    if kind == "histology":
        return load_histology(spec["cancers"], spec.get("name", "_".join(spec["cancers"])),
                              spec["task"], data_dir)
    if kind == "pam50":
        return load_pam50(spec["task"], data_dir)
    raise ValueError(f"unknown tcga task kind {kind!r}")
