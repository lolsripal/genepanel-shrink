"""Download and load TCGA RNA-seq (UCSC Xena) as model-ready matrices."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

XENA_BASE = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"
FILES = {
    "HiSeqV2.gz": "TCGA.{c}.sampleMap/HiSeqV2.gz",  # log2(RSEM norm count + 1)
    "clinicalMatrix.tsv": "TCGA.{c}.sampleMap/{c}_clinicalMatrix",
}
TUMOR, NORMAL = "Primary Tumor", "Solid Tissue Normal"


@dataclass
class Dataset:
    cancer: str
    task: str
    X: np.ndarray            # samples x genes, float32
    y: np.ndarray            # 1 = tumor, 0 = normal
    groups: np.ndarray       # patient id, for grouped CV
    genes: np.ndarray
    samples: np.ndarray
    stage: np.ndarray        # collapsed stage (I..IV / NA) of the patient's tumor


def download(cancer: str, data_dir: str | Path, force: bool = False) -> Path:
    out = Path(data_dir) / "raw" / cancer
    out.mkdir(parents=True, exist_ok=True)
    for name, remote in FILES.items():
        dest = out / name
        if dest.exists() and not force:
            continue
        url = f"{XENA_BASE}/{remote.format(c=cancer)}"
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            tmp.rename(dest)
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
    clin = pd.read_csv(raw / "clinicalMatrix.tsv", sep="\t", index_col=0, low_memory=False)
    return expr, clin


def load(cancer: str, task: str, data_dir: str | Path, stage: str | None = None) -> Dataset:
    expr, clin = _load_raw(cancer, data_dir)
    clin = clin.reindex(expr.index)
    # Clinical sample_type is authoritative (it separates e.g. FFPE from frozen tumor);
    # the barcode type code must agree with it.
    code = expr.index.str[13:15]
    stype = clin["sample_type"]
    is_tumor = (stype == TUMOR) & (code == "01")
    is_normal = (stype == NORMAL) & (code == "11")
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
