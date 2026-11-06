"""UCSC Toil recompute: TCGA and GTEx processed by one RNA-seq pipeline.

Source matrix: log2(TPM + 0.001), 60,498 GENCODE v23 genes x 19,131 samples (1.3 GB gz).
We stream it once, keep only the samples we need and the genes that also appear in the
v1 TCGA (Xena HiSeqV2) gene list, convert to log2(TPM + 1) so thresholds are on a scale
similar to v1, and cache the subset as parquet.

Tasks (spec `kind`):
  tumor_vs_gtex     TCGA primary tumor vs GTEx healthy tissue (breast or lung)
  normal_vs_gtex    TCGA adjacent normal vs GTEx healthy: a study-effect control. If this is
                    easy, part of any tumor-vs-GTEx signal is batch, not biology.
  tissue_of_origin  which of the 12 largest TCGA cancer types a primary tumor is
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Dataset, encode, fetch

BASE = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download"
FILES = {
    "tpm.gz": f"{BASE}/TcgaTargetGtex_rsem_gene_tpm.gz",
    "phenotype.txt.gz": f"{BASE}/TcgaTargetGTEX_phenotype.txt.gz",
}
PROBEMAP_URL = f"{BASE}/probeMap%2Fgencode.v23.annotation.gene.probemap"
TISSUES = {
    "breast": {"tcga": "Breast Invasive Carcinoma", "gtex": "Breast - Mammary Tissue"},
    "lung": {"tcga": "Lung Adenocarcinoma", "gtex": "Lung"},
}
N_TYPES, PER_TYPE = 12, 300  # tissue-of-origin: 12 largest cancer types, <= 300 tumors each


def download(data_dir, force: bool = False) -> Path:
    out = Path(data_dir) / "raw" / "toil"
    for name, url in FILES.items():
        fetch(url, out / name, force=force)
    fetch(PROBEMAP_URL, Path(data_dir) / "raw" / "ref" / "gencode.v23.probemap", force=force)
    return out


def phenotype(data_dir) -> pd.DataFrame:
    p = Path(data_dir) / "raw" / "toil" / "phenotype.txt.gz"
    if not p.exists():
        download(data_dir)
    ph = pd.read_csv(p, sep="\t", encoding="latin-1").set_index("sample")
    return ph.rename(columns={"primary disease or tissue": "disease", "_primary_site": "site",
                              "_sample_type": "sample_type", "_study": "study", "_gender": "sex"})


def tissue_types(ph: pd.DataFrame) -> list[str]:
    t = ph[(ph.study == "TCGA") & (ph.sample_type == "Primary Tumor")]
    return t.disease.value_counts().index[:N_TYPES].tolist()


def tissue_sample_ids(ph: pd.DataFrame, seed: int = 0) -> pd.Index:
    """The capped, seeded tissue-of-origin sample (<= PER_TYPE tumors per type)."""
    prim = ph[(ph.study == "TCGA") & (ph.sample_type == "Primary Tumor")]
    rng = np.random.default_rng(seed)
    out = []
    for d in tissue_types(ph):
        ids = np.sort(prim.index[prim.disease == d].to_numpy())
        out.extend(rng.choice(ids, size=min(PER_TYPE, len(ids)), replace=False))
    return pd.Index(sorted(out))


def needed_samples(ph: pd.DataFrame, seed: int = 0) -> pd.Index:
    keep = [set(tissue_sample_ids(ph, seed))]
    for t in TISSUES.values():
        keep.append(set(ph.index[(ph.study == "TCGA") & (ph.disease == t["tcga"])
                                 & ph.sample_type.isin(["Primary Tumor", "Solid Tissue Normal"])]))
        keep.append(set(ph.index[(ph.study == "GTEX") & (ph.disease == t["gtex"])]))
    return pd.Index(sorted(set().union(*keep)))


def _universe(data_dir) -> set[str]:
    """Genes in the v1 Xena HiSeqV2 matrix, so all chapters share one gene namespace."""
    import pyarrow.parquet as pq

    p = Path(data_dir) / "processed" / "BRCA_expr.parquet"
    if p.exists():
        return set(pq.read_schema(p).names) - {"__index_level_0__", "sample"}
    from .tcga import _load_raw
    return set(_load_raw("BRCA", data_dir)[0].columns)


def extract(data_dir) -> pd.DataFrame:
    """Stream the 1.3 GB matrix once and cache the needed subset (samples x genes)."""
    cache = Path(data_dir) / "processed" / "toil_subset.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raw = Path(data_dir) / "raw" / "toil"
    if not (raw / "tpm.gz").exists():
        download(data_dir)
    ph = phenotype(data_dir)
    want = set(needed_samples(ph))
    pm = pd.read_csv(Path(data_dir) / "raw" / "ref" / "gencode.v23.probemap", sep="\t")
    universe = _universe(data_dir)
    sym = dict(zip(pm["id"], pm["gene"]))
    genes, rows = [], []
    with gzip.open(raw / "tpm.gz", "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        cols = [i for i, s in enumerate(header) if s in want]
        samples = [header[i] for i in cols]
        cols = np.array(cols)
        seen = set()
        for line in fh:
            gid, rest = line.split("\t", 1)
            g = sym.get(gid)
            if g is None or g not in universe or g in seen:
                continue
            seen.add(g)
            vals = np.array(line.rstrip("\n").split("\t"), dtype=object)[cols].astype(np.float32)
            rows.append(vals)
            genes.append(g)
    X = np.vstack(rows).T  # samples x genes, log2(TPM + 0.001)
    X = np.log2(np.clip(np.exp2(X) - 0.001, 0, None) + 1).astype(np.float32)
    df = pd.DataFrame(X, index=samples, columns=genes)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def patient(sample: pd.Index) -> np.ndarray:
    """TCGA-XX-XXXX-01 -> TCGA-XX-XXXX; GTEX-XXXX-... -> GTEX-XXXX (donor)."""
    s = pd.Series(sample)
    return np.where(s.str.startswith("GTEX"), s.str.split("-").str[:2].str.join("-"), s.str[:12])


def load(spec: dict, data_dir) -> Dataset:
    expr = extract(data_dir)
    ph = phenotype(data_dir).reindex(expr.index)
    kind, task = spec["kind"], spec["task"]
    classes = ["healthy", "cancer"]
    if kind in ("tumor_vs_gtex", "normal_vs_gtex"):
        t = TISSUES[spec["tissue"]]
        tcga = (ph.study == "TCGA") & (ph.disease == t["tcga"])
        pos = tcga & (ph.sample_type == ("Primary Tumor" if kind == "tumor_vs_gtex" else "Solid Tissue Normal"))
        neg = (ph.study == "GTEX") & (ph.disease == t["gtex"])
        keep = pos | neg
        y = pos[keep].to_numpy(int)
        if kind == "normal_vs_gtex":
            classes = ["GTEx normal", "TCGA adjacent normal"]
    elif kind == "tissue_of_origin":
        full = phenotype(data_dir)
        types = tissue_types(full)
        keep = pd.Series(ph.index.isin(tissue_sample_ids(full)), index=ph.index)
        y, classes = encode(ph.loc[keep, "disease"], order=types)
    else:
        raise ValueError(f"unknown toil task kind {kind!r}")
    e = expr.loc[keep]
    return Dataset(cancer=spec.get("name", spec.get("tissue", "TOIL")), task=task,
                   X=e.to_numpy(np.float32), y=y, groups=patient(e.index), genes=e.columns.to_numpy(),
                   samples=e.index.to_numpy(), classes=classes,
                   meta=ph.loc[keep].reset_index(drop=True), source="toil")
