"""Backwards-compatible import path for the v1 TCGA loader."""
from .sources.base import Dataset  # noqa: F401
from .sources.tcga import FILES, NORMAL, TUMOR, XENA_BASE, collapse_stage, download, load  # noqa: F401
