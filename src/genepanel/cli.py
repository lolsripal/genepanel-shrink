"""Command line: genepanel {download,run,analyze,biology,report,confounding,external,all} [--config ...]"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml

from . import preprocess
from .sources import load_spec, specs_from_config

OVERRIDABLE = {"cv", "selection", "models", "all_genes_models", "random_baseline",
               "preprocess", "prediction_models"}


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def dataset_config(cfg: dict, spec: dict) -> dict:
    """Global config with this dataset's `overrides:` applied."""
    over = spec.get("overrides") or {}
    bad = set(over) - OVERRIDABLE
    if bad:
        raise ValueError(f"cannot override {sorted(bad)} per dataset")
    return _merge(cfg, over)


def load_items(cfg):
    """[(Dataset, per-dataset config, spec)] for every dataset in the config."""
    out = []
    for spec in specs_from_config(cfg):
        dcfg = dataset_config(cfg, spec)
        ds = load_spec(spec, cfg["data_dir"])
        if dcfg.get("preprocess"):
            ds = preprocess.filter_genes(ds, **dcfg["preprocess"])
        key = {k: v for k, v in spec.items() if k != "overrides"}
        out.append((ds, dcfg, key))
    return out


def load_datasets(cfg):
    return [ds for ds, _, _ in load_items(cfg)]


def cmd_download(cfg, args):
    from .sources import platelets, tcga, toil

    for spec in specs_from_config(cfg):
        src = spec.get("source", "tcga")
        if src == "tcga":
            for c in spec.get("cancers", [spec.get("cancer", "BRCA")]):
                print("downloaded", tcga.download(c, cfg["data_dir"], force=args.force))
        elif src == "platelets":
            print("downloaded", platelets.download(spec["study"], cfg["data_dir"], force=args.force))
        elif src == "toil":
            print("downloaded", toil.download(cfg["data_dir"], force=args.force))


def cmd_run(cfg, args):
    from .evaluate import run_experiment

    items = load_items(cfg)
    for ds, _, _ in items:
        counts = ", ".join(f"{n} {c}" for c, n in zip(ds.classes, np.bincount(ds.y)))
        print(f"{ds.cancer}/{ds.task}: {len(ds.y)} samples ({counts}), {len(ds.genes)} genes")
    m, _, t = run_experiment(items, cfg)
    print(f"{len(m)} evaluations; mean fold time {t.total_seconds.mean():.0f}s")


def cmd_analyze(cfg, args):
    from . import analysis

    analysis.run(cfg, load_items(cfg))


def cmd_biology(cfg, args):
    from . import biology

    biology.run(cfg, pubmed=not args.no_pubmed)


def cmd_report(cfg, args):
    from . import plots

    plots.make_all(cfg)


def cmd_confounding(cfg, args):
    from . import confounding

    confounding.run(cfg, load_items(cfg))


def cmd_external(cfg, args):
    from . import external

    external.run(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="genepanel")
    ap.add_argument("command", choices=["download", "run", "analyze", "biology", "report",
                                        "confounding", "external", "all"])
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--force", action="store_true", help="re-download data")
    ap.add_argument("--no-pubmed", action="store_true", help="skip PubMed co-mention counts")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    cmds = {"download": cmd_download, "run": cmd_run, "analyze": cmd_analyze,
            "biology": cmd_biology, "report": cmd_report, "confounding": cmd_confounding,
            "external": cmd_external}
    steps = cfg.get("pipeline", ["download", "run", "analyze", "biology", "report"])
    for name in (steps if args.command == "all" else [args.command]):
        cmds[name](cfg, args)


if __name__ == "__main__":
    main()
