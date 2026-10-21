"""Command line: genepanel {download,run,analyze,biology,report} [--config ...]"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import data, preprocess


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_datasets(cfg):
    out = []
    for cancer in cfg["cancers"]:
        for task, opts in cfg["tasks"].items():
            ds = data.load(cancer, task, cfg["data_dir"], stage=(opts or {}).get("stage"))
            out.append(preprocess.filter_genes(ds, **cfg["preprocess"]))
    return out


def cmd_download(cfg, args):
    for c in cfg["cancers"]:
        print("downloaded", data.download(c, cfg["data_dir"], force=args.force))


def cmd_run(cfg, args):
    from .evaluate import run_experiment

    datasets = load_datasets(cfg)
    for ds in datasets:
        print(f"{ds.cancer}/{ds.task}: {len(ds.y)} samples ({ds.y.sum()} tumor, "
              f"{(1 - ds.y).sum()} normal), {len(ds.genes)} genes")
    m, _, t = run_experiment(datasets, cfg)
    print(f"{len(m)} evaluations; mean fold time {t.total_seconds.mean():.0f}s")


def cmd_analyze(cfg, args):
    from . import analysis

    analysis.run(cfg, load_datasets(cfg))


def cmd_biology(cfg, args):
    from . import biology

    biology.run(cfg, pubmed=not args.no_pubmed)


def cmd_report(cfg, args):
    from . import plots

    plots.make_all(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="genepanel")
    ap.add_argument("command", choices=["download", "run", "analyze", "biology", "report", "all"])
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--force", action="store_true", help="re-download data")
    ap.add_argument("--no-pubmed", action="store_true", help="skip PubMed co-mention counts")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    cmds = {"download": cmd_download, "run": cmd_run, "analyze": cmd_analyze,
            "biology": cmd_biology, "report": cmd_report}
    for name in (["download", "run", "analyze", "biology", "report"] if args.command == "all"
                 else [args.command]):
        cmds[name](cfg, args)


if __name__ == "__main__":
    main()
