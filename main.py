"""One command: audit all ZuCo 1.0 tasks, label words, export class EEG, run probes."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd

from word_sentiment_eeg.common import digest, file_digest, write_json
from word_sentiment_eeg.data import inspect_sources, load_sources
from word_sentiment_eeg.evaluate import run_probe
from word_sentiment_eeg.labels import annotate, apply_reviews, ensure_lexicon, load_lexicon
from word_sentiment_eeg.report import class_exports, make_plots, write_report


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()


def execute(args, config, run_dir):
    inventory = inspect_sources(args.thesis_root, config)
    write_json(run_dir / "data_inventory.json", inventory)
    table, x, sources = load_sources(inventory)
    lexicon_path = Path(args.lexicon) if args.lexicon else ensure_lexicon(
        Path(args.thesis_root) / "CachedArtifacts/word_sentiment_eeg/lexicons/NRC_VAD_v1")
    lexicon = load_lexicon(lexicon_path)
    fingerprint = digest({"config": config, "sources": sources, "lexicon": file_digest(lexicon_path),
                          "reviews": file_digest(args.reviews) if args.reviews else None,
                          "raw_dir": args.raw_dir,
                          "raw_inventory": [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                            for p in sorted(Path(args.raw_dir).rglob("results*_*.mat"))] if args.raw_dir else [],
                          "code": {str(p.relative_to(Path(__file__).parent)): file_digest(p)
                                   for p in [Path(__file__), *sorted((Path(__file__).parent / 'src').rglob('*.py'))]}})
    old_manifest = run_dir / "manifest.json"
    if old_manifest.exists():
        old = json.loads(old_manifest.read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError("Inputs/code/config changed. Use a new --run-tag to preserve earlier results.")
        if old["status"] == "completed" and (run_dir / "REPORT.md").exists():
            print("Identical completed run found; reusing saved outputs. REPORT.md:", run_dir / "REPORT.md")
            return
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit = "uncommitted"
    versions = {}
    for package in ("numpy", "pandas", "scipy", "h5py", "matplotlib", "threadpoolctl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    manifest = {"schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "fingerprint": fingerprint, "sources": sources, "git_commit": commit,
                "lexicon_sha256": file_digest(lexicon_path), "versions": versions,
                "python": platform.python_version(), "reviews": args.reviews,
                "dataset_scope": "ZuCo 1.0 SR/NR/TSR caches; ZuCo 2.0 and TeCo not included"}
    write_json(old_manifest, manifest)
    table = annotate(table, lexicon, config["label_thresholds"])
    table = apply_reviews(table, args.reviews)
    table.to_csv(run_dir / "all_words.csv.gz", index=False, compression="gzip")
    unique = table.drop_duplicates("item_id").drop(columns=["subject", "row_id", "source_file", "source_word_row", "source_sentence_row", "sample_id_original", "eeg_status", "n_fixations", "trt_ms", "ffd_ms", "gd_ms"])
    unique.to_csv(run_dir / "word_inventory.csv", index=False)
    review = unique[unique.needs_context_review | unique.lexical_class.isin(["unknown", "ambiguous"])].copy()
    review["reviewer"] = ""
    review["notes"] = ""
    review.to_csv(run_dir / "context_review.csv", index=False)
    class_exports(table, x, run_dir)
    if args.raw_dir:
        from word_sentiment_eeg.raw_export import export_available_raw
        raw_report = export_available_raw(table, args.raw_dir,
            Path(args.thesis_root) / "CachedArtifacts/word_sentiment_eeg/raw_by_class", fingerprint)
        write_json(run_dir / "raw_export.json", raw_report)
        manifest["raw_export_status"] = raw_report["status"]
    manifest["plot_status"] = make_plots(table, run_dir / "plots", x)
    evaluations = []
    for task in config["tasks"]:
        for classes in (["negative", "positive"], ["negative", "neutral", "positive"]):
            selected = table.task.eq(task) & table.eeg_status.eq("ok") & table.lexical_class.isin(classes)
            if config["exclude_context_flagged"]:
                selected &= ~table.needs_context_review
            selected &= ~table.contextual_class.isin(["exclude", "uncertain"])
            subset = table.loc[selected].copy()
            for protocol in config["protocols"]:
                tag = f"{task}_{len(classes)}class_{protocol}"
                path = run_dir / "evaluations" / f"{tag}.json"
                if path.exists():
                    result = json.loads(path.read_text())
                    if result.get("fingerprint") == fingerprint:
                        print("Reusing", tag, flush=True)
                        evaluations.append(result)
                        continue
                print(f"Evaluating {tag}: {len(subset):,} rows", flush=True)
                if len(subset):
                    result, predictions = run_probe(subset, x[subset.row_id.to_numpy(int)].reshape(len(subset), -1), config, protocol, classes)
                else:
                    result = {"protocol": protocol, "classes": classes, "status": "insufficient_data", "completed_folds": 0, "total_folds": config["folds"], "folds": []}
                    predictions = pd.DataFrame(columns=["row_id", "fold", "true", "eeg", "nuisance", "eeg_plus_nuisance", "majority"])
                result.update(task=task, fingerprint=fingerprint)
                write_json(path, result)
                predictions.to_csv(path.with_suffix(".predictions.csv"), index=False)
                evaluations.append(result)
                write_report(run_dir, table, inventory, evaluations, manifest)
    manifest.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                    experiment_status="complete" if all(e["status"] == "completed" for e in evaluations) else "coverage_limited",
                    rows=len(table), unique_items=table.item_id.nunique(),
                    missing_task_data=[s["task"] for s in inventory if not s["files"]])
    write_json(old_manifest, manifest)
    write_report(run_dir, table, inventory, evaluations, manifest)
    (run_dir / "failure.json").unlink(missing_ok=True)
    print("Saved report:", run_dir / "REPORT.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis-root", default="/content/drive/MyDrive/Thesis")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/default.json"))
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--lexicon", help="Existing NRC VAD v1 TSV; otherwise fetched in Colab only")
    parser.add_argument("--reviews", help="Optional reviewed contextual annotation CSV")
    parser.add_argument("--raw-dir", help="Export available ZuCo 1.0 word.rawEEG from this directory; no downloads")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.run_tag:
        config["run_tag"] = args.run_tag
    if Path(config["run_tag"]).name != config["run_tag"] or config["run_tag"] in {".", ".."}:
        raise ValueError("run_tag must be a simple folder name")
    run_dir = Path(args.thesis_root) / "Results/word_sentiment_eeg" / config["run_tag"]
    run_dir.mkdir(parents=True, exist_ok=True)
    # Bound BLAS threads on Colab; no library installation is attempted here.
    try:
        from threadpoolctl import threadpool_limits
        context = threadpool_limits(limits=2)
    except ImportError:
        from contextlib import nullcontext
        context = nullcontext()
    with open(run_dir / "run.log", "a", buffering=1) as log, context:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                execute(args, config, run_dir)
            except Exception:
                traceback.print_exc()
                write_json(run_dir / "failure.json", {"status": "failed", "time_utc": datetime.now(timezone.utc).isoformat(),
                                                       "message": "See run.log for the exception. Completed stages remain on Drive."})
                raise


if __name__ == "__main__":
    main()
