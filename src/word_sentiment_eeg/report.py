"""Professor-facing class exports and a compact report, all saved on Drive."""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_npz, write_json
from .data import BANDS


def class_exports(table, x, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    coverage = table.groupby(["task", "subject", "lexical_class", "eeg_status"], dropna=False).size().rename("n").reset_index()
    coverage.to_csv(out / "coverage_by_task_reader_class.csv", index=False)
    table.groupby(["task", "lexical_class"]).agg(
        word_occurrences=("item_id", "nunique"), word_forms=("word_form", "nunique"),
        reader_word_rows=("row_id", "size"), context_flagged=("needs_context_review", "sum")
    ).reset_index().to_csv(out / "class_counts.csv", index=False)
    rows = []
    if 'lexical_context_agree' in table:
        other = table[~table.lexical_context_agree | table.contextual_class.eq('uncertain')]
        other.to_csv(out / 'context_disagreement_and_uncertainty.csv', index=False)
        table.loc[table.eeg_status.eq('ok'), ['row_id','task','subject','word','word_index',
            'lexical_class','contextual_class','source_file','source_word_row']].to_csv(out / 'contextual_class_source_index.csv', index=False)
    for task, task_rows in table.groupby("task"):
        for label in ("negative", "neutral", "positive"):
            subset = task_rows[(task_rows.lexical_class == label) & (task_rows.eeg_status == "ok")]
            if 'lexical_context_agree' in subset:
                subset = subset[subset.lexical_context_agree]
            ids = subset.row_id.to_numpy(int)
            # Actual EEG band vectors, not a list of class labels or a sentence average.
            atomic_npz(out / "eeg_by_class" / task / f"{label}.npz",
                       features=x[ids], row_id=ids, bands=np.array(BANDS),
                       subject=subset.subject.to_numpy(str), item_id=subset.item_id.to_numpy(str),
                       context_flagged=subset.needs_context_review.to_numpy(bool))
            subset.to_csv(out / "eeg_by_class" / task / f"{label}_metadata.csv", index=False)
            # One mean per reader/class; readers then receive equal weight in plots.
            for subject, group in subset.groupby("subject"):
                values = x[group.row_id.to_numpy(int)]
                mean = values.mean(axis=0)
                for b, band in enumerate(BANDS):
                    rows.append({"task": task, "class": label, "subject": subject, "band": band,
                                 "n_words": len(group), "mean_across_105_channels": float(mean[b].mean())})
    pd.DataFrame(rows).to_csv(out / "reader_class_band_means.csv", index=False)
    write_json(out / "eeg_by_class" / "README.json", {
        "shape": "reader_word_rows × 8 bands × 105 electrodes",
        "representation": "ZuCo precomputed TRT word band power, not raw time-series EEG",
        "source": "Original preprocessed word caches, unchanged physical feature values",
        "row_id": "Joins to all_words.csv for exact source-file/word-row/reader mapping",
        "context": ("Primary exports contain only lexical/contextual agreement. Disagreements and uncertain rows are kept in separate CSVs; the contextual source index points to the unchanged original caches."
                    if 'lexical_context_agree' in table else "Legacy lexical comparison: all labeled complete-EEG rows exported; context flags remain in metadata."),
        "independence": "Repeated readers/words are dependent. These exports are descriptive, not an inferential class-effect test."
    })


def make_plots(table, out, eeg=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib unavailable; CSV and NPZ outputs still saved"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts = table.drop_duplicates(["task", "item_id"]).groupby(["task", "lexical_class"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot.bar(ax=ax, rot=0, color=["#b6a7c9", "#bd5e61", "#87929b", "#3b9575", "#d9c48d"][:len(counts.columns)])
    ax.set_ylabel("Unique word occurrences (readers counted once)")
    ax.set_title("Word label coverage across ZuCo 1.0 tasks")
    ax.legend(title="Lexical label", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "class_coverage.png", dpi=180)
    plt.close(fig)
    if eeg is not None:
        labels = ["negative", "neutral", "positive"]
        tasks = sorted(table.task.unique())
        fig, axes = plt.subplots(len(tasks), 1, figsize=(10, 3.3 * len(tasks)), squeeze=False)
        export_rows = []
        for task, ax in zip(tasks, axes[:, 0]):
            means = {label: [] for label in labels}
            eligible = table[(table.task == task) & (table.eeg_status == "ok")]
            if 'lexical_context_agree' in eligible:
                eligible = eligible[eligible.lexical_context_agree]
            for subject, reader in eligible.groupby("subject"):
                values = eeg[reader.row_id.to_numpy(int)].mean(axis=2)
                mean, sd = values.mean(axis=0), values.std(axis=0)
                z = (values - mean) / np.maximum(sd, 1e-8)
                for label in labels:
                    mask = reader.lexical_class.eq(label).to_numpy()
                    if mask.any():
                        value = z[mask].mean(axis=0)
                        means[label].append(value)
                        for band, v in zip(BANDS, value):
                            export_rows.append({"task": task, "subject": subject, "class": label, "band": band,
                                                "reader_standardized_mean": float(v), "n_word_rows": int(mask.sum())})
            for label, color in zip(labels, ["#bd5e61", "#87929b", "#3b9575"]):
                if means[label]:
                    ax.plot(BANDS, np.mean(means[label], axis=0), marker="o", label=label, color=color)
            ax.axhline(0, color="#d1d5db", linewidth=1)
            ax.set_title(f"{task}: equal-reader class means (descriptive; primary eligible word rows)")
            ax.set_ylabel("Within-reader standardized band value")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "class_band_profiles.png", dpi=180)
        plt.close(fig)
        pd.DataFrame(export_rows).to_csv(out.parent / "reader_standardized_class_band_means.csv", index=False)
    return "saved"


def write_report(run_dir, table, inventory, evaluations, manifest):
    run_dir = Path(run_dir)
    n_items = table.item_id.nunique()
    eligible = table.eeg_status.eq("ok") & table.lexical_class.isin(["negative", "neutral", "positive"])
    prepared = manifest.get('label_provenance', {}).get('mode') == 'prepared_agent_annotations'
    provenance = manifest.get('label_provenance', {})
    explanation = (f"This run uses prepared single-agent annotations covering {provenance.get('annotation_sentences', 0):,} distinct source sentences and {provenance.get('annotation_rows', 0):,} word occurrences. Both lexical and contextual labels were assigned before this EEG run. These are agent annotations, not human consensus or a measure of the reader's emotional state. The primary screen uses words whose lexical and contextual class agree; other labels remain in the full audit."
                   if prepared else "This explicit legacy comparison uses automatic NRC lexical labels, not the prepared agent annotations.")
    lines = ["# Word sentiment EEG — report", "", explanation, "",
             f"- Unique task × sentence × word occurrences: {n_items:,}",
             f"- Reader × word rows: {len(table):,}; complete EEG and a class: {int(eligible.sum()):,}",
             f"- Context review candidates (unique items): {table.loc[table.needs_context_review, 'item_id'].nunique():,}",
             f"- Dataset identity: ZuCo 1.0; all requested tasks: {', '.join(manifest['config']['tasks'])}",
             "- Signal representation: precomputed word TRT band power (8 × 105), not sentence EEG or raw waveforms.",
             ("- Labels are categorical agent decisions: positive, negative, neutral or uncertain. Neutral is an explicit decision within a reviewed sentence, not a missing-word fallback. Unmatched cache words fail alignment validation or receive the documented alignment exclusion."
              if prepared else "- NRC comparison thresholds are recorded in the run config; unknown entries never become neutral."),
             "- Negation, irony, idioms and proper-name uses are documented in annotation flags and notes. They are not mechanically inverted or copied from sentence sentiment.", "",
             "## Data coverage", "", "| Task | Subject files | Missing expected readers |", "|---|---:|---|",
             *[f"| {s['task']} | {len(s['files'])} | {', '.join(s['missing_expected_subjects']) or 'none'} |" for s in inventory],
             "", "## EEG-only screen", "",
             "Each task is evaluated separately. All readers of a sentence remain in one split. The stricter protocol also removes every test word form from training. Word forms are normalized spellings, not lemmas; morphological relatives may remain. Readers may overlap unless the optional subject protocol is selected.", "",
             "| Task | Classes | Protocol | Status / folds | EEG balanced accuracy | Nuisance | Majority |", "|---|---|---|---|---:|---:|---:|"]
    for result in evaluations:
        scores = result.get("scores", {})
        score = lambda k: f"{scores[k]['balanced_accuracy']:.3f}" if k in scores else "—"
        lines.append(f"| {result['task']} | {len(result['classes'])} | {result['protocol']} | {result['status']} ({result['completed_folds']}/{result['total_folds']}) | {score('eeg')} | {score('nuisance')} | {score('majority')} |")
    lines += ["", "The nuisance model uses length, training-corpus frequency, sentence position/length and available fixation durations/counts. The ridge strength and train-only PCA size were fixed before the run. No model selection uses test scores.", "",
              "Permutation p-values in evaluation JSON are exploratory word-form label randomization screens, uncorrected across tasks/protocols. They preserve repeated instances of a word form, but do not isolate sentiment from all correlated semantics/context. Partial folds and absent classes prevent a complete result; do not report only the successful folds as the planned experiment.", "",
              "## Files to review", "",
              "- `all_words.csv.gz`: every original word row, including missing EEG and unmatched labels.",
              "- `word_inventory.csv`: one row per word occurrence shared across readers.",
              ("- `context_review.csv`: agent-uncertain or unresolved-alignment rows for optional professor review; other contextual labels are already prepared."
               if prepared else "- `context_review.csv`: optional contextual annotation sheet for the legacy comparison."),
              "- `eeg_by_class/<task>/<class>.npz` and metadata CSV: actual word band-power arrays by class.",
              "- `evaluations/`: metrics, exact splits via row IDs, fold coverage, predictions and null values.",
              "- `raw_export.json` (when requested): availability and paths of original word.rawEEG fixation segments, grouped by class under CachedArtifacts. Raw segments are variable-length preprocessed recordings, not fixed-window/deconvolved FRPs. Missing raw fields/tasks are reported without substituting sentence signals.",
              "- `manifest.json`, `data_inventory.json`, `run.log`: provenance, source hashes, versions and progress.", "",
              "## Interpretation and next step", "",
              "The earlier sentence-level negative results do not demonstrate that EEG has no sentiment information. This run is a different, word-level hypothesis. Conversely, a difference in class-average EEG does not establish single-trial decoding or a causal sentiment effect.", "",
              "First inspect class balance, independent word forms, task/reader coverage and the context review examples. Compare EEG with the nuisance controls and any applicable permutation screen, especially under sentence + word-form holdout. A word-form label permutation is not computed when that form has multiple sense-dependent class labels; JSON reports this explicitly. Keep binary and ternary results separate. Any promising pattern needs a prespecified confirmatory follow-up with reader generalization, stronger confound matching and multiplicity control.", "",
              "A directly relevant precedent is [Pfeiffer et al. (2020)](https://doi.org/10.1016/j.neuroimage.2020.116934): fixation-related word sentiment analysis on ZuCo SR, with negation exclusions and deconvolution. Their reported decoding used trial averages; this band-power screen is not a reproduction of that method.", ""]
    (run_dir / "REPORT.md").write_text("\n".join(lines))
