# Word sentiment EEG — prepared agent annotations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/short.ipynb)

## Current run: word onsets and class ERPs

**Review an already completed run:** open [the saved-results review notebook](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/review.ipynb),
use CPU, and choose **Runtime → Run all**. It audits the 36 saved reader/task
records and portable ZIP, then makes subject-specific plots from the saved
class averages. It does not download source EEG or rerun extraction. New
outputs go beside the existing results under `audit/` and `plots/`. This also
exports a word-selection table with both labels, sentence context and actual EEG
inclusion counts, and adds six fixed geometric scalp regions with a sensor map.
These regions describe scalp measurements, not localized cortical generators.

**See the labels now:** [master word-label CSV](annotations/word_labels.csv)
(20,458 word occurrences; includes the whole sentence, lexical/contextual labels,
notes and alignment flags). After the review notebook completes,
`audit/word_selection.csv` adds actual accepted EEG counts, while
`audit/accepted_word_labels.csv` contains words with at least one accepted event.

**Persian TeCo:** [input-preflight notebook](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/teco_preflight.ipynb)
follows the user's existing TeCo notebooks and native schema. Native sentence
EEG is `raw_data` (126 channels, documented 1200 Hz); per-fixation waveforms are
`word.rawEEG`. A bounded check of existing files matched **240/240 sampled first
fixations exactly across all 12 readers** and joined all 165 sentence texts per
reader. See [implementation trace and evidence limits](docs/TECO_IMPLEMENTATION_TRACE.md).
The notebook exports token/structure reports and repeats a bounded native-onset
check on one Drive file. It is not an ERP analysis or Persian word annotation.
`TRT_total` remains time-collapsed; full onset validation and Persian-context
labels are still needed for class ERPs. No dataset is downloaded or copied.
Persian texts and research files are not committed to this public repository.

The existing neutral class means neutral **words in their local context**, pooled
across sentence contexts. It is not restricted to globally neutral sentences.
A neutral-word/neutral-sentence control is a separate analysis that needs
sentence-context annotation and new averages from source epochs; the existing
pooled means cannot be filtered back into that subset. NR/TSR task membership is
not itself a neutral-sentence label. Context must be matched or stratified for
all classes to avoid changing only the neutral group's sentence context.

Real source data have now been checked: the prior SR audit verified word
waveforms for 9 readers, and a fresh cloud sample verified word segments and
onset recovery in NR and TSR. Three SR readers lacked `word.rawEEG`. See
[verified availability and scope](docs/DATA_AVAILABILITY.md); this is not a
claim that every task/reader has usable waveforms.

The short notebook now runs **`run_compact_erp.py`**, using the already-prepared labels.
Open the [GitHub notebook in Colab](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/short.ipynb),
select a CPU runtime, and choose **Runtime → Run all**. Allow Drive mounting.
The repository is public: **no GitHub token or Colab secret is required**.
No annotation adjudication or code editing is required. Code stays on GitHub
and in Colab's temporary runtime; onset tables and class ERP results are
saved to Drive. The older feature/classification runner remains `main.py`; it is not
run by the short notebook.

After the Drive storage limit was reached, the default notebook switched to
compact storage: it keeps per-word onsets/labels/exclusions and full 105-channel
class averages, without archiving individual EEG trial arrays. It reuses valid
old analyses and leaves all original caches untouched. Outputs are isolated in
`Thesis/Results/word_sentiment_eeg/v4_word_erp_compact/`, including the portable
`compact_results.zip`. Estimated additional storage is 100–200 MB including
the ZIP. The notebook stops if mounted Drive space is below 1 GiB. Pending
uploads can consume recently freed space. See [compact run details](docs/COMPACT_RUN.md).

The original full-trial mode remains available in `notebooks/full_trials.ipynb`
and `run_erp.py`; its larger storage requirements and repair behavior are
described below.

The extraction cell explicitly streams live command output into Colab. During
quiet stages, it prints an elapsed-time/status message every 30 seconds and
updates `progress.json` in the results folder on Drive. A heartbeat confirms
the child process is alive, not that it has completed another reader or that a
particular percentage is finished. Drive heartbeat writes use a separate thread
so a blocked upload cannot stop notebook log streaming. Already-running cells do
not acquire this display change automatically; use the updated notebook on the
next start/resume rather than interrupting a healthy run just for the display.

- Covers SR, NR and TSR separately. Uncertain labels, lexical/context
  disagreements, and alignment warnings are excluded automatically.
- Locates the **first stored fixation** in `sentenceData.rawData`, validating
  the entire segment against all 105 channels. A rejected first fixation is
  never silently replaced by a later one. Nonunique/unmatched segments are
  excluded. Exact matching and constant per-channel offsets are recorded.
- Saves onsets relative to sentence EEG (zero-based samples and milliseconds),
  with original word positions, sentence text, class, event IDs and exclusions.
- Epochs: **−200 to +800 ms**, 500 Hz, average reference, 30 Hz zero-phase
  low-pass applied to the whole sentence, −200..0 ms baseline correction,
  rejection above ±100 µV after baseline correction. Source EEG is already
  preprocessed by ZuCo; these are not acquisition-raw recordings.
- Saves all-channel reader averages and equal-reader
  grand averages. The same readers (≥5 accepted epochs in **each** class) are
  used for all three comparison curves. Tasks are not pooled. Individual trial
  EEG is saved only in the separate full-trial mode.

The authors' original [SR](https://osf.io/download/369ur/),
[NR](https://osf.io/download/gan9v/), and [TSR](https://osf.io/download/mksxg/)
MATLAB scripts slice word EEG from fixation event latency to latency+duration.
The stored fixation duration is in **samples**, not milliseconds; segment
length is duration+1 because MATLAB bounds are inclusive. We use the actual
segment position and sampling rate, not a cache field named `ffd_ms`, to time
these epochs. First-fixation onset is not sentence stimulus onset.

Existing `.mat` files under the three configured `Thesis/Data/zuco_*_raw`
folders are reused. Missing sources download in Colab from the authors' OSF
inventory (`configs/zuco1_sources.json`), one file at a time, with expected-size
validation. Temporary downloads are deleted after extraction. Original Drive
files are never deleted. **No downloads or installations run on the Mac.**

In full-trial mode, small results: `Thesis/Results/word_sentiment_eeg/v3_word_erp/` contains
`REPORT.md`, `manifest.json`, `provenance.json`, `class_counts.csv`,
`class_erp_roi.csv`, task grand-average NPZ files and class plots. Trial epochs
and onset tables: `Thesis/CachedArtifacts/word_sentiment_eeg/v3_word_erp/`,
under a scientific-processing/annotation fingerprint and task/reader folders.
New trial epochs are split into chunks of at most **64 trials** (~13.5 MB
uncompressed), instead of one very large file per class. `epochs_index.json`
lists each class's files and global start/stop rows. Valid older `*_epochs.npz`
files are reused directly. Arrays have `eeg_uV[trial,channel,time]`, `event_id`,
`times_ms`, `channel_names` and `sfreq_hz`; the onset table's
`epoch_index_in_class_file` is the global class row across the listed chunks.
The full first run may take hours and the persistent trial epochs may occupy
tens of GB. Completed readers are reused on reruns. Files without usable EEG
and download failures are explicit in the report; failed downloads can be
retried by running the same notebook again.

On restart, the runner audits **all** existing reader checkpoints before new
extraction and saves `resume_plan.json`. It reports two separate counts:
analyses/onset tables already computed, and readers with complete trial exports.
The saved-analysis count never resets merely because a new session starts.
If a reader's means/onsets are valid but some trial files are absent, its
analysis is kept. Only missing trial chunks are reconstructed from the saved
onset samples, with source-hash validation; labels and fixation onsets are not
recomputed and valid positive/negative/neutral files are not overwritten.
New readers run first; export-only repairs follow. Completed repair chunks are
reused after a disconnect. Figures/grand averages are built once at the end,
rather than repeatedly while walking saved readers.

The original `a49719ea0d6626bc…` cache remains compatible when its annotation,
scientific-code, source-inventory and montage hashes match. Changes to resume,
logging or storage orchestration do not require repeating scientific extraction.
Streaming means use float64 accumulation, stored as float32; this can differ
from old float32 batch means only at numerical rounding precision. Existing
saved means are kept byte-for-byte. The runner refuses to silently overwrite
a checkpoint with incompatible settings or inconsistent means/onset counts.

These are descriptive **fixation-related potentials (FRPs)**. Adjacent
fixations overlap, and this version does not deconvolve that overlap or prove
a sentiment-specific response. ±1 SEM shading is across readers, not trials;
no significance tests are claimed. Real ERP results await the user's Colab run.

The requested annotation pass is complete: **1,037 distinct English sentences,
20,458 source word occurrences**, across **all three ZuCo 1.0 tasks**. Labels
were assigned by Codex reading the sentences and recording token decisions.
They are not generated by a sentiment dictionary when Colab runs.

- [Prepared word labels](annotations/word_labels.csv)
- [Uncertain word cases](annotations/uncertain_words.csv)
- [Lexical/contextual differences](annotations/lexical_context_differences.csv)
- [Annotation policy](annotations/POLICY.md)
- [Coverage and counts](annotations/annotation_manifest.json)
- [Open the short Colab runner](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/short.ipynb)

## What is already done

Every source token has separate `lexical_class` and `contextual_class` values:
positive, negative, neutral or uncertain. The table also preserves the whole
sentence, source sentence ID, word position, task membership, context flags,
notes and single-agent provenance. **These are not human consensus/gold labels.**

The agent decisions are stored in `annotations/agent_decisions.tsv`. Every
sentence has an explicit review record. All non-neutral decisions and lexical
overrides are recorded there; unlisted tokens within that reviewed sentence are
explicit neutral decisions. Missing sentence reviews cause compilation to fail.
`compile_annotations.py` expands the decisions into the full CSV with exact
surface-word validation. No dictionary or sentiment classifier supplies labels.

| Task | Distinct sentences reviewed | Notes |
|---|---:|---|
| SR | 400 | Original word boundaries checked against the saved SR word table |
| NR | 300 | Factual reading material included |
| TSR | 386 | Corresponds to 407 presentations, including repeated sentences |

NR and TSR share some sentences, giving 1,037 distinct sentences across the
whole corpus. The label can be reused across readers/tasks without dropping
those identities. This version covers **ZuCo 1.0 SR/NR/TSR**; it does not claim
ZuCo 2.0 or TeCo annotation.

One sentence (original ID `113637237`, ordinal 648) has a source encoding/
word-count discrepancy. It is text-reviewed but flagged. Its cache rows must
match exactly or are explicitly excluded from EEG alignment. Other unexpected
missing/mismatched annotations are errors; they never become neutral by default.

## Context handling

A word does not inherit the sentence's overall review label. “Film” remains
neutral in “a terrible film.” Negation is interpreted in scope: “never fails”
can affirm success, “not only good” does not reverse good, and “not good” denies
positive evaluation. Qualified degree and double negation can remain uncertain.

Proper names and titles are distinguished from ordinary word senses: Love in
“Gladys Love Smith” is neutral in context, while its ordinary lexical label is
positive. Military intelligence is not an assertion of cognitive excellence.
Sarcasm, idioms and missing discourse are marked with reasons; the agent does
not pretend to know the reader's feelings or an unstated speaker intention.

For the initial EEG screen, use **agreement between lexical and contextual
labels**, retaining uncertainty and disagreements in separate outputs. This
conservative subset avoids mixing negated positives into a straightforward
positive-word class. It is not a claim that the excluded words have no useful
EEG. All original annotations remain available for later separate analyses.

## Optional legacy feature/classification analysis

The current short notebook runs `run_erp.py`, as described above. The older
`main.py` runner is available separately for band-power feature exports and
controlled classification probes using existing SR/NR/TSR word caches. It is
not launched by Run all in the current notebook.

The legacy default run tag is `v2_agent_annotated`. Identical completed runs are reused;
changed sources/labels/code/config require a new tag. Logs and stages save to
Drive. Prepared labels can be inspected now, without running any EEG analysis.

## Drive organization

```text
MyDrive/Thesis/
  Data/word_sentiment_eeg/                 # prepared labels and policy
  CachedArtifacts/zuco_multimodal_sentiment/word_eeg/
    TRT/                                 # existing SR caches
    TRT_NR/                              # existing NR caches
    TRT_TSR/                             # existing TSR caches
  CachedArtifacts/word_sentiment_eeg/raw_by_class/
  Results/word_sentiment_eeg/v2_agent_annotated/
```

The result folder contains the full annotation/EEG join, alignment report,
coverage by task/reader/class, primary concordant class EEG arrays, separate
context-disagreement/uncertainty rows, contextual-class source indices,
plots, split membership, predictions, controls, provenance and `REPORT.md`.

TRT arrays contain **word-level band-power features** (8 bands × 105 channels),
not raw waveforms. Available original `word.rawEEG` fixation segments are also
exported from existing MATLAB files, preserving variable length and both labels
in metadata. These preprocessed fixation segments are not deconvolved FRPs.
Missing raw fields/tasks are reported; sentence EEG is never substituted.
Raw exports are grouped lexically; metadata carries contextual labels so these
must be filtered when a concordant or contextual group is wanted.

Class feature exports may take hundreds of MB; available raw segments may add
roughly 0.5–3 GB. These are estimates. Original EEG stays in place and the notebook
prints actual saved disk use. The 6–7 MB annotation CSV is the main text artifact.

## Evaluation and limits

Each task is tested separately, for binary and ternary classes. Five folds hold
out whole sentences across readers. A stricter protocol also purges test word
spellings from training. Optional reader+sentence+word-form holdout is available.
Train-only scaling/PCA and fixed ridge probes compare EEG, nuisance-only,
EEG+nuisance and majority predictions. Nuisance variables include length,
training-corpus frequency, position and available eye measures.

Insufficient classes/folds are reported. Predictions across readers and words
are dependent. A word-form permutation screen is used only if every spelling
has one consistent target label; otherwise its inapplicability is explicit,
while the other probes still run. P-values are exploratory and uncorrected.
No positive result alone establishes a sentiment-specific neural mechanism.

Cropping natural-reading EEG does not make it an isolated-word experiment.
[Pfeiffer et al. (2020)](https://doi.org/10.1016/j.neuroimage.2020.116934) is a close
word-sentiment FRP precedent, with a different deconvolution/trial-averaging
method. This first screen does not reproduce its decoding result.

Legacy NRC comparison code remains explicitly available with `--lexicon FILE`,
but is not the default and does not replace the prepared agent annotations.

## Verification

Tests check full 1,037-sentence/20,458-token coverage, exact token joins, contextual
examples, missing-label rejection, known alignment exclusions, split leakage,
known synthetic signal/null behavior and class exports. Cloud integration checks
both MATLAB formats, plotting and resumability, for prepared labels and the
explicit legacy comparison. **Real EEG results remain pending a Colab run.**
