# Word sentiment from word-aligned EEG

[Open the short Colab runner](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/short.ipynb)

This experiment follows the thesis's sentence-level EEG results with a different question:
**Does the EEG recorded while a reader fixates an emotionally valenced word distinguish
negative, neutral and positive lexical classes?** All three available ZuCo 1.0 tasks
(SR, NR, TSR) are included; sentence sentiment labels are retained as provenance only.

The implementation lives in Python files. The notebook only mounts Drive, obtains
the code, calls `main.py`, and displays the saved report. A CPU Colab runtime is
sufficient. No model checkpoint, LLM, GPU training, or large EEG download is needed.

## Run

1. Open `notebooks/short.ipynb` in Colab. The repository is private, matching the existing thesis repository.
2. Allow its existing `GITHUB_TOKEN` Colab secret to read this repository. A fine-grained token needs **Contents: read** for `parmisbathayan/word-sentiment-eeg`; a classic token needs private repository access. The notebook explains this if access is missing.
3. Choose **Runtime → Run all**, then approve Drive mounting. The defaults match the existing `MyDrive/Thesis` layout. No file picking or manual labeling is required for this exploratory first pass.

The runner installs small scientific dependencies **inside Colab only**, reuses all
three existing word caches, obtains the small NRC VAD v1 lexicon from its author
once, and saves outputs as it goes. It also exports original fixation EEG from
existing `Data/zuco_og_raw` MATLAB files where `word.rawEEG` is present. This raw
export may take longer than the feature analysis; unavailable tasks/fields are
explicitly reported. It never downloads missing EEG files.

No installer or environment is run on the Mac. Local synthetic tests use only
the existing Codex bundled NumPy/Pandas runtime. Do not run the cloud dependency
installation commands on the Mac without the user's explicit approval.

## What the first run saves

- Every word and its source alignment, including skipped words and missing EEG.
- Lexical valence/arousal/dominance, conservative class labels, and context flags.
- An optional context-review sheet, with one row per word occurrence across readers.
- EEG arrays for each class and task, with reader IDs and matching metadata.
- Available original variable-length `word.rawEEG` fixation segments by class.
- Binary and ternary ridge screens, majority and nuisance baselines, and a
  word-form permutation screen, separately for each reading task.
- Full split membership, predictions, coverage failures, source hashes,
  dependency versions, logs, a class coverage plot, and `REPORT.md`.

The controls test the implementation and limit several obvious explanations;
they do not make this first run a confirmatory neuroscience result.

## Word labels and context

The initial target is **lexical valence**, from the human-rated English NRC VAD
v1 lexicon. This is an automatic word-to-lexicon mapping, not human contextual
annotation of ZuCo, and not a measure of the reader's emotional state.

| NRC v1 valence | Class |
|---|---|
| ≤ 0.35 | negative |
| 0.45–0.55 | neutral |
| ≥ 0.65 | positive |
| between these intervals | ambiguous; excluded from class probes |
| no exact normalized match | unknown; excluded, never assumed neutral |

The boundaries are predeclared engineering choices. They are not standard NRC
categories, and they are not tuned after looking at EEG. Exact normalized
spellings preserve internal hyphens/apostrophes; no hidden lemmatization is used.

For “not wonderful,” the word's lexical label remains positive. A nearby-negation
flag marks it for review; `contextual_class` remains `unreviewed`. Contrast and
mixed-polarity sentences receive additional flags. These simple rules do not
resolve negation scope, irony, word sense, or sarcasm. The conservative default
probe excludes flagged sentences/words. All class exports retain the flags,
including excluded examples for inspection.

Optional reviewed contextual labels can be supplied with `--reviews FILE.csv`.
They never overwrite lexical labels. This first implementation does not train a
contextual sentiment classifier; the two annotation questions remain separate.
See [the annotation protocol](docs/ANNOTATION.md).

## EEG representations

The primary analysis uses the existing **TRT word band-power caches**:
8 frequency bands × 105 electrodes for each reader × word occurrence. These
are EEG-derived features, not raw waveforms. No sentence pooling is performed.
Every missing/partly missing recording is kept in the audit, then excluded from
the complete-EEG probe. Flat feature columns (including a reference channel if
flat) are removed using training data only.

The optional raw export reads `sentenceData.word.rawEEG`, preserving the
variable-length fixation segments as stored in ZuCo's preprocessed files. Each
class archive contains `samples [105, total_samples]`, `offsets [segments+1]`,
`row_id`, and `sfreq_hz=500`. Recover segment `i` with
`samples[:, offsets[i]:offsets[i+1]]`. Match it to the adjacent metadata CSV.
These are **not** fixed-window or deconvolved fixation-related potentials.
No raw field is replaced with sentence EEG. Missing raw fields/tasks appear
in `raw_export.json`.

## Evaluation

Each task is evaluated independently, with fixed ridge strength and train-only
preprocessing/PCA (at most 64 components). Both binary negative/positive and
ternary negative/neutral/positive targets are tested. The default 5 folds use:

| Protocol | Test generalization | Caveat |
|---|---|---|
| `sentence` | unseen sentences, every reader grouped with that sentence | Known word forms and readers can recur |
| `sentence_word_form` | unseen sentences and spellings; overlapping test word forms purged from training | Same readers and morphological relatives may recur |
| `subject_sentence_word_form` (optional config) | unseen reader group, sentence and spelling | Smaller crossed test set; not the default first pass |

Minimum class counts **and independent training word-form counts** are checked
inside every fold. Unusable folds are reported rather than silently dropped.
Purging can sharply reduce usable data; a partial run is labeled partial.

Baselines use the exact same partitions: majority class, nuisance-only (word
length, training-text frequency, position, sentence length and available eye
measures), EEG, and EEG+nuisance. Per-occurrence training weights prevent words
read by more readers from automatically contributing more total weight.
Reported prediction units are dependent reader-word observations; scores do not
carry an independent-trial confidence interval.

The 99-label-permutation screen assigns a shuffled class to each **word form**,
consistently across its occurrences/readers, and fits the same ridge operator.
It is an exploratory lexical-label randomization screen. Its p-values are
uncorrected across the task/protocol/target family, and exchangeability does not
remove all correlated semantics or sentence context. Do not claim a sentiment
mechanism from a significant screen. Stronger matching, reader generalization,
multiplicity control and a prespecified confirmatory design are follow-ups.

## Drive layout

Existing data is read in place:

```text
MyDrive/Thesis/
  Data/zuco_og_raw/results*_SR.mat
  CachedArtifacts/zuco_multimodal_sentiment/word_eeg/
    TRT/       # SR — existing 12 subject caches
    TRT_NR/    # NR — existing 12 subject caches
    TRT_TSR/   # TSR — existing 12 subject caches
```

Only two new project-owned branches are created:

```text
MyDrive/Thesis/
  CachedArtifacts/word_sentiment_eeg/
    lexicons/NRC_VAD_v1/                   # shared once; original notices retained
    raw_by_class/<fingerprint>/<task>/<reader>/
  Results/word_sentiment_eeg/v1_lexical_audit/
    manifest.json
    data_inventory.json
    all_words.csv.gz
    word_inventory.csv
    context_review.csv
    class_counts.csv
    coverage_by_task_reader_class.csv
    reader_class_band_means.csv
    eeg_by_class/<task>/<class>.npz
    eeg_by_class/<task>/<class>_metadata.csv
    raw_export.json
    evaluations/
    plots/
    run.log
    REPORT.md
```

Feature class exports are generally hundreds of MB in total; available raw
segments may add roughly 0.5–3 GB, depending on coverage and fixation duration.
These are estimates, not measured run sizes. The notebook prints actual saved
disk usage. Raw input files are never copied or deleted. Temporary code and
installed packages live in the disposable Colab VM, not Drive.

An identical completed run is reused; interrupted runs reuse completed probe
files. A change to source contents, lexicon, review file, code or configuration
requires a new run tag. This prevents overwriting a result with different inputs.

After the run, tell Codex it finished. The compact report and result JSONs can
be read through the connected Drive without uploading EEG to chat or GitHub.
No results are pushed automatically. The lexicon's terms prohibit redistributing
it; raw EEG, tokens, labels and row-level data also stay on Drive.

## Research basis and limitations

[Pfeiffer et al. (2020)](https://doi.org/10.1016/j.neuroimage.2020.116934)
is a close precedent: word-valence FRPs during ZuCo SR reading. Their method
excluded nearby negations, matched confounds, used deconvolution, and reported
decoding from averages of multiple trials. This first band-power screen is a
different method and is not a reproduction of their performance.

ZuCo participants still read words **inside whole sentences**. Cropping a
recording to a word does not make it an isolated-word experiment. Nearby neural
responses, eye movements, word identity and context remain possible causes of
differences. Prior sentence-level negative results mean the tested pipelines
did not recover useful signal; they do not prove EEG has no sentiment content.

ZuCo 2.0 and TeCo are not silently pooled into this first run. This version
targets all three already-available **ZuCo 1.0** tasks. Extensions need separate
source manifests, task/reader IDs and label/normalization audits.

See [research and data evidence](docs/RESEARCH_AND_DATA.md) for verified locations,
sources and the boundary between prepared code and actual experiment results.

## Validation

Offline synthetic tests cover skipped-word alignment, leakage assertions,
lexical/contextual label separation, known signal recovery, null controls,
training-only preprocessing, insufficient classes and class EEG exports.
GitHub Actions additionally tests both MATLAB representations and a complete
synthetic run, including resumability and the official lexicon's live schema.
Synthetic results verify software behavior; scientific scores remain pending
until the notebook runs on the real Drive data.
