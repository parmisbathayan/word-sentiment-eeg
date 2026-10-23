# Native TeCo compact word FRPs

`notebooks/teco.ipynb` runs `run_teco_erp.py` on existing mounted Drive files.
Use CPU. The public repository contains code and the generic native montage;
Persian corpus texts, labels, reader files and results remain private on Drive.

Inputs:

- Native reader pickles discovered under `Thesis/Data/teco_pickles`, the original
  `pickle_files_acc3/acc5` layouts or `TeCo/task1_with_total`. A complete canonical
  roster of 12 readers is required. Alternate copies are logged, not counted.
- Private `Thesis/gpt uploaded files/word_sentiment_eeg/teco/word_labels.csv`.
- The original generic 126-channel montage, ordered exactly as source arrays.
  `assets/teco_montage.txt` is copied from the existing project montage.

Sentence identity must match exact Persian text SHA256 and native `sentenceId`;
word identity must match native dictionary index and explicit `content`. No
English labels, whitespace fallback or compressed feature positions are used.

The first stored fixation is located inside `trial.raw_data` across its complete
waveform and all126 channels. Missing/invalid first fixations are excluded;
later fixations are never promoted. `fixPositions` is not an EEG sample index.
Native rate is documented1200Hz, supported by original project notebooks, not
an independently recovered acquisition header. The mislabeled old600Hz
intermediate is not used.

Whole-sentence average reference and fourth-order30Hz zero-phase lowpass precede
fixed -200..800ms inclusive epochs (1201samples). Baseline is -200..0ms excluding
zero. Absolute amplitude above100 in the original assumed microvolt scale is
excluded. Amplitude calibration and preprocessing independence remain limitations.

Five separately saved groups:

1. Negative words with clear matching lexical/contextual labels.
2. Neutral words with clear matching lexical/contextual labels in any sentence.
3. Positive words with clear matching lexical/contextual labels.
4. A subset of group2 whose independently reviewed sentence context is strictly neutral.
5. A separate subset of group2 in original source-rated neutral sentences
   (`original_sentence_label == 0`). Overall source-neutral sentiment does not
   mean there is no affective content.

Controls, uncertain words, context reversals and alignment warnings are excluded.
Labels are single-agent annotations, not human consensus gold. The stricter
neutral subset does not make the sentence contexts of valenced groups match.

The three grand-average comparisons have separately recorded reader rosters:
>=5epochs in each of the relevant three groups, equal reader weight. No missing
group is replaced by zero in a grand average. Scalp regions are fixed native
geometric bins, disjoint and exhaustive: Y>=.5anterior; Y<=-.8occipital;
remainingY<=-.4posterior; remainingX<=-.65leftlateral; remainingX>=.65rightlateral;
remainingcentral. They describe sensors, not localized cortical sources.

Outputs are in one folder:
`Thesis/Results/word_sentiment_eeg/teco_v1_word_erp_compact/`.
Each reader has `class_means.npz`(5x126x1201), `word_onsets.csv.gz` and a
checksum/source manifest. Group NPZs retain all channels and reader means;
regional/group/reader figures, counts and window values are produced once at
the end. Original EEG and individual trial arrays are not copied into results.
Between-reader SEM is omitted and recorded as missing when only one reader qualifies.
These descriptive FRPs retain overlapping eye-movement/fixation responses;
plots and conventional300–500/500–800ms windows are not significance tests.

Resume checks label/code/montage fingerprint, source path/size/mtime, checksums,
arrays, unique event IDs and event counts. SourceSHA256 is computed on initial
extraction; source bytes are not rehashed on a metadata-identical resume. Changed
code/labels/montage require a new run tag. Corrupt readers rebuild individually;
one reader is never added twice. A folder is only marked complete after the full
roster and final summaries finish.

No TeCo extraction result is claimed by preparing this code. Cloud synthetic
tests exercise real waveform matching, missing-first-fixation behavior, strict
baseline, corrupt-checkpoint rejection, source-change rejection and duplicate
reader rejection without private corpus inputs.
