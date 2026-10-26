# Independent full-trial preparation before classification

Run `notebooks/prepare_trials.ipynb` in Colab, CPU. It handles ZuCo SR/NR/TSR
(36 reader/task recordings) and the validated native TeCo task1 files (12 readers).
This is a separate data-preparation step. Classification/correlation are next.

## What is preserved

Each usable word occurrence retains a float32 **all-channel, 125-Hz**
first-stored-fixation epoch: ZuCo 105 × 126; TeCo 126 × 126 (8-ms time intervals). There is no averaging over trials/channels or time-bin feature selection.
Processing: average reference, fourth-order zero-phase 40-Hz low-pass on the
whole sentence, event-aligned anti-alias polyphase resampling to 125 Hz, −200…800 ms inclusive, baseline −200…0 excluding time zero.
Baseline per channel is stored separately, so subtraction can be reversed.
40 Hz follows the user's requested setting for future reconstruction; the
completed 30-Hz analyses remain a separate historical condition.

All stored source words have event metadata. Signal preservation does not
require sentiment agreement, content-word status, noncontrol status or a
confident label. Above-100-µV epochs and below-100-ms fixations remain with
flags. Impossible boundaries/missing or invalid first fixations cannot produce
complete epochs and are recorded explicitly. Later valid fixations never
replace invalid first slots. No new blink or negation exclusion is invented.
Units remain native assumed µV for TeCo. No overlap correction is claimed.

Metadata preserve dataset, task, reader, source sentence position/key, native
word index, normalized Unicode word identity, token, full sentence when
available, all available label layers, grammatical class, original rating,
control status, historical acceptance at 30 Hz, recovered onset and method,
first-fixation duration, next recovered **word-first-fixation** interval, flags,
and a unique event ID. The interval is not the next fixation among all eye events.
Missing private revised labels are `unknown`; exact source identities permit
joining them later without EEG regeneration. Label snapshots stay on private
Drive. No Persian text or signal is committed publicly.

## Storage and resuming

Output: `Thesis/Results/word_sentiment_eeg/trial_archive_125hz/`.

- `protocol.json`, `preflight.json`, `manifest.json`, `reader_coverage.csv`.
- `{zuco,teco}_channel_regions.csv` and private annotation snapshots.
- `<dataset>/readers/<task>/<reader>/checkpoint.json`, `events.csv.gz`.
- `blocks/block-N/events.csv.gz` and `epochs-N.npz` (at most 32 trials/shard).

NPZ keys: `eeg_uV`, `baseline_uV`, `event_id`, `times_ms`, `channel_names`.
The event's `epoch_file` + `epoch_row` join to exactly one array row. Every
committed file has SHA256; committed blocks are validated, not appended again.
An interrupted uncommitted block may be rewritten. No reader becomes complete
until source word accounting reconciles to the historical inventory.

The notebook checks actual Google Drive quota and verifies that the API account
owns a probe created through the mounted output folder. It budgets uncompressed
125-Hz candidate epochs plus metadata and 0.5-GB headroom; compression is not assumed. Insufficient or
unverified capacity stops before source extraction. The archive has a 9-decimal-GB cap, with no silent event downsampling if a bound exceeds the cap. Expected size is about 8–9 GB before compression. It can take hours. Do not run two instances concurrently. Sources absent from
Drive may be downloaded temporarily in Colab, one at a time, then removed.
Completed archives can resume/validate without redownloading missing sources.

## Limits and the fixed next analysis plan

This full epoch archive supports changing channel subsets, region/time windows
within the saved epoch, time-domain features and spectral features within the
preserved band without reading source EEG again. Different epoch windows,
later-fixation trials, frequencies removed by 40-Hz/upstream filters, or
continuous-event deconvolution may still need the original source recordings.
Original sources must be preserved. It would be misleading to promise that no
future scientific question can ever need them.

After preparation, use one fixed classification/similarity notebook:

- Main binary positive vs negative; secondary three-class with neutral.
- Regularized logistic regression and shrinkage LDA; dummy baseline. No
  hyperparameter sweep or neural network. Standardization fit on training only.
- Sentence-group, word-group and reader-group evaluations reported separately.
  Each tests a different generalization claim; reader separation alone does
  not prevent shared text/lexical identity across folds.
- Balanced accuracy, macro F1, counts and confusion matrices; no neutral
  downsampling by default. Keep dataset/task analyses separate.
- Within-reader same-class vs different-class feature similarity, and repeated
  word vs different-word pairs from distinct occurrences. Match class where
  possible, exclude self-pairs and flag overlapping same-sentence epochs.
  These correlations remain descriptive, not proof of sentiment coding.
- Same-word/same-reader diagnostic uses different event IDs and reports
  separately from unseen-word generalization. Never train/test the same event.

Those are planned analyses, not results or implemented models in this notebook.

## 10-GB storage policy

Source sampling is still 500/1200Hz; retained signals are 125Hz with all channels.
Resample with scipy.signal.resample_poly (zero-phase anti-alias FIR), after
whole-sentence 40Hz filtering. The resampling segment starts 400ms before each
fixation; trim its first25 output samples to preserve the requested −200…800ms
epoch and exactly place time0 at sample25. Extra200ms context surrounds both
epoch edges. Only unavailable context outside the sentence uses reflection,
recorded per event; whole-sentence filtering remains unchanged. Baseline is
computed from the25 stored pre-onset samples. Native-rate 40Hz epoch amplitude
is flagged before downsampling, and stored-signal peak is recorded separately.
No float16 quantization, channel removal, trial averaging or class downsampling.

The 8ms grid supports the planned 40Hz FRP/classification comparisons but does
not preserve native sample precision. The resampling adds an anti-alias filter,
so it must be reported, not described as lossless preservation of native EEG.
NPZ compression itself is lossless for the stored float32 arrays.
