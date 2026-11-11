# TeCo paper-style analysis — Task 1, raw FRPs

Run in this order:

1. [CPU preparation](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/teco_paper_prepare_cpu.ipynb): CPU, Run all; wait for `CPU PREPARATION COMPLETE`.
2. [GPU analysis](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/teco_paper_analyze_gpu.ipynb): T4 GPU, Run all, same Drive account.

Code is pinned to the immutable `teco-paper-raw-v1` tag. The existing ZuCo
workflow and completed results are not changed by this addition.

## What is shared with the ZuCo reconstruction

- All recoverable stored word-fixation segments, including rereading fixations.
- 1–50-Hz bandpass, average reference, 125 Hz, −600…1000-ms epochs, no baseline subtraction, 100-ms duration threshold, ±90 assumed µV rejection.
- Gaussian trial-level ML mixed models with reader random intercept/slope.
- GFP computed on individual trials before averaging; electrode/time contrasts.
- 5000 trial-paired electrode-shuffle TCT permutations and 1000 trial-pair sign-flip spatial/temporal cluster permutations.
- The raw-FRP figure uses positive/negative/contrast heatmaps, GFP, pointwise tests, corrected contours, TCT and fixed 224–304-ms reference topographies.
- A selection/individual-trial figure and a controls figure. Both PNG and SVG saved.

## Important differences

Native TeCo pickles provide sentence reading EEG and separate word-associated
fixation segments. They do not provide the validated full blink, saccade and
keypress timeline required for our ZuCo regression cleaning. This adaptation
therefore produces **raw FRPs only**. It does not repeat raw data in a panel
called “cleaned.” Full eye-velocity/neighbor-timing controls are labeled
unavailable. Word rawET fragments do not establish a continuous gaze timeline.

Labels are private, agent-reviewed Persian annotations, not SST annotations or
human consensus. Lexical and word-in-context labels must agree; sentence labels
need not agree. Neutral/uncertain/disagreeing labels are excluded here. This is
not equivalent to explicitly removing every event preceded by negation.

Matching balances exact word length and target-fixation-duration tertiles, within
reader. It cannot reproduce full previous/next-fixation timing matching. Epoch
quality exclusions precede matching here. Sentence-edge filtering/epoch loss
also differs from the original continuous-block route. Frequency distributions
are shown but not automatically matched. These limits must accompany inference.

Control sentences' **reading** is included; `answer_data` is never read. Only
Task 1 is supported until other tasks' native signals are supplied and validated.

## Storage, inputs and outputs

- Native pickles: `Thesis/Data/teco_pickles/<reader>_with_total_task1/task1.pickle`, with known historical alternative layouts checked. One file per canonical reader.
- Private CSV: `Thesis/gpt uploaded files/word_sentiment_eeg/teco/word_labels_three_level.csv`, or existing `word_labels.csv` if the revised file is absent. Exact sentence hash/token/index joins; no positional fallback.
- Outputs: `Thesis/Results/word_sentiment_eeg/teco_paper_raw_staged/`.
- **2 GB total additional cap**; approximately 0.61 GB temporary permutation maps plus 0.15 GB safety reserve, leaving approximately 1.24 GB for CPU archives and metadata. Quota checked before source reads. A budget failure keeps earlier reader checkpoints.
- Per reader: lossless float32 `prepared_trials.npz` (trial ×126 channels ×201 samples), label/event/reader/word/sentence identities, ranked pair indices, private accepted metadata, audit of all stored candidate segments, sufficient statistics, source hash, matching policy, checksum commit marker.
- Archive includes accepted matched positive/negative trials, not neutral or rejected waveforms. Original source pickles are retained without saving another copy.
- Completed readers skip EEG loading after checksum verification. A disconnect during an unfinished reader can repeat that reader. GPU permutation chunks track included reader/input signatures and resume without double additions. Temporary null maps are deleted only after successful final figures; accepted trials remain.

## Validation status

Cloud synthetic tests cover multiple stored fixations per word, control reading,
invalid segments, duration/order mismatch, duplicate recovered onsets, archive
corruption, unchanged-resume behavior, changed-input refusal, single-representation
permutations, 126-channel montage projection and figure generation. They do not
establish that all private native files meet the schema; the first CPU run checks
those actual files and records rejected/missing events. No TeCo result is claimed
before that run completes.
