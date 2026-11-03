# Pfeiffer 2020: reproduction before TeCo adaptation

Status 2026-11-03: user selected Colab/Python only, no MATLAB. Source audit,
Python synchronization, sparse nuisance-GLM library, synthetic tests and a
one-block engineering runner are implemented. The strict real-data timing
validation is not passed. The full reproduction is **not implemented or run**. The existing classification
and neutral/nonneutral follow-up are different experiments and remain unchanged.

## Python pilot validation status

`src/word_sentiment_eeg/paper_python.py` independently implements shared-trigger
clock fitting, a time-expanded sparse fixation/saccade/blink/keypress design and
per-channel least-squares nuisance subtraction. Synthetic clock recovery,
trigger-mismatch rejection and recovery of known overlapping responses passed
in GitHub Actions. This is not numerical equivalence to Unfold.

On real ZAB SR block 1, the released preprocessed EEG and corrected eye events
have 98 exactly sequence-matched sentence-boundary anchors. A global linear fit
has maximum residual **3.9733 ms**, above the paper's stated 2-ms/one-native-sample
level. Removing keypress triggers from synchronization anchors did not resolve
this. See `paper_python_timing_audit.json` for the full public trigger diagnostics.
No triggers were discarded to manufacture a passing fit. This needs investigation
of source/version and synchronization differences before a faithful reproduction.

The default pilot stops at 2 ms. A separately named explicit engineering option
permits at most 8 ms (one 125-Hz output sample) for testing the remaining code;
its report retains `native_one_sample_target_met=false`. Such a diagnostic run
must never be described as passing the original paper's timing criterion or as
a sentiment result. No word-bound mapping, original SST join, trial matching,
FRP inferential statistics or paper decoding has yet been run.

The nuisance-model pilot uses the raw eye-event table, preserving the short
fixations absent from the corrected word-selection export. Word-bound selection
will need separately validated corrected gaze coordinates.

The experimental Colab pilot is `notebooks/zuco_paper_pilot.ipynb`. It is marked
as not ready for faithful reproduction; do not ask the user to run it as a
working full reproduction while the strict timing check fails. It downloads
approximately 51 MB of public source files temporarily in the cloud and saves
only coefficients, event tables, QA means and diagnostics (expected <5 MB).

Source: https://doi.org/10.1016/j.neuroimage.2020.116934

## What the published decoding result actually tests

The reported single-trial classifiers were at chance. The approximately 0.60
logistic-regression result used same-sentiment, within-reader 40-trial averages.
The study compares 1, 10, 20 and 40 trials per average. Its decoding features
are 105 scalp electrodes by 63 samples in the 0–500 ms interval, followed by
training-fitted normalization and supervised neighborhood component analysis.
It uses 100 stratified 95/5 train/test splits, not our sentence/word/reader
holdouts. This must be stated when comparing scores. Original samples must
never occur in both train and test averages; save average membership explicitly.

One decoding detail needs the original scripts or an explicit reconstruction
decision: 5% of the reported 328 forty-trial averages is about 17 test samples,
fewer than the 24 possible reader × sentiment strata. Standard joint-stratified
scikit-learn splitting cannot represent all those strata in that test set.
The prose may describe marginal/approximate matching instead. Do not silently
change the test fraction or claim an exact joint-stratified implementation.

## Work packages and acceptance checks

| Stage | Required implementation / evidence | Current status |
| --- | --- | --- |
| Study code | Exact custom scripts, release, dependencies and license | Not located after paper, author GitHub and OSF script inventory checks. No claim of absence. |
| ZuCo inputs | SR only, 12 original readers, raw block EEG, eye-tracker events, triggers, word bounds, SST word/phrase/sentence labels | OSF block EEG/ET/bounds located. Schema and synchronization need validation. |
| Preprocessing | Automagic 1.9, 105 scalp + 9 EOG selection, clean_rawdata, PREP/CleanLine, 1–50 Hz EEGLAB FIR, MARA ICA, spline interpolation, quality measures | Not implemented in this reproduction. Corpus preprocessing is not assumed identical. |
| Synchronization | EYE-EEG trigger-based synchronization, sample-level timing check, 125 Hz resampling | Native block and event schema audit comes first. |
| Oculomotor cleaning | Unfold fixation intercept/duration, saccade intercept/10-spline amplitude, blink intercept, keypress intercept; −600…1000 ms; subtract modeled contribution | Full event timeline required. No invented zero blink/keypress series; no joining discontinuous sentence signals as continuous. |
| Selection | Word-associated fixations, not first-only; remove blink-preceded/out-of-bounds/short (<100 ms) fixations, neutral, preceding three-word English negation window | Requires native events and original labels. Maintain sequential exclusion counts and independent flags. Exact blink rule needs verification. |
| Trial matching | Match conditions per reader on linguistic/oculomotor properties; paper describes word lengths 1–20 and neighboring fixation onset distributions | Exact timing-stratification rules/seed need custom code or documented reconstruction; never call approximate bins exact. |
| Epochs | −600…1000 ms; reject raw FRP excursions beyond ±90 µV and exclude same IDs from cleaned FRPs | Current −200…800 ms, 40 Hz, first-fixation archive cannot reproduce this stage. Baseline handling must follow source evidence. |
| FRP inference | GFP/time GLMM, electrode/time GLMM, spatial cluster permutations, topographic consistency; raw versus cleaned comparison | Needs paper formulas, adjacency, permutations, random effects and full electrode coordinates; not replaced by uncorrected t-tests. |
| Decoding | Same-reader/sentiment/similar-length averages 1/10/20/40, 0–500 ms all-channel features, scaler → NCA → SVM/logistic, 100 stratified 95/5 splits | Not implemented here. Training-only tuning (paper says 10-fold inner CV); exact optimization grid and averaging remainder rules require verification. |
| Controls | Fixation duration, word length/frequency, neighboring fixation onset probabilities | Preserve all events; do not substitute only the next stored word fixation. |
| Reporting | Original counts/figures and sampling uncertainty; list all departures and ambiguous settings | Paper has a positive/negative count discrepancy between prose and flowchart: preserve it, do not force counts. |
| TeCo | Validated native continuous EEG + full events; Persian labels/negation rules and correct montage/rate | Native reading pickles located, but full saccade/blink/keypress timing and original preprocessing provenance not validated. This will be an adaptation, not an English replication. |

## Reusable code and execution constraints

- Unfold: https://github.com/unfoldtoolbox/unfold (MATLAB, GPL-3.0).
- Automagic: https://github.com/methlabUZH/automagic (MATLAB, GPL-3.0).
- FieldTrip: https://github.com/fieldtrip/fieldtrip (MATLAB, GPL-3.0).
- Dataset scripts: OSF q3zws/scripts/Matlab_scripts includes firstLevel_SR.m,
  prepareETData.m and SyncET2EEG.m. This establishes preprocessing code
  availability, not availability of the complete 2020 paper. Verify permission
  and license before vendoring source.
- These MATLAB tools cannot simply be imported into Python Colab. A MATLAB cloud
  runtime/license or a separately validated Python reimplementation is needed.
  A Python implementation must be named as such, not the authors' original code.

## Sources newly verified

Author OSF root https://osf.io/q3zws/ includes task1- SR / Raw data and
Preprocessed, each with reader folders. ZAB raw block 1 has
ZAB_SR1_EEG.mat (53,488,721 bytes) and ZAB_SR1_ET.mat (813,937 bytes).
Preprocessed copies include gip_ZAB_SR1_EEG.mat (50,454,745 bytes) and
ZAB_SR1_corrected_ET.mat (810,904 bytes). Word-bound files are also listed.
Original labels/word-event associations still need a validated source join.

NEMAR nm000278 is an alternative new BIDS distribution, with published
fixation/saccade/blink events and original subject IDs. A sampled SR event table
contains those types and trigger codes, but no word labels or identified keypress
column. Do not assume its conversion or preprocessing reproduces the 2020 input.
https://www.nemar.org/dataset/nm000278?v=v1.0.0

## Storage and next action

Run scripts/paper_source_audit.py in GitHub Actions. It reads public metadata and
four temporary public sample MAT files (about 106 MB combined), inspects schema,
and deletes sources. Only a small JSON report is retained in an Actions artifact.
No Mac EEG/dependency downloads, no Drive writes, no existing analysis changes.

Once event schemas and exact label joins are established, stage one reader in
temporary cloud storage, validate synchronization/selection/cleaning, then run
all SR readers. Store small audit tables, averages, membership IDs and model
outputs; avoid a second full trial archive on Drive. Determine the storage budget
before starting that reconstruction. Test TeCo only after the ZuCo route works.
