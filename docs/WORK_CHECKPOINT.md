# Work checkpoint — 2026-11-03

Read this before resuming after context/token interruption. This is a checkpoint,
not a claim that the entire paper reproduction is complete.

## User requirements

- Colab/Python only; no MATLAB access. GitHub holds code/notebooks. Drive holds
  private inputs/results. No Mac dependencies, environments, models or datasets
  may be downloaded/installed without immediate explicit approval.
- Additional Drive storage is limited to approximately 10 GB for this project.
  Avoid another EEG archive. State extra output sizes before user runs.
- Preserve original trial archive, existing feature caches, historical plots and
  classification results. No re-extraction or completed-fit reruns unnecessarily.
- Keep explanations short, clear and concrete. Do not call an engineering pilot
  an exact paper reproduction. ZuCo first, then TeCo adaptation.

## Completed data and classification

- `trial_archive_125hz`: manifest previously verified complete, 48 recordings,
  153,074 individual trials, all native channels, 40-Hz low-pass, 125 Hz output,
  −200…800 ms, first stored fixation only. Counts before analysis exclusions:
  SR 48,856; NR 44,656; TSR 38,320; TeCo task1 21,242.
- `classification_similarity`: Drive manifest verified completed for all four
  tasks on 2026-11-03. Original analyzer is
  `scripts/analyze_trial_archive.py`. Do not edit its scientific code casually:
  its full-file fingerprint is part of the completed analysis protocol.
- All three holdouts ran separately: sentence, word and reader. No simultaneous
  reader+word+sentence holdout was claimed.
- Main positive/negative logistic balanced accuracy, sentence/word/reader:
  SR 51.54/51.01/51.81%; NR 53.56/50.95/52.19%;
  TSR 50.11/51.78/49.36%; TeCo 49.50/49.53/50.59%.
  Three-class results were near 33.3% chance. No significance claims.
- Main policy is lexical=contextual-word labels, not all-three agreement.
  TeCo auxiliary content-word policies were skipped due to missing grammar
  metadata, not because all its experiments failed.

## Urgent neutral/nonneutral follow-up fix

User ran `notebooks/neutral_nonneutral.ipynb` and stopped after completing SR
cases: the old hard guard was 25 MB; saved output had reached 30.5 MB. This is
not evidence of Drive being full or corrupt EEG. Sorted task order is TeCo,
ZuCo NR, SR, TSR, so TSR likely remains after this interruption.

- `scripts/classify_neutral_nonneutral.py` now checks 60 MB, rather than 25 MB.
  Expected total is approximately 40–50 MB; this is an estimate, not a Drive
  measurement. No allocation or second feature cache is created.
- The code recognizes exactly the original script SHA256
  `1f0ba76d81f74b38d24b3bb9dfbbd2f498cb55c73d7231ea59aa25af16312ebe`
  as a storage-only migration. Helper hash, all source/cache hashes and target
  policy must be unchanged. It writes a migration note and preserves case files.
- Completed cases are checksum-validated and logged `KEEP`. Only unfinished
  cases fit. No source EEG reads or feature preparation are allowed.
- Local bundled-runtime checks passed for old-protocol migration and rejection
  of changed cache identities. Cloud test also checks changed helper rejection.
- User action once pushed: reopen the updated neutral/nonneutral notebook and
  Run all (setup pulls latest main). Do not delete saved results.

## Paper reproduction: important evidence

Pfeiffer et al. 2020, https://doi.org/10.1016/j.neuroimage.2020.116934.
The paper's own single-trial accuracy is approximately 50%. Its approximately
60% logistic result used 40-trial same-sentiment averages. Main comparison uses
SR only, original SST labels, 105 channels × 63 samples (0–500 ms), NCA, SVM/LR,
100 stratified 95/5 splits. These differ from our single-trial region features.

No complete custom study-code repository has been located. Public MATLAB tools
exist (Automagic, Unfold, FieldTrip), plus OSF dataset preprocessing scripts.
Do not present those as the complete study code. User chose a Python methods
reimplementation. MATLAB numerical equivalence has not been established.

See `docs/PAPER_REPRODUCTION_PLAN.md` for the full methods checklist, unresolved
matching/tuning/stratification details and explicit departures.

## Reproduction code implemented and validation status

- `scripts/paper_source_audit.py`, workflow `paper-source-audit.yml`:
  verified raw/preprocessed original ZuCo block inputs in cloud; saved only
  metadata in `docs/paper_source_audit.json`. Source files deleted afterwards.
- `src/word_sentiment_eeg/paper_python.py`: strict trigger synchronization,
  continuous sparse FIR nuisance design, fixation duration, saccade cubic
  B-splines, blink/keypress intercepts, per-channel LSMR solver, QA epochs.
- `scripts/zuco_paper_pilot.py`: one ZAB SR1 engineering block. Reads released
  preprocessed 105-channel EEG and RAW eye-event timeline (so short events are
  retained before selection), applies explicit Python FIR 1/50-Hz cutoffs and
  resamples 500→125 Hz. No word labels or sentiment analysis yet.
- `tests/cloud_paper_python.py`: synthetic clock recovery, rejection of missing/
  noisy triggers and recovery of known overlapping responses passed in cloud.
- Real source clock fit has 98 sentence-boundary anchors, max residual about
  3.973 ms, RMS about 1.169 ms. It fails the paper's stated 2-ms native target.
  Full public anchor diagnostics: `docs/paper_python_timing_audit.json`.
- Default synchronization/pilot stops at 2 ms. Explicit diagnostic-only option
  `--engineering-timing-tolerance-ms 8` permits testing code at one 125-Hz output
  sample. Reports retain `native_one_sample_target_met=false`. Never call this
  passing native alignment or a successful paper reproduction.
- Three-channel engineering check succeeded, workflow run 41652671339.
- Full 105-channel RAW-event engineering check succeeded, run 41652945887,
  code commit 26c2e0327937096e617a3fbf2e913e8b7278d379. This still does not
  resolve timing or establish physiological specificity of nuisance subtraction.
- `notebooks/zuco_paper_pilot.ipynb` is explicitly marked experimental/not ready
  for faithful reproduction; do not hand it to the user as a finished paper run.
- Cloud sources are temporary (~51 MB per pilot); Drive untouched by our tests.
  Pilot outputs only betas, small event table, QA means/plot, diagnostics (<5 MB
  expected). No full continuous or epoch cache saved.

## Further implementation and original-source audits

All work below is a Python reconstruction, not verified numerical parity with
the authors' MATLAB tools. No full paper sentiment results exist yet.

- `paper_labels.py`: original SST trees, word/phrase/sentence labels with strict
  alignment. Optional original-dictionary lookup is explicit and never invents
  phrase/sentence labels. Original ZIP SHA256 is pinned; cloud download fallback
  accepts only the exact Stanford-distributed bytes. No agent-label fallback.
- `paper_linguistic_audit.py`: native SR entries 5:405 contain 400 sentences and
  **7,129 source word occurrences**. First-half 250 bounds and second-half 150
  bounds each match all corresponding sentence word counts. Latest successful
  audit is Actions run 41669196748; metadata in `paper_linguistic_audit.json`.
  Strict SST matches 382/400 sentences. Optional dictionary fallback labels
  6,959 word occurrences (5,832 neutral, 711 positive, 416 negative), leaving
  170 unknown. These are source counts, not extracted reader EEG trials.
- Raw ZAB SR1 has 49 sentence start/end pairs versus nominal 50 source bounds.
  No sentence ID text was found in its eye messages. The missing/omitted trial
  identity is NOT established. Never assign sentence labels by guessed order.
  The corpus scripts' eight 50-trial blocks also differ from the paper's block
  description; source/version reconciliation remains open.
- `paper_fixations.py`: bounds mapping, explicit blink-predecessor operational
  rule, short-fixation and original-label selection, three-word negation window,
  reconciling sequential outcomes. Requires validated sentence/event mapping.
  Exact paper blink/bounds conventions have not been established.
- `paper_python.py` now fits a shared duration/spline basis and coefficients
  across a reader's blocks while keeping block edges separate. A synthetic
  two-block known-response check passed; a full real reader has not run.
- `paper_decoding.py`: checkpointed scaler → supervised NCA → RBF SVM/LR,
  train-only fitting and nested tuning, exact event-membership checks, metrics,
  per-case checksums, convergence flags and fit-budget preflight. Cloud checks
  passed in runs 41668102213 and 41668313953. No actual paper scores yet.
  The paper grid is now verified: NCA 10/20/40/80/160/320/640/1280; ten C and
  gamma values from 1e-7 to 1e2; ten inner folds. Log spacing is an explicit
  interpretation. A naive full run is 880,200 classifier fits per averaging
  size (3,520,800 for all four); it is refused by default. Fixed mode uses the
  reported median 320 NCA components, NOT a claimed recovered optimal setting.
  Joint reader×class stratification must be chosen explicitly and fails if
  infeasible; no silent fallback. Event disjointness does not prove temporal
  EEG-interval disjointness.
- `paper_epochs.py`: one-event-at-a-time paired raw/clean epochs after matching,
  shared raw ±90 µV rejection, block boundaries, nonfinite-data checks, explicit
  units and baseline policy. Small class means/audit avoid another trial tensor.
  Extracts the 105×63=6,615 all-channel 0…496 ms samples at 125 Hz. Three new
  synthetic tests passed locally with the bundled runtime; all 13 lightweight
  paper tests passed. This component has not processed source sentiment events.

## Resume next, in order

Additional implementation after the first checkpoint:
`src/word_sentiment_eeg/paper_trials.py` provides per-reader/explicit-stratum
binary matching with reconciling exclusions and within-reader/class averages
for requested sizes 1/10/20/40 with exact event-membership tables. Multi-trial
averages never include two events from the same sentence text. Remainders keep
their actual counts. Three bundled-runtime synthetic tests passed. No guessed
word-length/timing bins are built in, and these components are not yet wired
to original SST/word-bound inputs or used to produce paper decoding results.
Event membership disjointness does not automatically guarantee disjoint raw
EEG intervals; preserve that distinction when implementing evaluation.

1. Neutral/nonneutral storage fix is pushed and full cloud tests passed. Link:
   https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/550091a/notebooks/neutral_nonneutral.ipynb
   Drive manifest and runtime status now verified completed: 44 cases, all four
   tasks, child exit code 0. No rerun is required.
2. Preserve this checkpoint and the current code commits.
3. Investigate the native timing discrepancy against EYE-EEG/source versions.
   No discarding anchors or silently loosening a replication criterion to pass.
4. Validate the unresolved recorded-trial-to-sentence join (bounds-to-source
   word counts now pass). Keep all word-associated
   fixations for the paper route, unlike the historical first-fixation pipeline.
5. Implement paper exclusions/matching with sequential counts, paired raw/clean
   epoch IDs, FRP inference, 1/10/20/40 averages with nonoverlapping membership,
   training-only NCA/tuning and documented evaluation. Unknown original settings
   stay explicit; the 95/5 joint reader×class stratification has a feasibility
   ambiguity for 328 averages (17 test samples versus 24 strata).
6. Shared reader/block nuisance fitting is implemented and synthetically tested;
   validate it on a full real reader after source identity/timing gates pass.
7. Budget persistent outputs before scaling up. Then adapt to TeCo only after
   verifying full saccade/blink/keypress timing, sampling and Persian labels.
   TeCo reading pickles alone have not established a full continuous timeline.

## Do not confuse outputs

Historical ERP plots: 30 Hz. Individual archive/classification: 40 Hz at125 Hz.
New reproduction engineering pilot: explicit independent 1–50 Hz Python FIR on
released corpus-preprocessed continuous EEG, not a fresh Automagic/MARA run.
No new paper FRP statistics, paper decoding scores or TeCo reproduction results
have been produced. User's presentations and original results remain intact.

## CNN follow-up and further alignment investigation

User additionally authorized a CNN. `scripts/classify_cnn.py` and
`notebooks/cnn.ipynb` now implement one compact independent EEGNet-inspired
8/2/16 model, no tuning search. Uses existing archive all-channel 0…792-ms
signals (100 samples), original word-agreement/quality/duplicate exclusions.
Both positive/negative and neutral/nonneutral targets, three separate grouped
sentence/word/reader holdouts, three outer folds each: 24 cases / 72 fold fits.
Inner grouped validation chooses stopping epoch (maximum 25, patience 4).
Word holdouts purge temporally overlapping training events, including inner
validation. Scaling/class weights fit training only; neutral is not downsampled.
One seed per fold: exploratory, no significance claim or tuning search.

Model training, grouped validation, checkpoint reuse and actual shard-to-mmap
identity/time/channel joins passed cloud synthetic checks in run 41713134102.
No actual CNN results or user-data training have run. Select a Colab GPU.
Additional Drive output estimate 20–60 MB, hard guard 150 MB. Temporary Colab
staging up to roughly 3 GB per task, removed after task completion. No second
Drive EEG archive or old feature preparation. A lost Colab runtime requires
re-reading unfinished tasks' archive shards; completed tasks and folds reuse
validated results. Epoch/optimizer checkpoints resume within an unfinished fold.

New reproduction audit: `paper_alignment_diagnostics.py`, Actions run
41713018027, numerical report `paper_alignment_diagnostics.json`.
Raw/preprocessed EEG trigger codes and latencies are identical. Raw/corrected
ET produce the same timing anchors. An independent calculation using EYE-EEG's
regression direction and rounded endpoint grid still has three sentence-anchor
errors of two native samples; the strict paper target is not met. This does NOT
establish the old first-fixation analyses are invalid; this is a separate native
source route for reproduction. No troublesome anchors were discarded.
The original corpus script uses per-file start/end trigger settings and assigns
sentences/bounds by event order after missing-file removal; the sampled 49/50
trial discrepancy remains unresolved. No guessed label-to-recording join has
been accepted. Script context was inspected read-only in cloud, not vendored.

Latest user asks: report neutral/nonneutral results; give CNN notebook; continue
paper reproduction. The source timing/mapping gates still prevent claiming a
completed faithful reproduction. Preserve this distinction in all handoffs.

## CNN NumPy JSON-key failure — 2026-11-07

User's first actual CNN run finished loading TeCo (through yaser 50/50 blocks),
then failed writing `split_spec.json` before the first model training. The shared
splitter returned np.int64 class-count dictionary keys for CNN's numeric labels.
Prior tests exercised splitting and training separately, missing this I/O boundary.

Fix commit `5322c1c`: convert ONLY the CNN split-count keys to native strings and
counts to Python ints. No shared analyzer/common helper, model, selection, splits,
or staging fingerprints changed. The exact old CNN script SHA256 is
`fe7f1154a29905488381a5199a8cf9c9d4d2a774db001158d9d1a2addd3113fb`.
`reconcile_protocol` permits only this old serialization-only revision with all
other data/runtime/helper/spec fields unchanged and records the migration.

Added full `run_case` training-to-score JSON/prediction tests for both binary
targets, all-holdout split JSON tests, and old-protocol migration/changed-input
rejection checks. Cloud matrix now checks Python 3.12 AND 3.13, matching Colab.
Cloud check run: 42353392663 (inspect conclusion before reporting passed).

User recovery: keep the CURRENT Colab runtime; rerun notebook setup, then the
training cell. Setup pulls main; expect migration message and `KEEP RUNTIME STAGE
teco/task1: no archive shard reads`. That reuse still requires the original cloud
temporary files and matching checksums. A runtime reset requires reloading from
the archive. Do not claim training started or results exist from the failed run.

## Content-word neutral/nonneutral sensitivity analysis

User asked to remove function words from the neutral/nonneutral experiment.
New independent script `scripts/classify_content_neutral_nonneutral.py` and
notebook `notebooks/neutral_nonneutral_content.ipynb` reuse the complete feature
caches. Keep only grammatical_class=content in BOTH classes; exclude function,
uncertain and missing grammar. Word-alone/context agreement remains; no sentence
sentiment restriction or neutral downsampling. Existing grammar is an agent
review/conservative screen, not human gold contextual POS annotation.

Grammar is overlaid by exact word occurrence identity ONLY; it never replaces
historical sentiment labels. Duplicate physical/signal groups are removed before
grammar filtering so partial duplicate retention cannot change the original QC.
All three holdouts and logistic/LDA/dummy/pre-onset/nuisance controls inherited
from the original follow-up. Saves per-task selection counts, confusion plots,
predictions and before/after comparisons. No EEG reads or feature preparation.
New output: classification_similarity/neutral_nonneutral_content/. Expected
additional Drive output 10–40 MB, checked against 60 MB. No actual results yet.

Private TeCo grammar-only CSV (3,016 rows, 152,141 bytes) derived from the existing
private reviewed three-level CSV and uploaded to the verified existing Thesis/
gpt uploaded files/word_sentiment_eeg/teco folder. SHA256 is pinned in the code:
2b0cb972609f1878e17b4483023f13191fbaa422536bcffbdc57dcd39adcf347.
Private data is ignored by Git; no Persian stimulus text was published.
English grammar uses the public existing three-level annotations. Both source
hashes are recorded in the new output protocol; changed inputs refuse mixed reuse.

Regression tests cover exact grammar joins, unchanged sentiment columns,
reconciling selection counts, missing labels, neutral words in positive sentences,
nonunique identity rejection, and the full inherited classifier path. Cloud run
42368040964 verifies these (check conclusion before claiming passed).

User should finish the running CNN before launching another notebook/runtime.
Then open neutral_nonneutral_content.ipynb, CPU, Run all. Existing runs/CNN code
and their protocols were not modified by this addition.

## ZuCo main-paper figure reconstruction (2026-11-07)

User narrowed the current request to ZuCo and the paper's figures; no TeCo now.
New entry point: notebooks/zuco_paper_figures.ipynb; see ZUCO_PAPER_FIGURES.md.
Independent Python implementation of the four main figures, with original SST
labels, all eligible word-associated fixations and −600…1000 ms epochs.
Published decoding Table 1 remains a separate, unresolved exact-replication task.

The recorded-trial/sentence-order ambiguity is bypassed by exact full-105-channel
waveform matching of released sentence EEG inside native blocks. Cloud mapping
audit 42368914686 found 48 valid sentence identities in ZAB block 1, with one
missing released waveform; no assumed 50-trial order. Released high-level MAT
schema audit 42368682021 confirms it lacks the full native event timeline.

Native source duration exports were found physically inconsistent (~1.4e14 ms).
The new preparation route recomputes duration from recorded end-minus-start,
validates 0…10,000 ms, and saves mismatch/range audits. Blink timestamps/durations
remain separately observed. The prior first-fixation archive used stored segment
lengths, so this does not establish a bug in that archive. Earlier engineering
runs using the exported duration field are superseded, not scientific validation.

Real-source pilot 42371243545 PASSED: ZAB block 1, all 105 channels, validated
word mapping/selection, nuisance model and resumable trial permutations (4 test
shuffles only). It accepted 22 negative +22 positive events after matching and
rejection. This is an engineering pilot, not a full-reader or group result.
Native trigger residual remains 3.973 ms; explicit 8-ms output-sample tolerance
used. Do not claim the published 2-ms criterion passed.

Cloud numerical tests verify the sufficient-statistic Gaussian likelihood against
a full dense covariance likelihood, trial-level channel/sign permutations,
nonunique sentence rejection, temporal duration and no double-add on resume.
Cloud layout tests 42371307216 and 42371580987 passed; actual published reference
figure images were inspected, then synthetic (watermarked) four-figure previews
rendered and visually checked. A latest test also covers saved-analysis checksums.

Full-reader ZAB native pilot 42371583171 launched to validate all eight source
blocks; inspect conclusion before claiming it passed. The full 12-reader group
analysis and 5,000/1,000-permutation run have NOT run. No actual paper-class effect
or replication claim yet. Outputs cannot be copied from existing first-fix data.

Destination is a new zuco_paper_figures folder. Verified additional Drive-space
preflight: ~1.5 GB peak allowance, ~1.01 GB null maps before compression; final
50–200 MB estimate (not full-run measured). Native sources temporary in cloud.
Keep sufficient statistics, coefficients, event audits and only E62 trial rasters;
no second all-channel EEG archive. Atomic reader/permutation checkpoints and
checksums prevent repeated additions. Delete only own null_work after successful
inference/figures. Completed runs skip downloads/refits; plots-only redraw exists.

All preprocessing/matching/statistical differences are documented with output.
Existing CNN, neutral/content classification and earlier ERP output unchanged.
