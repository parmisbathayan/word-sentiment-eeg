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

1. Verify/push the neutral/nonneutral storage-only fix and provide its link.
   Do not force another original analysis/extraction run.
2. Preserve this checkpoint and the current code commits.
3. Investigate the native timing discrepancy against EYE-EEG/source versions.
   No discarding anchors or silently loosening a replication criterion to pass.
4. Validate original word-bound/event and SST joins. Keep all word-associated
   fixations for the paper route, unlike the historical first-fixation pipeline.
5. Implement paper exclusions/matching with sequential counts, paired raw/clean
   epoch IDs, FRP inference, 1/10/20/40 averages with nonoverlapping membership,
   training-only NCA/tuning and documented evaluation. Unknown original settings
   stay explicit; the 95/5 joint reader×class stratification has a feasibility
   ambiguity for 328 averages (17 test samples versus 24 strata).
6. Full per-reader nuisance fits should share coefficients across that reader's
   blocks without pretending discontinuous blocks are contiguous. The current
   one-block implementation is only an engineering pilot.
7. Budget persistent outputs before scaling up. Then adapt to TeCo only after
   verifying full saccade/blink/keypress timing, sampling and Persian labels.
   TeCo reading pickles alone have not established a full continuous timeline.

## Do not confuse outputs

Historical ERP plots: 30 Hz. Individual archive/classification: 40 Hz at125 Hz.
New reproduction engineering pilot: explicit independent 1–50 Hz Python FIR on
released corpus-preprocessed continuous EEG, not a fresh Automagic/MARA run.
No new paper FRP statistics, paper decoding scores or TeCo reproduction results
have been produced. User's presentations and original results remain intact.
