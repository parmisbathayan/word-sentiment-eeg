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

Full-reader ZAB pilot 42371583171 exposed blocks above the 8-ms limit. The
conservative exclusion policy was added, then pilot 42371942981 PASSED all eight
source blocks: five included, three explicitly excluded (3, 5, 7); 243 mapped
sentences, 287 negative +286 positive accepted events, 286 ranked trial pairs.
All 105 channels completed shared-reader nuisance fitting. Four test permutations
only. See paper_full_reader_validation.json; this is not group inference. The full 12-reader group
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

Final notebook execution is pinned to tested code revision
5fc7e92b5dbb0d6697e517d21835094ba520e5aa. Cloud check 42372106932 PASSED the
numerical, input-change/resume, source-exclusion and complete rendering tests.
The notebook can be opened from main but fetches this pinned execution revision;
future unrelated commits will not alter an interrupted run's preparation code.
Remaining methodological work: recover better source synchronization to retain
clock-excluded blocks; strict original 2-ms parity is not established. Current
figures report the reduced source coverage. No full 12-reader figure run yet.

## CPU -> GPU persistent preparation (2026-11-10)

User explicitly wants CPU data preparation followed by a fresh GPU runtime.
New notebooks: zuco_paper_prepare_cpu.ipynb, zuco_paper_analyze_gpu.ipynb.
Both pin execution to 53d11de68587c008c8ed18ede7ad4d4895309df0.
New script scripts/zuco_paper_stages.py and helper paper_stage_archive.py.
New folder zuco_paper_figures_staged; original active/pinned reproduction untouched.

Reader raw/clean accepted signals are saved as LOSSLESS float32 before any
permutations, not collapsed means or float16. Include all105 channels,201samples,
channel names, times/units, positive/negative labels, trial IDs, metadata and pair
indices. prepared.json is the CPU checkpoint, separate from GPU manifest.json.
CPU reruns verify/skip prepared readers. GPU loads these files without native
source reads, alignment or cleaning. Per-attempt status files preserve history.
An unfinished CPU reader can still need repeating; do not promise per-channel
resume. Earlier incomplete beta-only outputs lack validated full trial provenance;
no silent adoption/deletion. The first staged CPU pass must build this archive.

Storage allowance explicitly increased from old1.5GB plan to4GB TOTAL, including
~1.01GB temporary null maps+250MB margin; CPU prepared files capped~2.7GB with
uncompressed next-reader write guard. Drive free-space gate runs before extraction.
No Mac dependencies/data installed/downloaded. Numerical tests all in cloud.

Cloud native test42797968301 PASSED: ZABblock1 all105channels -> CPU archive ->
separate process analysis(test4permutations onCPU) -> rerun CPU skips sourcework.
No claim of GPU hardware timing benchmark or full12-reader result. Regression
checks include exact float32/ID roundtrip, no repeated permutations, corruption
and storage rejection. New metadata/montage unit checks in latest paper-stages CI.

Terminology: the reproduction's GPU stage runs permutations, not CNN training.
Individual trials are preserved for later classifiers. Existing CNN still consumes
the previous first-fixation archive; no new classifier silently added in this turn.

## ZDM blink validation failure and checkpoint-preserving fix (2026-11-10)

User saved ZAB CPU checkpoint, then ZDM failed in block5. Native cloud source audit
42803449309 confirmed one blink with equal latency/endtime7838804 and recorded
2ms duration, both raw/corrected. Repair accepts this only when recorded duration
matches the measured native gaze period (<=4ms); all other endpoint checks remain.
New single_sample_blinks_retained/source_rows/recorded_eye_sample_ms audits.

Exact prior53d11de protocol code hashes are embedded in migration helper. Verify
all untouched code/config/data/software, prepared-reader checksums and positive
historical endpoint ranges; no GPU outputs or complete CPU root marker allowed.
Save old protocol+recovery lineage, keep ZAB bytes unchanged, restart incomplete
ZDM. Both stage notebooks now pin be08d01b94212425f9ac3de4649d89d469f37bc5.
Unit cloud run42803758567 PASSED including valid source-sized single blink,
negative/incompatible duration rejection, exact migration preservation and changed
method/environment/input refusal. Full nativeZDM pilot42803764187 and repaired
endpoint audit42803769178 launched; inspect results before claiming success.

Final validation: native ZDM full-reader pilot42803764187 PASSED. All8 blocks
were inspected; blocks3/5/7 excluded by the unchanged8ms clock threshold
(42.448/56.522/22.346ms). The single-sample blink in block5 no longer crashes
validation. Included blocks completed all105-channel cleaning and persistent CPU
trial storage; a separate process completed test permutations; CPU rerun skipped
ZDM. Four test permutations only, no full group result. Endpoint audit42803769178
also passed and reports measured2ms eye period and one retained source blink,
row7, in both raw and corrected exports. No user Drive data modified in testing.

## ZDN incomplete source-pair recovery (2026-11-10)

User hit Missing native eye-event source for oip_ZDN_SNR4_EEG.mat. Direct OSF
metadata read confirmed actual source omissions, not merely an SR/SNR typo:
ZDN4 raw+corrected missing,5corrected missing,7/8raw missing. A corrected8file
in the raw folder must NOT be used as the raw event timeline. Resolver now joins
exact reader/block/type, supports SR/SNR/zero-padding, checks alternate folder
for correctly typed misplaced files, rejects ambiguous candidates. Missing pairs
become explicit excluded_missing_source_files diagnostics, no downloads/no guessed
onsets, unknown fixation count (not zero). Coverage rendering handles absent
synchronization and reports missing sources. Figure1 counts available observations.

Both initial53d11de and blink-fixedbe08d01 exact staged code sets can migrate an
incomplete CPU run, retaining all checksum-verified completed readers with valid
duration audits and originally complete source pairs. Settings/dependencies/data/
software must match; no GPU work permitted. Preserve original endpoint recovery,
add source_inventory_recovery/protocol backup; do not rewrite reader bytes.

Notebook revision3749540 pins execution665c53247d0c00828913c8a9dd57cf7f3a214e0d.
BothCPU/GPU must use this updated pin. CPU table includes source block coverage.
Unit42881341453 PASSED source pairing/missing/ambiguity/migration plus prior
lossless/archive/blink tests. Run42881344710 audited inventories for all12 readers
successfully and is running fullZDN preparation/CPU-to-analysis test; inspect final
conclusion before reporting native pass. Existing user Drive outputs untouched.

Final source recovery validation:42882099801 PASSED full ZDN source loop,
all105-channel cleaning of accepted data, persistent CPU trials, separate-process
test permutations and CPU rerun skip. The prior665c532 native test exposed ZDN6
unverified shared trigger sequence; prepare caller now records that known source
validation failure as explicit alignment exclusion without changing synchronization
logic/tolerance or guessing triggers. Only ZDNblock2 survives:1/3 fail8ms,
4/5/7/8missing source pairs,6unverified trigger sequence. Coverage is materially
limited; report it and consider with/without-ZDN sensitivity later.
Latest notebook revision a4fe239 pins3a547ae7b7c8cc5fd4914c29acb1192013765a8a.
Unit42882096987 PASSED. Metadata-only all12 audit is stored in
paper_source_pair_audit.json; ZKW6also missing raw ET and handled explicitly.
Native numeric summary in paper_ZDN_source_validation.json. No actual user Drive
files modified in tests; new run migrates known prior protocol and verifies/keeps
prepared ZAB/ZDM, then rebuilds unfinished ZDN. Do not delete output folders.

## Grid-verified fixation/saccade samples and complete endpoint audit (2026-11-11)

User CPU run saved earlier readers then failed ZJM6 on fixation duration validation.
Source target audit42904208364 found equal endpoints5941166 and corrupted stored
fixation duration140484085284866, in raw+corrected. All-reader audit42903540454
PASSED its scan of186available eye files across12readers, identifying only two
previously unhandled physical endpoint cases: that ZJM6fixation and ZJS1saccade
(timestamp6899426, corrupted stored duration140458315481090), both raw/corrected.
No other negative/nonfinite/>10s failures in the audited files; known zero blinks
already supported. Audit summary is docs/paper_eye_endpoint_audit_summary.json.

New code8ed1628a09c108644e47b2929b08da24855f41f1 supports equal endpoints as one
native sample ONLY when actual native gaze timestamps verify the event is on the
measured<=4ms grid; prior recorded-duration single-blink exception retained.
Other positive end-minus-start durations and all rejection thresholds unchanged.
Corrupted stored duration is never used for fixations/saccades. Two-ms fixations
remain excluded from sentiment trials by<100ms. Nuisance model retains observed
onsets. Actual-target repair audit42905072148 PASSED both raw/corrected ZJM6and
ZJS1 with one sample event each; no unhandled endpoint error remains in those
four files. Unit42905069383 PASSED grid/offgrid/negative/NaN and sample exclusion,
plus checkpoint migration preserving source-fixed readers including missing files
and old recovery history. Full nativeZJMpilot42904745881 still running; inspect
before claiming completed cleaning/data handoff.

Known prior3a547ae exact code signatures added to migration alongside53d11de and
be08d01. Validate same data/settings/software/dependencies, all reader checksums
and historical positive duration audits; recognize existing missing-file exclusions
without treating them as data. Immutable protocol_migrations recovery history;
no overwriting prior endpoint/source recovery. Earlier prepared readers are skipped;
unfinished ZJM restarts CPU processing. No user Drive outputs changed in tests.
Notebook revision0d993b9 pins8ed1628 for BOTHCPUandGPU. User must use updatedGPU
as well after CPU completion. No source-only older notebook should be rerun.

Final native validation42904745881 PASSED: inspected all8ZJMblocks, block6
explicitly excluded by unchanged8ms alignment rule(max10.165ms) after duration
parsing; all105channels cleaned for included blocks; CPU archive saved; separate
process completed test permutations; CPU rerun skipped the reader. Only4test
permutations, no full-group result. Summary in paper_ZJM_source_validation.json.
All186available eye-file audit + actual four-file ZJM/ZJS repairs + migration and
sampling regression tests completed successfully. Latest notebook0d993b9 pins
8ed1628. Resume withCPU Runall inclsetup; expect CHECKPOINT MIGRATION and KEEP CPU
READER for earlier saved readers; incompleteZJM restarts. UpdatedGPU later must
use same pin. No folder deletion or extra trial archive/storage budget increase.

## 2026-11-11: completed staged ZuCo review and TeCo adaptation

- Verified private Drive staged manifests: all 12 ZuCo readers complete, 6,634
  accepted trials /3,314 ranked pairs. Of96 source blocks,54 included,37 clock
  exclusions,5 missing source pairs.61 candidate clusters; zero corrected p<.05,
  minimum p=.3866133866. Main published topographic effect NOT reproduced.
- Saved control figure shows word-frequency imbalance (reader-paired p=1.05e−7).
  Fixed 224–280/304-ms maps are reference windows, not detected effects. Exact
  saved TCT interval not recomputed in this review; do not invent its bounds.
- Review saved in docs/ZUCO_COMPLETED_COMPARISON.md. No new ZuCo extraction.
- Added independent TeCo Task1 RAW-FRP adaptation: teco_paper_stages.py,
  teco_paper.py, teco_paper_stats.py, render_teco_paper.py; two Colab notebooks
  teco_paper_prepare_cpu and teco_paper_analyze_gpu. All stored word fixations,
  controls' reading included, lexical=contextual label agreement, no sentence
  constraint, no neutral in this binary paper-style route. All native126channels.
- Full TeCo blink/saccade/keypress timeline is NOT validated. Do not present this
  as full cleaned replication; Figure3 and full eye controls unavailable. Exact
  length/target-duration matching replaces neighbor-timing matching explicitly.
- Additional2GB total cap incl~.61GB temporary null maps;~1.24GB CPUtrial allowance.
  Persistent float32 epochs, checksummed reader checkpoints and deduplicated
  permutation resume. Native private run has NOT been performed by this agent.
- Cloud synthetic validation workflow: .github/workflows/teco-paper.yml.
  First run43039298287 passed; further small plot-caption/scale edits retested.
  Final release is immutable tag teco-paper-raw-v1; see git log/test run for hash.
- Nothing installed/downloaded as dependencies or EEG datasets on Mac. Used
  existing bundled Python only for stdlib syntax/notebook construction.

## 2026-11-11: user requests recovering excluded ZuCo blocks

- Read all 12 saved source_mapping.json files on Drive (metadata only). The37
  alignment exclusions split into36 max-affine-residual failures and1 trigger
  sequence mismatch (ZDN6). All36 residual failures contain only1–3 anchors >8ms
  among98–102 anchors; median absolute residuals~0.68–1.92ms.18/36 blocks have
  maximum residual <=12ms. Detailed nonstimulus audit: paper_clock_recovery_audit.json.
- Original global-max rule rejected whole recordings for sparse discrepant
  anchors. This warrants validation of local/sentence-level salvage. NO clocks
  changed, blocks recovered, or completed results overwritten in this review.
  Validate against independent released word-fixation waveform/onset matches;
  do not remove anchors or loosen thresholds solely to match paper trial count.
- Five missing-source blocks: ZDN4(raw+corrected ET),5(corrected ET),7/8(raw ET),
  ZKW6(raw ET). These are missing from the currently discovered native inventory,
  not proved permanently unavailable or absent from every dataset release.
- IMPORTANT frequency clarification: paper2.7/3.3 used a two-sample comparison of
  frequencies for UNIQUE word identities. Our Figure4 uses reader-paired means
  over accepted FIXATIONS. Thus p=1.05e−7 is not an equivalent replication of
  their frequency test. Do not say that p disproves their frequency control or
  that absence of significance means perfectly matched word frequencies.
- User asked for email draft requesting authors' preprocessing/deconvolution,
  matching/statistics/word-label mapping code and selected fixation IDs, plus
  synchronization guidance/source-file locations. Draft in chat only; no email
  sent and no mailbox draft created. Next step is validated timing recovery,
  not another blind full extraction or forcing exactly12,789 trials.

## Active 2026-11-11: independent timing recovery attempt (email deferred)

User explicitly asked to attempt recovery before contacting authors. No email
sent; keep previous draft for after recovery investigation. Newly added audit:
- `paper_clock_audit.py`: original affine and Theil–Sen trigger-clock candidates;
  reports all outlying anchors; no thresholds/extraction behavior changed.
- `audit_paper_clock_recovery.py`: independently locate released sentence and
  stored word-fixation waveforms in105-channel native EEG. Associate corrected
  ET fixations by sentence trigger identity, word bounds, order and duration,
  NEVER nearest predicted clock time. Candidate clocks validated on matched
  waveforms, not the triggers used to fit them. No sentiment labels involved.
- Predeclared audit pass: >=100 fixation checks across>=10sentences, p99 error
  <=2ms and max<=4ms. This is an audit criterion, not a pipeline adoption step.
- Source search traverses the complete raw/preprocessed public SR folder trees,
  including misplaced subject files; does not claim every OSF archive searched.
- Cloud synthetic checks pass (push run43065265998). Real source audit dispatched
  as43065333762: ZABblocks3/5/7, ZPHblocks1/2/3/7; ZABalso broadETsource search.
  Workflow `.github/workflows/paper-clock-recovery.yml`, native input true.
- Public source EEG downloads permitted ONLY in temporary cloud runner storage;
  no Mac dependency/environment/dataset installation/download. Artifact output
  smallJSON only, no EEG/stimulus text. No Drive quota consumption or old-result
  edits. Investigate candidate clocks and validation coverage before any rerun.

### Recovery pilot outcome and extension

- Real native audit43065333762 completed successfully. All7 rejected pilot
  blocks (ZAB3/5/7, ZPH1/2/3/7) pass Theil–Sen candidate timing validation.
  5,487 fixation waveform checks across315 sentence-reader instances; max
  absolute error1.8985ms across these checked onsets. Large trigger residuals
  remain visible; candidate method does not delete anchors. NO new accepted
  sentiment trials or updated FRP results yet. Metrics saved in
  docs/paper_clock_pilot_results.json; all underlying signals stayed in cloud.
- Public source scan covered452 entries across26 folders in complete raw and
  preprocessed SR trees. Still no source pair for missingZDN4/5/7/8 and ZKW6.
  Do not claim absence from all other tasks/releases/archives/the entire website.
- Full remaining audit dispatched43066690280, revision036cff5: remaining29 numeric
  residual failures plus ZDN6 sequence mismatch,10 readers,max3 cloud workers.
  It skips the7 already-tested blocks rather than downloading them again.
- New audit-only reading-sequence fallback requires an exact complete10/11
  sequence if full10/11/12/13 sequence disagrees (e.g. missing question trigger).
  It still requires independent waveform validation. No guessed event ordering.
- Next: inspect completed native job logs via gh api, retaining only summary
  metrics/source metadata, not EEG or full fixation-event data on Mac. Then
  decide conservative adoption/reconstruction scope; email remains deferred.

### Independent audit complete; guarded recovery extraction being validated

- Full native run43066690280 succeeded for all10 remaining readers. Combined
  with pilot43065333762,37 clock-excluded blocks audited.34 approved,25,401
  waveform checks,max1.97505ms. Could attempt88/96blocks (54old+34); not yet
  newly accepted full-run trial counts. UnresolvedZGW5,ZJN1,ZDN6;5source pairs
  still missing. The authoritative consolidated report is now
  docs/paper_clock_recovery_verified.json (inclsource hashes and source scan).
- Added proof-bound context wrapper paper_verified_recovery.py and separate
  zuco_paper_recovered_stages.py. Old helpers/old pinned notebooks unchanged.
  New outputzuco_paper_figures_recovered,4GB cap,CPU->GPU same archive layout.
  Exact source SHA/candidate parameters and numeric evidence required. No mere
  relaxed threshold. Requires refitting full reader nuisance models/matching.
- First native extraction pilot43068181611 stopped before EEG downloads due to
  a subset-selection bug (proofs for omitted blocks treated as missing). Fixed
  in1b4908e and added synthetic coverage. New native pilot43068348201 running
  ZAB3 through CPU archives and4+4CPUpermutations (skip group inference).
- Recovery notebook files prepared locally, not released/tagged until native
  pilot passes. They pin immutable verified-clock-recovery-v1 (tag pending).
- Email remains deferred. No Drive changes; no Mac dependencies/EEG downloads.

- Intermediate pilot/remaining JSON reports consolidated into the authoritative
  paper_clock_recovery_verified.json (full metrics, source hashes and source
  metadata search preserved). Earlier snapshots remain in git history.

### Recovery release ready

- Native extraction pilot43068348201 PASS (revision1b4908e): recoveredZAB3
  included,45negative+45positive accepted, raw/clean float32 archive verified,
  saved-trial CPU->permutation handoff completed (4+4engineering permutations).
  No group inference claim. No raw EEG or trial arrays downloaded toMac/Drive.
- Latest synthetic proof-binding tests43069023056 PASS (f764c2b).
- Final notebooks zuco_paper_recover_prepare_cpu / analyze_gpu pin immutable
  verified-clock-recovery-v1. Newoutzuco_paper_figures_recovered,4GB additional
  cap incl~1GB temporary nulls. Source proofs and validation status bound into
  protocol. Existing ZuCo/TeCo scripts and outputs unchanged.
- Full user reconstruction NOT run yet; cannot claim new full-run counts.
  Wait for currentTeCo job to finish,CPUrecoveryRunall,thenGPUanalysisRunall.
  Explain why reader-level model/matching refit is necessary; do not append90
  pilot trials toold6634. Email remains deferred until recovery is assessed.

## 2026-11-12: recovery initial quota stop

User's first CPU attempt stopped before protocol write/source reads: required
4,000,000,000 bytes, authenticated free3,726,021,983. Updated BOTH recovery
notebook commands to3,600,000,000bytes. Same immutable scientific code pin,
no script changes, no data precision/channel/time/permutation reductions.
Temporary null reserve1,013,040,000 + safety250,000,000 leaves2,336,960,000
CPUarchive allowance. CPU/GPU caps match, so handoff protocol stays consistent.
Initial reserve now fits reported free space, but this is not proof of final
size; per-reader archive guards still stop safely if actual output hits cap.
No Drive data deleted or modified by this fix. User should reload CPU notebook
fromGitHub main and Runall; no need to free/delete results merely for this stop.

## 2026-11-12: Colab checkout setup failure fixed

User's recovery setup fetched verified-clock-recovery-v1 but checkout of that
local tag name returned128 before extraction. Original Git stderr not supplied,
so don't assert a broader cause. Updated BOTH recovery and BOTH TeCo paper
notebooks: exact immutable underlying commit SHA, fetch then verify FETCH_HEAD
commit, detach FETCH_HEAD, verify HEAD. Existing correct checkout skips fetch.
Git failures now include captured command output for diagnosis. Scientific code
unchanged: recovery5049fadfe80153a7ccb7cce11e18e19bac0f3f62;
TeCo9651d15759091b01635180282962d9e8129c9a3b. Recovery budget remains3.6GB
in both stages. No deletion/reset of runtime or Drive results required.
Cloud run43200071546 passed: execute actual Git setup AST from all4notebooks
on fresh --no-tags shallow clones, then unchanged resume. No local clones,
packages/environments or EEG downloads. Main notebook code pushed7f0b195.

## 2026-11-12: Zyion quota / Trash investigation — IMPORTANT BEFORE GPU RUN

- User asked what can be removed. Connector profile is Parmis Gmail, NOT Zyion;
  shared-folder metadata cannot establish Zyion ownership/quota. Used IAB Drive
  tab9 account2; visibly verified Zyion Mutch zyionygvmutch39@gmail.com.
- Drive quota showed11.53GB/15GB used,3.47GB available (Google UI units);
  GoogleDrive11.5GB,Gmail29MB,Photos0. Largest live files were12 prepared_trials
  archives~9.2–159.1MB each. GoogleOne largeDrivefiles category~1GB.
- Zyion Trash visibly contained null_work plusmany tct_XXXXX__12.npz and
  cluster_XXXXX__12.npz files ownedbyZyion. Completed oldstaged manifest verified
  statuscompleted, all12readers and saved analysis/figures; null_work absent
  fromlivecompletedrun. Old archived incompletepaper/null_work wasemptytoo.
- Actual quota accounting bug: permutation_reader writes newimmutable generation
  perreader and unlinksoldgeneration. ColabDrive deletion sendsoldfiles toTrash;
  disk_bytes(root) ignoresTrash. Thus~1GB nominal work reserve can accumulate
  manyGB of deleted generations. Finalrmtree(null_work) also fillsTrash. Repeated
  atomic modelcheckpoint replacements may addsmaller overhead. Do not promise
  3.6GB actualaccount footprint until GPU temp persistence is redesigned/tested.
- No filesdeletedbyagent. User independentlyemptiedTrashmid-investigation.
  Verified page nowexplicitly saysTrashisempty, quota sidebarstill~11.52GB.
  OfficialGoogleOneHelp answer9776477 saysbulk-deletion quota updates can take
  up to48–72hours. Don't delete additional EEG/trials merely for this lag.
- Files toKEEP: trial_archive_125hz, classification featurecache, accepted
  prepared_trials.npz, private labels, saved analysis/plots/manifests. Superseded
  trial archives canbeassessedlater,afternewreconstructionvalid,notnow.
- Pending: fix/retest GPU temporary-file persistence so oldchunkgenerations do
  not accumulateinDriveTrash. Must preserve CPU archives + scientific protocol,
  randomseed/numerical semantics and meaningful resumability. Do NOT silently
  moveallprogress tovolatile runtime and claim cross-runtime resume. No GPU
  cleanup involving permanentdeletion without action-time userconfirmation.
