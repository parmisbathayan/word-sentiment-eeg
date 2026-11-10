# CPU preparation, then a GPU runtime

1. Run `notebooks/zuco_paper_prepare_cpu.ipynb` on CPU. Wait for **CPU PREPARATION
   COMPLETE** and the per-reader prepared-trial table.
2. Switch runtime type to T4 GPU (a fresh runtime is fine), then run
   `notebooks/zuco_paper_analyze_gpu.ipynb`.

The second notebook reads verified Drive files. It never reads/downloads native
source EEG or refits the cleaning model. This reproduction's GPU work is trial
permutation testing; mixed models still use CPU. It is not CNN training. The
older CNN notebook uses the earlier first-fixation archive and remains unchanged.
The staged archive retains complete accepted raw/clean fixation trials and their
labels/identities, so a later classifier can consume these paper-specific trials.

## Persistent output

`Thesis/Results/word_sentiment_eeg/zuco_paper_figures_staged/`

- `protocol.json`: pinned method/code/environment and storage settings.
- `readers/<reader>/prepared_trials.npz`: lossless float32 raw and cleaned trials,
  representation × accepted event × 105 channels × 201 time samples. No float16,
  clipping, or reduction to aggregate features. Negative/positive labels, original
  event IDs and ranked positive/negative pair indices are included.
- `accepted_trial_metadata.csv.gz`: corresponding row order, reader, word,
  word index, sentence key, block/onset, original labels and selection flags.
- `statistics.npz`, source mappings, coefficients and diagnostics support figures.
- `prepared.json`: checksum-protected **CPU completion** checkpoint. It is written
  before any permutations; completed CPU readers are skipped on rerun.
- `manifest.json` inside each reader: **GPU permutation completion** checkpoint.
- `cpu_preparation_manifest.json`: all requested readers are CPU-prepared.
- `status/`: separate timestamped status file for each notebook attempt, preserving
  the previous attempt's last status when a new execution starts.
- `null_work/`: resumable permutation sums; deleted only after successful figures.
- `analysis/`, `figures/`: the four paper-style figures and numerical audits.

The preparation preserves the existing paper policy: original SST labels, eligible
word-associated fixations, 1–50 Hz/125 Hz, −600…1000 ms, raw/clean shared trial IDs,
explicit clock exclusions. It does not add neutral trials or change selection.
See ZUCO_PAPER_FIGURES.md for scientific departures from the published methods.

## Storage and earlier outputs

The staged workflow has a **4 GB total disk cap**, not a 4 GB trial archive plus
extra permutation files. About 1.01 GB is reserved for 5,000/1,000 null maps and
250 MB for final statistics/figures and atomic-write overhead. CPU checkpoints
are limited to approximately 2.7 GB, including trial arrays, audits and models.
A conservative uncompressed-size check runs before writing the next reader's
trial archive. Free Drive account space is checked before source extraction.
The actual archive is losslessly compressed, so it can occupy less than the cap.
This is an allowance, not a measured full twelve-reader output size.

The old `zuco_paper_figures/` is untouched. Stop its active job before starting the
staged version to avoid concurrent storage use. Its incomplete outputs lack the
validated individual-trial handoff and model-design provenance; the first staged
CPU pass must create this archive. Do not claim that old coefficients were safely
reused or delete them automatically. No automatic relabeling or rebalancing.

A runtime ending **during** a CPU reader can still require repeating that unfinished
reader. Once that reader's `prepared.json` exists and verifies, changing runtimes
will not repeat its CPU work. Interrupted GPU permutations resume reader/chunk
membership with trial-input hashes, without adding the same reader twice.

## Single-sample blink source correction (2026-11-10)

ZDM block5 contains a blink with latency=endtime=7838804 and recorded duration
2 ms. This is present in both raw and corrected eye exports. The original validator
incorrectly required every endpoint difference to be positive. The corrected rule
retains a zero-span blink only when its positive recorded duration equals the
measured native gaze sampling period (maximum4ms). Onsets are unchanged. Negative,
nonfinite or incompatible zero-span records still fail. Audit fields record the
source row and period; this is not a blanket relaxation of eye-event validation.

The exact53d11de staged protocol can migrate this endpoint-only correction when
all other settings, code dependencies, sources and numerical versions match.
Prepared reader checksums and historical positive duration ranges must verify.
Migration is restricted to incomplete CPU preparation without GPU work. It records
the old protocol and preserved reader checkpoint hashes. Existing ZAB files are
not rewritten, relabeled or reprocessed. The unfinished ZDM reader must restart
its CPU work because it had no complete CPU checkpoint. Both notebooks pin the
same updated execution revision. Keep the existing Drive folder and files.

## Incomplete native source pairs (ZDN, 2026-11-10)

The public source listing has preprocessed ZDN EEG blocks without the raw and/or
corrected eye-event files required by this workflow. Metadata for block4 lacks
both eye files; block5 lacks the corrected file; blocks7/8 lack the raw eye file.
A corrected block8 file in the raw folder is not an uncorrected timeline.
The resolver now joins exact reader, block and file type with SR/SNR aliases,
searches the alternate folder when a correctly typed file was misplaced, rejects
ambiguous choices, and explicitly marks genuinely unavailable source pairs.

Missing-file blocks are not downloaded, inferred or assigned to another reader.
They are recorded as excluded_missing_source_files in the reader source mapping
and final source_coverage.csv. Their unobserved fixation count stays unknown;
it is not fabricated as zero. Figure1 counts only available observations and
points to the coverage report. This reduces source coverage relative to the
complete published cohort; it does not establish an exact numerical replication.

The exact previous blink-fixed version (be08d01) and initial staged version
(53d11de) have guarded incomplete-CPU migrations. Unchanged code dependencies,
settings, environment and all saved-reader checksums/duration audits must verify.
Old completed readers had complete source pairs because their earlier resolver
would otherwise have stopped; they are preserved without rewriting their data.
Prior migration records are retained; source_inventory_recovery.json records the
new lineage. Partial readers restart preparation; saved readers are skipped.
Recovering absent public source files from another release remains future work.

The all12-reader file inventory audit also found ZKW6 missing its raw eye-event
file; this is handled by the same explicit exclusion policy. See the metadata-only
paper_source_pair_audit.json. Other readers have complete named file pairs; this
does not guarantee that all their signal/timing checks will pass.

The native ZDN test additionally found block6 has a missing/reordered/ambiguous
shared trigger sequence. Such a source now gets an explicit unverified-trigger
alignment exclusion without inventing a clock. The fixed synchronization helper
itself is unchanged; the caller records the known validation failure and omits
that block. Previously completed readers could not have contained such a failing
block because the old code would have stopped before a completion checkpoint.
Timing tolerances remain8ms. Source coverage may therefore be substantially
smaller than8 blocks for some readers. Later checks should consider a sensitivity
analysis excluding a reader with very limited remaining source coverage, and
recovering missing files or better synchronization from an alternate release.
