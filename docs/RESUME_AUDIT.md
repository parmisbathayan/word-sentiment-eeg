# Resume/storage audit, 2026-10-22

The existing results should be preserved and audited on resume. A full restart
is not required merely because the earlier per-session manifest count fell.

## What was checked

- The current unit/integration suite passes: 38 unit checks, synthetic v5/v7.3
  extraction, a restart without source MATLAB files, legacy-cache recognition,
  missing-neutral repair, interrupted repair and preservation of existing files.
- A separate [real-recording comparison](real_erp_compatibility.json) ran in
  the cloud using 10 readable NR/ZJS sentences, 230 words and 63 accepted epochs
  (62 neutral, 1 negative, no positive epochs in this particular sample).
- Against the original `c0d40db` runner, all sampled individual EEG waveforms
  were bit-for-bit identical, and onset tables, labels and exclusions matched.
- Maximum class-mean difference from float64 streaming accumulation was
  0.000000476837158203125 µV; stored legacy class means are never rewritten.
- A removed real neutral chunk was reconstructed identically from the saved
  onset samples. Other class files, means and the onset table were unchanged.
- The real audit did not connect to or modify the user's Drive. Its temporary
  source EEG and output arrays were deleted; only a small report was retained.

## Saved-data layout observed on Drive

One cache namespace was present: `v3_word_erp/a49719ea0d6626bc/`.
The latest inspected legacy summary recorded 27/36 reader/task analyses with
the same processing fingerprint: all 12 SR, all 12 NR, and TSR ZAB/ZDM/ZDN.
This is the summary's coverage, not certification that every trial export is
present. Earlier checks found missing neutral trial files for four of the first
five readers, despite their preserved means/onset tables and other class files.

## What the updated runner does

It validates every reader checkpoint before scheduling work, preserving all
compatible means/onsets and complete trial exports. Missing trial chunks are
reconstructed from cached onsets only. New-reader extraction and missing-export
repair have separate counts and stages. The epoch definition, labels, channel
order, filters and onset matching have not changed. Legacy and chunked storage
are read through one explicit epoch-file index/loader.

The audit is bounded: it does not prove the entire dataset has finished or that
every old archive byte has been read and checked. Startup validation checks
schema, channel/time axes, event IDs and counts; the source hash is checked for
repairs. Any inconsistent checkpoint is reported and preserved rather than
silently overwritten. The full real-data run remains to be completed in Colab.

## Recovery after the 15:17 UTC run

The Colab log ended with 29/36 validated analyses and 29/36 complete trial
exports. The seven remaining legacy checkpoints were missing required analysis
files: NR/ZKH (onset table), NR/ZKW, NR/ZMG, NR/ZPH, TSR/ZAB, TSR/ZDM and
TSR/ZDN (class means). The previous runner preserved these folders but skipped
them, so another unchanged run could not finish them.

The runner now schedules compatible checkpoints with missing analysis files for
rebuilding in `<current fingerprint>/recovered/<task>/<reader>/`. It verifies
the original source SHA-256 before extraction. Every surviving original file
remains untouched; the recovered folder becomes the sole active analysis for
that reader on subsequent runs. A reader/task can occur only once in the run
state. Changed settings, mismatched identity, or inconsistent saved data still
fail validation rather than being silently accepted. `recovery_plan.json` and
`resume_plan.json` identify the rebuilds. The final console output lists any
remaining failures directly.

The cloud integration test reproduces missing means and missing onsets across
both MATLAB formats, verifies that surviving files retain their hashes, checks
that rebuilt EEG matches the prior EEG, and reruns without source files to
confirm that each recovered reader is reused exactly once.
