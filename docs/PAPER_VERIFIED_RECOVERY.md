# Validated clock recovery — 2026-11-11

Author email is deferred until this recovery attempt is assessed.

## Actual independent audit findings

All37 previously clock-rejected blocks were tested in cloud against native
105-channel EEG and the stored word-fixation waveforms. Candidate clocks were
fit to triggers, not to the word waveforms used for validation. Word-event pairs
were established by sentence waveform identity, word rectangles, order and
matching duration. Nearest predicted clock time was never used for pairing.

- **34 blocks pass**:25,401 independently matched fixation checks, all checked
  onset errors below2ms (largest1.97505ms).
- Original54 included blocks plus34 validated candidates allow an attempt at
  **88/96 source blocks**. This is not an accepted sentiment-trial count.
- ZGW5 and ZJN1 fail the predeclared timing audit; ZDN6 still lacks an exact
  shared trigger sequence, even using only10/11 reading markers.
- Five other blocks still lack required source pairs in the discovered inventory:
  ZDN4/5/7/8 and ZKW6. A fresh metadata scan covered452 entries across26 folders
  in the full public raw/preprocessed SR trees. This is not proof of absence
  from every task, archive or release on the website.

Candidate approval requires >=100 matches across>=10 sentences,99th-percentile
error<=2ms and maximum<=4ms. All approved candidates happen to have max<2ms.
Ambiguous/missing words and waveforms are not validation evidence. This validates
many word-fixation onsets distributed across each approved block, not every
individual nuisance event or every unobserved interval. Source checksums and
all candidate errors remain in `paper_clock_recovery_verified.json`.

The old maximum-trigger-residual criterion rejected entire blocks for1–3 bad
markers. The new option does not delete those markers or simply increase8ms.
It permits a specific trigger-clock candidate only after independent waveform
validation, exact native source-file SHA256 checks and parameter revalidation.
Unapproved blocks keep the old exclusion behavior. Neither timing validation
nor successful extraction guarantees the same sample or finding as the paper.

## Run the reconstruction after the current TeCo job

1. [CPU preparation](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/zuco_paper_recover_prepare_cpu.ipynb): CPU,Run all; wait for CPU PREPARATION COMPLETE.
2. [GPU analysis](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/zuco_paper_recover_analyze_gpu.ipynb): T4 GPU,Run all,same Drive.

No new accepted full-run counts or group findings exist before these notebooks
complete. Raw/clean individual trials are retained. New results go only to
`Thesis/Results/word_sentiment_eeg/zuco_paper_figures_recovered/`; the completed
`zuco_paper_figures_staged/` archive/results remain unchanged.

Adding blocks changes each reader's shared nuisance basis/coefficients and
matching pool. Refitting the full reader is required; appending new cleaned
trials to old averages would combine different fitted models. Completed readers
in the NEW reconstruction resume with checksum validation. Unfinished readers
may need repeating after interruption. GPU permutations resume without double
reader additions and require no repeat native extraction.

**Additional storage cap:4GB**, including about1GB temporary null maps and a
250MB safety reserve. CPU archive allowance approximately2.7GB. Authenticated
Drive quota is checked before extraction. No additional source EEG copies remain
on Drive. Temporary permutation maps are removed after successful final figures.
Do not run multiple jobs into the same destination.

## Terms

- Block:one source recording chunk (eight per reader,twelve readers).
- Timing marker/trigger:a shared event message such as reading-screen onset,
  logged in both EEG and eye-tracking systems to relate their clocks.
- Trigger-sequence mismatch:the common event messages/count/order do not form
  an exact unique pairing. Reading order is not guessed to fill a missing event.
- Trial matching:selecting positive/negative trials with comparable word length
  and fixation-timing properties, balancing counts within the declared strata.

## Tests

Synthetic clock audit and proof-binding tests reject incomplete event sequences,
corrupt waveforms, changed source files, insufficient evidence and out-of-range
validation metrics. They preserve bad-marker diagnostics and restore original
extraction functions after the scoped recovery operation. The native extraction
pilot exercises ZAB3, a block previously rejected by the8ms marker rule.
