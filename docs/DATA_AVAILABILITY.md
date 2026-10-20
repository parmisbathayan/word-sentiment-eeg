# Verified availability of word-segmented EEG

Evidence is saved in [source_availability_audit.json](source_availability_audit.json).
The prepared runner's actual `iter_sentences` and `locate` functions were used
in the [2026-10-20 cloud audit](https://github.com/parmisbathayan/word-sentiment-eeg/actions/runs/39455869451).
These checks used real source files, separately from the synthetic integration tests.

| Task | Evidence | Result |
|---|---|---|
| SR | Existing real-data extraction audit, 12 readers | 9 readers had word fixation EEG; approximately 94.0–96.7% of inspected fixation segments matched sentence EEG. ZDN, ZGW and ZJM lacked `word.rawEEG`. |
| NR | Fresh cloud inspection, reader ZJS, 10 readable sentences, 230 words | 108 finite fixation segments; 99 exactly matched sentence EEG, 9 unmatched. |
| TSR | Fresh cloud inspection, reader ZPH, 10 readable sentences, 275 words | 304 finite fixation segments; 297 exactly matched sentence EEG, 7 unmatched. |

NR and TSR have only been sampled here; this is not a claim that every reader
or every word in these tasks has usable segments. The full notebook checks
each selected task/reader and records automatic exclusions. Missing waveforms
cannot be reconstructed from band-power features alone.

## Data layout and why both fields matter

- `sentenceData[i].word[j].content`: the surface word and original position.
- `sentenceData[i].word[j].rawEEG[k]`: the EEG during fixation k on that word,
  with 105 channels and a variable number of samples. A word can have zero,
  one or multiple fixations; rejected fixations can be NaN.
- `sentenceData[i].rawData`: the surrounding sentence EEG, used to locate
  the stored segment and extract a pre-onset baseline and post-onset epoch.

For example, the real TSR sample contained a fixation on **won** with shape
105 × 62. It matched sentence EEG exactly at zero-based sample 343 (686 ms
after the sentence EEG's first sample at 500 Hz). The original first fixation
is used for the word epoch; later fixations are not substituted when the first
one is missing or invalid.

The existing word-level band-power NPZ caches are useful for feature analyses,
but they do not contain the time-domain waveform needed for these ERPs.
The runner therefore reads the original MATLAB fields above. Onset samples
are recovered by full-segment matching; no timing or missing EEG is invented.

The fresh source downloads ran only on GitHub's cloud runner, one file at a
time, and were deleted after inspection. No EEG was downloaded to the Mac.
