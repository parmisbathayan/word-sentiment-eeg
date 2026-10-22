# Persian TeCo implementation traced from existing scripts

Checked 2026-10-23 (Asia/Tehran). This is the user's Persian TeCo corpus.
The native-input adapter is `src/word_sentiment_eeg/teco.py`.
No private Persian text, EEG arrays or participant-level report is committed here.

| Existing source inspected | Evidence | Implementation decision |
| --- | --- | --- |
| `Thesis-Load & Explore data.ipynb`, cells 7, 13, 29, 63, 65–68 and saved outputs | `SAMPLE_RATE=1200`; sentence `raw_data` is 126 × time; `word.rawEEG` is one waveform per fixation; `rawET` has time/gaze/pupil columns; `fixPositions` lists fixation order | Locate the first stored waveform inside the sentence signal, validating all 126 channels. Do not treat fixation-order indices or eye-tracker timestamps as EEG sample offsets. |
| `Thesis-Data Structures.ipynb`, cells 2–11 | Native files come from `pickle_files_acc3/acc5`; controls and unfixated words are skipped in derived lists; native `sentenceId` differs from English source `sentence_id` | Support original account-split paths and current `teco_pickles` paths; retain source word indices and separate ID domains. Do not use compressed TRT-list positions as original word positions. |
| Same notebook, cell 10; corrected `glim_teco_zero_shot.py` | `resample_poly(..., up=1, down=1200//500)` produces 600 Hz, despite the intermediate filename | Use the native documented 1200-Hz reading signal for this branch. Do not silently treat the older intermediate as 500 Hz. |
| Existing `brainshaping/teco.py`, `EEG-spatial/eeg_spatial/teco.py`, Yaser reproduction loader | Sentence joins remove whitespace/ZWNJ; native word `content` preserves tokens; thin files contain time-collapsed vectors; `(2,)` placeholders mark missing features | Reuse normalized Persian-text joins and explicit word content. Keep original sentence sentiment separate from pending word sentiment; missing waveforms remain exclusions. |
| Corrected GLIM TeCo preflight | Interpolated 105-channel arrays follow per-reader trial order in `df_trials.parquet`, not the label CSV | Do not index derived EEG by CSV row number. Native EEG avoids this intermediate, using its own sentence text/ID and word dictionary. |
| Existing TeCo montage/structure reference | Native montage has 126 named channels, with coordinates scaled/oriented differently from the ZuCo montage | Preserve native channels. Do not reuse the ZuCo 105-channel coordinate thresholds or interpolate simply to run ERP analysis. Native region assignments need their own montage verification. |
| Existing spatial raw-waveform helper | Concatenates fixation chunks for a feature representation | Do not copy concatenation into ERP epoching: it removes real intervals between fixations. |

## Bounded checks on existing native files

The files were already present on the Mac. An existing bundled Python/NumPy
runtime read them without downloading data, installing dependencies, creating an
environment or modifying the source files. Only small diagnostic JSON reports
were generated; no EEG arrays were copied or exported.

- 12 native reader files, 165 trials each.
- 165/165 normalized Persian sentence-text joins per reader (1,980 in total).
- All 1,980 trials have reading arrays with the expected 126-channel shape.
  This is a shape check, not full EEG quality validation.
- Sampled at most five fixated words in each of the first four trials per reader:
  **240 first-fixation waveforms, all matched exactly inside `raw_data`**.
  Matching checked every sample of each selected waveform and all 126 channels.
- The native reading field was present in all 15 control trials per reader.
  Presence does not resolve the historical concern about control-trial quality;
  controls stay explicitly flagged for later inclusion decisions.

This establishes a working route for onset recovery. It is not an exhaustive
onset/epoch audit, independent amplitude-unit calibration, verification that the
current Drive copies are identical, or a completed TeCo sentiment analysis.
The 1200-Hz rate is supported by the original notebook constants and project
reference; an acquisition header was not independently recovered here.

## Updated Colab check

`notebooks/teco_preflight.ipynb` reads existing mounted files only. It:

1. Exports explicit Persian word content, original word indices, both sentence ID
   domains, original sentence sentiment and control flags; word sentiment remains
   `pending_Persian_agent_review`.
2. Recognizes native directory layouts and reader-name aliases, records alternate
   copies, and chooses at most one source path per reader.
3. Opens one raw reader file and checks a bounded set of first-fixation onsets,
   using `raw_data` only; it never substitutes `answer_data` or a later fixation.
4. Writes only small reports/inventories and sample onset rows to
   `Thesis/Results/word_sentiment_eeg/teco_preflight/`.

Next work is Persian word/context annotation, then a compact native TeCo ERP
runner with exhaustive onset checks, explicit control handling and a verified
126-channel montage. None of those outputs is claimed complete by this check.
