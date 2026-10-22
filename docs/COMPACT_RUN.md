# Compact ERP run after the Drive quota limit

Open `notebooks/compact.ipynb` in Colab, use CPU, and Run all. The setup cell
checks the mounted Drive before installation/extraction. Less than 1 GiB free
stops the run. Deleting files may not immediately change the mounted quota;
previous DriveFS uploads can still be pending. No old data is deleted by this
runner. The original full-trial notebook is retained as `full_trials.ipynb`.

`run_compact_erp.py` imports the unchanged scientific processing functions. It
calculates each accepted trial but accumulates class sums without buffering or
writing individual EEG trials. This changes storage, not event selection or
the ERP definition. Existing compatible onset tables/means are validated and
copied into one new results folder; missing analyses are extracted from source.
Copied files are checked by SHA-256. A known source hash is verified before
rebuilding a compatible but incomplete analysis. Each reader/task appears once.

The self-contained output is
`Thesis/Results/word_sentiment_eeg/v4_word_erp_compact/`. It includes:

- `readers/<task>/<reader>/`: onset tables, class means/counts, source hashes.
- Grand averages, plots, class counts, annotation CSV, channel montage and
  extraction provenance.
- `compact_results.zip`: a portable copy of these small results, without any
  old trial archives, download caches, logs, or progress files.

The expected additional footprint is about 100–200 MB including the ZIP, based
on existing per-reader means/onset sizes and grand averages. This is an estimate,
not a promise about available Drive space. Old cached trials remain untouched
and continue occupying space until separately cleaned up.

Compact completion means all requested onset tables and class averages are
present and validated. It does not claim individual EEG trials were archived.
Saved onset samples, source checksums and processing code support regenerating
those waveforms when needed. Full-trial extraction remains available separately.

Cloud integration checks compare freshly extracted compact/full ERP means and
every onset row, test compact reuse without source files, confirm no individual
trial files are written, and verify the original cache files remain unchanged.
