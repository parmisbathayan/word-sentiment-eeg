# One saved-result plotting notebook per dataset

- [ZuCo: all plots](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/zuco_plots.ipynb)
- [TeCo: all plots](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/teco_plots.ipynb)

CPU, Runtime → Run all, allow Drive mounting. These notebooks do not run
extraction or load source EEG. They read saved per-reader all-channel class
averages, verify hashes/counts/axes and produce one gallery for each dataset.
The ZuCo notebook creates a missing saved-file audit first; an existing audit
is reused and source-average checksums are checked again.

Each notebook shows group overall plots, group regional plots, the sensor map,
and every reader's combined overall/regional plot. ZuCo keeps SR, NR and TSR
separate. TeCo shows pooled neutral, strict neutral-context and source-rated
neutral comparisons with separate matched-reader rosters. All panels within a
comparison use the same readers and equal reader weights; counts are not used
to give a reader more weight. Subject figures show each available group with
its accepted-epoch count, without artificial trial-level uncertainty bands.

The overall panels distinguish three meanings:

1. **Whole-scalp signed mean:** arithmetic channel average. Average referencing
   makes this approximately zero, so a flat curve is expected. Its y-scale uses
   signed regional signals as a reference instead of enlarging rounding noise.
2. **Original ZuCo 12-channel ROI:** the signed overview used by the previous
   ZuCo plots. This is a selected scalp region, not the entire head.
3. **GFP:** population standard deviation across channels of each reader's
   class-average waveform, then averaged across readers. This nonnegative
   summary is separate from signed ERP voltage. It is not single-trial GFP or
   GFP of the grand average. Opposing reader topographies can cancel in a
   grand average without canceling this reader-level GFP summary.

The GFP definition follows the [MNE ERP tutorial](https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html#global-field-power-gfp).
Class-average GFP can vary with unequal trial counts and residual noise;
these plots do not establish significant or sentiment-specific differences.
The six geometric sensor groups do not localize cortical sources. The existing
fixation-response overlap, labels, preprocessing and native-unit limitations
remain unchanged.

Output paths:

```
Thesis/Results/word_sentiment_eeg/
├── v4_word_erp_compact/analysis/all_plots_v1/
└── teco_v1_word_erp_compact/analysis/all_plots_v1/
```

Each gallery contains `group/`, `subjects/`, `sensor_regions.png`,
`group_curves.csv.gz`, `reader_class_counts.csv`, `channel_regions.csv`,
`comparison_rosters.json`, `gallery.json`, `protocol.json` and `REPORT.md`.
Expected extra storage is a few to a few tens of MB per dataset, depending on
image compression. Reruns replace generated gallery outputs, without appending
readers/events or changing extraction checkpoints. Original compact ZIPs and
original plots are not rewritten. No trial arrays or additional source EEG
copies are saved. The separate ZuCo context-control reconstruction is not
executed by these plotting notebooks.

The Python entry points are `scripts/plot_zuco.py` and `scripts/plot_teco.py`,
each taking `--run-dir` for the corresponding saved result folder. Both use
`scripts/plot_saved_erps.py` so their averaging and reader rules remain consistent.
