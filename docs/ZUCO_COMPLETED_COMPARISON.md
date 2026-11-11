# Completed ZuCo reconstruction reviewed — 2026-11-11

Source: `Thesis/Results/word_sentiment_eeg/zuco_paper_figures_staged/` on Drive.
The root and analysis manifests report completion, with all 12 readers and saved
statistics/figures. This is a comparison of an independent Python reconstruction,
not evidence that the original authors' analysis is wrong.

| Finding | Published paper | This completed reconstruction |
|---|---|---|
| Accepted fixation trials | 12,789 | 6,634 (3,314 ranked positive/negative pairs) |
| Original source blocks | Full paper source processing | 54 of 96 included; 37 fail the configured clock/alignment checks and 5 lack source pairs |
| Significant corrected scalp clusters | Left-central and right-posterior effects around 224–304 ms after cleaning | None; minimum corrected cluster p=0.3866133866 |
| Overall field strength (GFP) | No significant valence effect | No sustained difference visible in the saved GFP significance strips |
| Frequency control | Authors report no confounding linguistic difference | Saved control figure reports reader-paired fixation-weighted word-frequency p=1.05e−7; this is a different test from the paper’s unique-word comparison |

The raw and cleaned positive-minus-negative maps look very similar, while the
common fixation response decreases after cleaning. Their similarity alone does
not establish a bug; shared nuisance responses can cancel in a contrast. Exact
TCT intervals have not been summarized from the saved arrays in this review.
Do not identify our TCT result as the paper's 224–304-ms finding from appearance.
The scalp panels at 224–280/304 ms use predeclared reference windows: a colored
map does not mean that window is significant in our results.

The paper's main topographic result was **not reproduced**. The plots can be
compared descriptively, with the missing blocks, smaller retained sample,
fixation-weighted frequency imbalance, non-equivalent frequency control test, Python implementation and preprocessing departures
reported. Completion means the configured analysis finished, not that the
published methodology or findings were fully reproduced. No new ZuCo extraction
is needed merely to view these results.

Primary source: Pfeiffer et al. (2020), NeuroImage 218, 116934,
https://doi.org/10.1016/j.neuroimage.2020.116934.

Reviewed private outputs: `analysis/analysis_manifest.json`, `clusters.json`,
`source_coverage.csv`, `exclusion_counts.csv` and saved raw, cleaned and control
figures. No source EEG or model arrays were downloaded to the Mac for this review.

Follow-up timing audit: all36 numeric residual failures have only1–3 trigger anchors exceeding8ms (among98–102). This suggests testing local recovery rather than assuming whole blocks are unusable. No recovery has yet been validated. See paper_clock_recovery_audit.json.
