# One fixed trial-averaging sweep

[Open in Colab — CPU, Zyion](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/zuco_averaging_sweep.ipynb?authuser=2). Let the current pooled GPU experiment finish before changing runtime or starting another reader-archive load.

This short follow-up changes only the number of same-class trials averaged. One setup: ZuCo SR, completed agent-labelled paper subset, cleaned EEG, 105 channel means at224–304ms, logistic regression C1/class-balanced/liblinear, three fixed held-out-reader folds. Scaling and missing-value imputation fit training data only. This choice follows the most promising earlier averaged result, making the analysis exploratory. No CNN sweep or other dataset/split/window search.

Sizes1,10,20,...,100. After each fold is formed, create disjoint exact-size averages separately within reader and class. Discard leftover events. Five deterministic grouping repeats at sizes10–100, one at size1. The same shuffled ordering within a repeat is used across sizes; sizes divide that ordering differently. Repeats are correlated regroupings, not independent experimental samples. Report mean/range/SD of balanced accuracy and dummy BA, with number of test examples and original events used. The range is not a confidence interval or statistical significance test. No best-seed reporting.

Use only readers with at least100saved events in both classes at every point, based on counts before averaging. This prevents reader composition from changing with averaging size. Print counts/exclusions; never choose readers based on scores. At least three readers are required; report insufficient coverage rather than silently changing the rule. The common cohort and new groupings mean the40-trial point may differ from the prior63.9% figure.

Readers are unseen, but sentences and word identities may overlap between training and testing. Known test labels construct test averages, so this measures class separability after averaging, not predictions for individual unknown words. As before, regression cleaning used complete saved recordings, not training-only fitted nuisance models. The agent subset intersects a parent SST-selected/matched/rejected archive and includes repeated fixations; it is not all agent-labelled words and not first fixation only. No original source extraction, relabeling, rematching or preprocessing is added.

## Storage and inputs

Read-only sources: `Thesis/Results/word_sentiment_eeg/zuco_agent_paper_subset/selection/trial_selection.csv.gz` plus `zuco_paper_figures_recovered/readers/<reader>/prepared_trials.npz` and parent checkpoints. The selection audit retains the parent sentiment column: the new script explicitly restores the retained **agent lexical_class** as the target, checking agreement with contextual_class, as the original subset runner did in memory. It separately verifies parent EEG labels against parent_sentiment. No label inference is performed.

First run verifies and loads one saved reader archive at a time on temporary Colab disk to compute the105window features. This may take several minutes despite no training bottleneck. The new feature cache is a few MB and contains only these105features plus event IDs; it supplements the existing EEG archive, which is not removed or replaced. Once cached, reruns do not reload EEG. Interrupted first-time cache construction may reload readers before the final feature cache has been committed.

All new persistent outputs share a **20 MB** cap in `Thesis/Results/word_sentiment_eeg/zuco_averaging_sweep/`:

- `features/`: compressed features, IDs and source/checksum manifest.
- `protocol.json`, `cohort.json`, `splits.json`: fixed settings, count-based cohort, original train/test event IDs.
- `cases/`: completed repeat scores and compressed predictions, including exact test-example member event IDs and per-fold counts/digests.
- `scores.csv`, `repeat_scores.csv`, `averaging_curve.png`: results and curve, with fewer test examples at larger sizes.
- `complete.json`, `runtime_status.json`: checked completion and live notebook status. The changing runtime status is excluded from scientific completion checksums.

Completed fits/results are reused. No changes to the running pooled notebook or any previous data/results. Colab-only code/dependency installation and training; no numerical dependency installation or EEG training on the Mac.
