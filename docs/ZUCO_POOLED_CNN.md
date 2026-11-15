# One model across SR + NR + TSR

[Open in Colab — T4 GPU](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/zuco_pooled_cnn.ipynb?authuser=2)

Choose T4 GPU and Run all using the Zyion account for both mounting and authenticating Drive. The notebook requests a GPU and checks it before setup. Colab controls whether one is allocated.

## Fixed experiment

- Combine positive/negative trials from all three ZuCo tasks. Twelve readers, 36 reader/task recordings. No TeCo and no three-class experiment in this notebook.
- Same original lexical/contextual-word agreement and amplitude/alignment policy. Neutral words are not used for this binary experiment. No requirement for whole-sentence sentiment agreement, no new function-word policy, no new trial matching.
- Global exact-waveform duplicate exclusion; physical trial keys include task because native sentence indices restart across tasks.
- Compact EEGNet-inspired CNN: all 105 channels and 0–792 ms sampled at 125 Hz. At most 25 epochs, validation early stopping with patience four, Adam, class-weighted loss, train-only channel scaling. This is our independent PyTorch model, not the original EEGNet authors' implementation.
- Logistic regression comparison: 105 channel means at 224–304 ms, train-only scaling and fixed regularization C=1 with balanced class weights. Different time inputs mean a comparison of complete pipelines rather than architecture alone. The window comes from prior work and is not a new independent discovery.
- Individual trials and same-class averages of **up to 40** trials. Group separately within each reader/task/class after partitioning the original events. Retain the last undersized group to preserve scarce task/reader data; report actual sizes. This differs from earlier exact-40 analyses that discarded remainders, so scores need not be directly comparable.
- Three separate grouped evaluations, normally three folds: unseen sentences, unseen word identities, unseen readers. Groups are global across tasks. Only the named dimension is excluded; this is not simultaneous generalization to new readers, words and sentences. Word splits purge overlapping training EEG windows using reader/task/sentence identity. Inner validation follows the same grouping; test labels do not select epochs.
- Constant-prior and task-only predictors as controls. Report pooled balanced accuracy, each task's balanced accuracy, the mean of three task scores, reader subgroup scores, confusion matrices and prediction tables. Task-only prediction uses training label proportions for each task and no EEG.

Averaged test examples are constructed using known class labels. Their performance measures class separability after combining responses, not individual-word recognition or an online deployment procedure. Small remainder groups retain more noise than full groups. No new claim of statistical significance is made; scores are descriptive out-of-fold estimates. Reader subgroup scores come from pooled models, not reader-specific classifiers.

## Inputs and time

Reuses `Thesis/Results/word_sentiment_eeg/trial_archive_125hz/` and the completed original `classification_similarity/features/zuco/` metadata caches. The metadata includes saved waveform hashes. No original-source downloads, feature preparation, EEG extraction, filtering, cleaning or relabeling. Checksums and row/event identities are verified before training.

This is the older 40 Hz / 125 Hz first-stored-fixation archive across three tasks. It is not the later SR-only paper reconstruction and has no additional paper nuisance-regression cleaning. Existing single-trial results did not establish a serious CNN advantage; logistic regression is included because trial averaging was more promising previously.

Loading compressed EEG over mounted Drive can take time even with a GPU. Progress shows recording out of 36, shard totals, fold and epoch. The selected binary EEG copy is roughly 0.5 GB and lives only on temporary Colab disk. If that runtime disappears it must be loaded again; it is not re-extracted.

## Persistent storage and resume

Only `Thesis/Results/word_sentiment_eeg/zuco_pooled_cnn/` is new, capped at **100 MB total**, including small model/optimizer snapshots. No existing EEG or results are deleted or rewritten. Most output should be substantially below the cap; the cap stops unexpected growth.

- `protocol.json`, `input_counts.json`: exact inputs, labels, versions and eligible counts.
- `splits/`: original-event IDs for train, CNN fit, validation and test, plus purge reports.
- `cases/<holdout>__max<1-or-40>__<model>/`: predictions, subgroup metrics and fold coverage. CNN subfolders additionally have immutable snapshots at five-epoch intervals and at early stopping/end, with completed fold predictions.
- `scores.csv`, `summary.json`: consolidated results, including actual average sizes.
- `complete.json`: scientific-output checksums, excluding the changing live runtime status.
- `runtime_status.json`: live status from the notebook wrapper; no EEG.

Completed cases/folds are reused. An interrupted fit resumes at its last snapshot; at most four completed unsaved epochs may repeat. A fully completed run verifies outputs before any EEG staging or training. Changed inputs, code or software versions are rejected to avoid mixing incompatible results. No changes to previously prepared notebooks or their scientific implementations.

## Validation

Cloud synthetic tests exercise all three holdouts with identities shared across tasks, task-aware overlap purging, no duplicated/dropped averaging members, OOF counts, baseline and logistic orchestration, actual CNN learning and immutable completed-fold/case reuse. They also construct a saved EEG shard, check row/hash/time/montage validation, and verify temporary-stage reuse without another shard read. Notebook hardware guards are tested with mocked CPU, GPU, TPU and GPU-detection failures. No local numerical dependency installation or EEG training is required.
