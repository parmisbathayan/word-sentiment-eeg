# Published classification recipe on the saved ZuCo SR reconstruction

[CPU notebook, Zyion](https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/zuco_paper_classification.ipynb?authuser=2).

This implements the reported full classification/search recipe, not the authors' exact numerical analysis. The source is our existing SST-labelled, regression-cleaned ZuCo SR reconstruction. It is not the agent subset. Retained trial IDs, timing/preprocessing and some undocumented averaging/splitting details differ from the publication. These differences cannot be removed by simply changing the classifier. No new scientific results are claimed before this notebook runs.

## Reported recipe implemented

-105scalp channels×63samples0,8,...496ms in the0–500ms post-fixation interval;6615features. Saved cleaned waveforms, not extracted band-power features.
- Single,10-,20-,40-trial averages, prepared once, keeping undersized remainder groups. Process40first to target the main reported result. Group within reader and word-level sentiment, favor similar lengths, use different sentence texts within each average. All event membership is saved and globally disjoint.
-100random95/5train/test splits of those examples, permitting reader, word and sentence overlap. No grouping by unseen identities. Each reader remains in train/test where feasible; test words are not forced to be either seen or unseen.
- Training-only standardization and supervised scikit-learn NCA. Nested10-fold training validation selects among NCA dimensions10,20,40,80,160,320,640,1280.
- Logistic regression with L2regularization and RBF-SVM. C spans ten log-spaced values1e−7…1e2; SVMgamma uses the same ten-value grid. No class-weight change. Inner accuracy selects settings; refit on all outer training examples and test once. Fit one NCA transform per inner-fold/component combination, reuse it for all classifier C/gamma candidates. This computational reuse does not move fitting outside training folds.
- Accuracy primary, balanced accuracy and F1/precision/recall secondary; majority-prior dummy comparison. Keep convergence warnings and counts for review.

## Explicit reconstruction choices and limitations

The publication does not supply original random seeds, exact split-rounding and length-averaging memberships, library/solver versions/tolerances or CI formula. Code uses reproducible seed2026, scikit-learn NCA default auto initialization/tolerance and100iterations, fixed documented shuffled inner folds, and pinned modern libraries. Lack of NCA convergence is recorded, not silently reported as full convergence.

For averaging, sort word lengths with random tie order within reader/class; greedily pack up to the requested size with distinct sentence texts and retain all tails. This is an interpretation of “similar word length” and different-sentence examples, not recovered author bins. Actual average counts/remainder sizes are printed and need not equal12,789/1,300/650/328.

For95/5splits, round proportional reader totals with at least one training and test average per reader, then allocate the global negative/positive test margin proportionally across readers. A roughly17-example40-trial test set cannot represent all24reader×class cells. The algorithm allows some cells to be absent rather than change the5%fraction or silently use sentiment-only splitting. This is explicit marginal allocation, not an assertion of exact original joint-stratification code. Actual test fractions use ceiling rounding and are saved per repeat. Odd test sizes can make ordinary dummy accuracy differ from50%; balanced accuracy remains the relevant class-equal baseline.

Unique physical event IDs never cross train/test. Different fixation events may contain overlapping raw EEG intervals; random-split reconstruction does not purge these or hold out sentences. Parent regression cleaning and trial matching used full recordings before classification splitting. Those limitations differ from strict new-reader/new-word generalization. Test labels form same-class averages, so averaged results are not single-word predictions.

The paper reports95%intervals of split means but not its exact CI implementation. We provide explicitly descriptive t intervals across the correlated random split scores, plus ranges and completed-split counts; these are not independent-reader uncertainty or evidence that our interval calculation matches the authors'.

## Compute, storage and resume

This is expensive: all four conditions require3,520,800classifier fits including final refits, plus up to32,800NCA fits after transform sharing. It can require many hours or multiple Colab sessions; no quick completion or particular score is promised. There is no automatic reduced grid or fixed320-component fallback. Official scikit-learn NCA usesCPU, so aT4does not accelerate this implementation.

The notebook creates a temporary cloud-only virtual environment with pinned requirements, avoiding an in-process NumPy upgrade/restart in Colab. No dependency/dataset installation or download occurs on the Mac. Cloud dependency files stay on runtime disk.

New private output: `Thesis/Results/word_sentiment_eeg/zuco_paper_classification/`, capped500MBtotal. It stores reader feature caches, averages/membership, small per-component/inner-fold parameter-score checkpoints, outer split predictions, a consolidated comparison plot/table and checksums. Existing EEG/results stay read-only. The single-trial condition references the reader feature caches rather than duplicating its full matrix. Source NPZs are copied to temporary Colab disk one reader at a time and deleted there after validation. Completed readers survive preparation interruption; completed grid blocks and outer repeats survive training interruption. An unfinished block/refit may repeat. No full NCA model history is written repeatedly to Drive.

A fresh runtime reloads compact features, not EEG once preparation is committed. Completed runs verify checksums before fitting. Different code, scientific libraries or source fingerprints are rejected instead of mixing results. Partial report cells show how many outer splits are complete, not a falsely completed100-split analysis.

## Validation

Cloud synthetic checks cover100deterministic95/5proportional splits including328examples/24joint cells, shared readers, disjoint train/test examples, no lost/duplicated source events, class/reader-pure averages and retained tails, both actual nested classifier grids and NCA, train-only normalization, transform reuse, changed-feature rejection and immutable resume. A synthetic saved EEG archive validates exact0–496ms channel-major extraction, cache reuse without EEG reads, and average-cache reload. Tests use tiny dimensions/grids for engineering validation; the real notebook retains full published search defaults.

Source: [Pfeiffer et al.(2020), methods2.4.5and Table1](https://doi.org/10.1016/j.neuroimage.2020.116934).
