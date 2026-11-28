# Cross-dataset EEG, PCA and VAE RSA — task 1

Run `notebooks/cross_dataset_rsa_vae.ipynb` in Colab with T4 GPU and the Zyion
Drive account. One notebook runs the complete planned task-1 RSA reference
analysis. This does not rerun classification or semantic-aspect grouping.

## Inputs and scope

- TeCo task 1 and ZuCo SR, 12 cached readers each. Tasks 2/3 are not included.
- Reuse `trial_archive_125hz` metadata and `classification_similarity/features`.
  Read no EEG shards, native files, waveforms or pickle archives; extract nothing.
- Original shared 40 Hz/125 Hz feature preparation, NOT the different ZuCo
  paper-regression-cleaned cohort that produced the earlier 63.9% classifier.
- Frozen original lexical/context-agreement sentiment rosters supply labels.
  Additional targets reuse verified saved automatic NLP annotations.
- Targets: positive/negative, neutral/nonneutral, noun/verb/adjective,
  content/function and recorded first-fixation duration below/at least 200 ms.
  Missing auxiliary label files are listed explicitly; missing core binary labels
  or changed checksums stop the run. No new NLP downloads or silent relabelling.

## Fixed experiments

Four compatible inputs: full 48 regional/time bins over 0–800 ms; the 12 bins
covering 200–400 ms; 18 regional mean/SD/range summaries; normalized full bins.
The caches cannot recover exact 224–304 ms. Do not label 200–400 ms as that window.
These features do not provide dense electrode voltage maps.

Three joint held-out-reader folds hold out four readers from each dataset per
fold. Training/validation/evaluation readers are disjoint. Choose one additional
validation reader from each training-side dataset. A separate all-reader fit
includes evaluation data and is explicitly descriptive; its internal validation
overlaps fitting data and is not evidence of generalization.

The same unsupervised model serves every target for an input/fold: labels do not
select training events, scales, dimensions or parameters. Use a balanced sample
of up to 12,000 fitting events per dataset for preprocessing/PCA, all fitting
events with equal dataset loss weights for VAE, and up to 3,000 validation events
per dataset for early stopping. All-data VAE training includes all valid cached
events from both datasets, not only sentiment-qualified events.

Train-only scalar RMS equalization per dataset and one shared imputer/scaler
produce compatible numerical ranges. This compares standardized patterns; it
cannot validate native TeCo microvolt units or compare absolute voltages.

Representations: original standardized features, up-to-eight-component PCA,
VAE posterior means averaged AFTER encoding, and encoding the EEG-feature
average. Use one shared VAE per fold/input: input→32 Tanh→8 latent dimensions
with symmetric decoder, Adam 0.001, KL weight 0.05, at most 60 epochs, patience 8.
Posterior means are deterministic; sampled latent vectors are not evaluated.
PCA/VAE similarity uses cosine, invariant to a shared orthogonal rotation.
EEG and pre-onset use pattern Pearson correlation. Clearly identify the metric.

For each target: exact 40 known-same-class events per average, within
reader/task/class, after splitting. Use equal numbers of groups for all classes
of one reader/task, capped at four per class; drop leftovers. Readers unable to
provide 40 events in EACH class are excluded from that target/partition and
listed. No smaller-group or Max40 substitution. Three predetermined regroupings
measure grouping sensitivity; regrouping 0 is the declared primary analysis.
This is label-informed pseudotrial RSA, not individual-trial decoding.

## Matrices, controls and inference

Compute TeCo–TeCo, ZuCo–ZuCo and rectangular TeCo–ZuCo matrices. Compare
dissimilarity (1−similarity) with the ideal 0 same-class / 1 different-class RDM
using Spearman rho. Do not square rho. Positive rho indicates greater
between-class dissimilarity. Also report same/different similarity means and
their difference. Dense cells are not independent samples or independent trials.

Within-dataset inference uses one triangle, no diagonal or same-reader pairs.
Cross-dataset entries join distinct readers by definition. Stimuli can repeat
across readers: report the fraction of evaluated pairs sharing a source word
occurrence. The same recorded EEG cannot cross reader-heldout boundaries.

Primary randomization: shuffle the WHOLE set of class names independently for
each reader, consistently across all that reader's groups. Binary targets are
whole-reader label swaps. Reuse each reader's permutation across the joint fold
aggregation. Use 999 null draws and a one-sided positive-structure test. Average
the three held-out fold rho values, then compare that mean against jointly
averaged null values. Require all three folds for a summary p value.

This is an exploratory, reader-conditional class-label-swap test. It assumes
reader contrast symmetry/class exchangeability under the null, conditional on
these stimuli and label-built groups. It is NOT a stimulus-random permutation
test: it neither permutes independent matrix cells nor establishes transfer to
unseen words/sentences or a language population. Word-property differences can
also produce a consistent reader contrast. Do not call results a reproduced
paper TCT, TANOVA, cluster test, or a pure sentiment mechanism.

Correct all available primary held-out tests jointly across inputs, targets,
representations and matrix types. Export both Holm FWER p and BH-FDR q and the
family size. All-data matrices and regroupings 1/2 are descriptive/sensitivity
analyses, not additional opportunities to choose a successful p value.

Controls: pre-onset 12-bin EEG on matching groups, plus four-property RDM
diagnostics (word length, relative position, sentence length, control flag).
Report property-model rho, EEG/property RDM rho and a partial-rank diagnostic.
This partial comparison removes only the modeled aggregate property-distance
relationship, not every possible confound. It is not the superseded duplicated
reader-row non-EEG classifier and has no classifier accuracy/p-value claim.
Missing control flags can become a constant imputed predictor.

Reliability: within-reader/class correlation between disjoint averages, where
at least two groups exist; otherwise unavailable. VAE diagnostics: reconstruction
MSE, zero-prediction MSE, active latent dimensions, latent variation and descriptive
dataset/reader variance fractions. Good reconstruction alone is not semantic
evidence; embeddings may predominantly preserve sessions or equipment.

## Storage, outputs and resume

One separate folder: `Results/word_sentiment_eeg/cross_dataset_rsa_task1`.
Hard total output cap: **50,000,000 bytes (50 MB)**. No new full feature/EEG copies,
permutation map banks or automatic deletions. Save compact models, optimizer/RNG
epoch progress, exact event memberships, scores, diagnostics and representative
full-input matrices (first held-out fold and all-data, primary grouping only).
Other combinations retain scores and exact deterministic reconstruction details.
Progress files are overwritten in place, avoiding Drive rename/Trash buildup.
Completed models and cases are checksum-verified and skipped on rerun.

Main outputs:
- `heldout_summary.csv`: mean three-fold rho, conditional p, Holm p and BH q.
- `results.csv`: every fold/all-data/input/target/representation/regrouping matrix.
- `unavailable.csv`, `missing_targets.json`: coverage and missing-label limits.
- `class_counts.csv`, `label_examples.csv`, `groups/`: exact data coverage/groups.
- `vae_diagnostics.json`, `models/`: model diagnostics and resumable checkpoints.
- `plots/`, `overview.png`: measured/ideal matrices and held-out effect overview.

Future expansion: repeat the full analysis with ZuCo SR+NR+TSR and TeCo tasks
1+2+3 after native data preparation and contextual sentiment annotation. This
requires explicitly extending/validating the loader; the task-1 notebook does
not silently pool extra tasks it finds. Supervised fine-tuning is also separate.

## Sources

- RSA and model RDM comparison: Kriegeskorte et al. (2008),
  https://www.mrc-cbu.cam.ac.uk/wp-content/uploads/www/sites/3/2013/01/kriegeskorte_RSA_frontiersSN2008.pdf
- VAE objective: Kingma & Welling, https://arxiv.org/abs/1312.6114
- PyTorch reproducibility limitations:
  https://docs.pytorch.org/docs/stable/notes/randomness.html

These support the methods, not the specific adequacy or findings of this run.
