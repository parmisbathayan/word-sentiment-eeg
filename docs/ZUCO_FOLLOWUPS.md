# What can be learned beyond class-average plots

The first completed result contains accepted-word onset tables and class means
for every reader and all 105 retained channels. It does not retain individual
word epochs in compact mode. That distinction determines which questions are
quick and which require rebuilding a richer result.

## Existing averages: reader consistency and effect estimates

`scripts/analyze_zuco_followups.py --run-dir <existing-result-folder>` reuses the
saved per-reader class arrays and the existing geometric scalp partition. It
tests all three within-reader contrasts: negative minus positive, negative minus
neutral, and positive minus neutral. Windows are 0–200, 200–300, 300–500 and
500–800 ms, with half-open intervals, in all six regions and three tasks.

The secondary family therefore has 216 comparisons. A separate primary
exploratory family uses the original 12-channel ROI in `erp.ROI`, the two existing
300–500 and 500–800 ms windows, all three contrasts and all three tasks: 18
comparisons with its own Holm correction. That ROI and those windows are fixed
before viewing the paired-reader test outcomes. They were already used in the
original saved-result review; calling this family primary does not make it a
preregistered or confirmatory analysis. We do not claim joint familywise error
control across both the primary and secondary families.

Each reader contributes one
paired difference per comparison, irrespective of their number of accepted
trials. Outputs contain every reader difference, mean/median/SD/SEM, paired dz,
direction counts and the range of means after leaving out one reader. That
range is a sensitivity description, not a confidence interval.

The two-sided exact sign-flip test enumerates all 2^n assignments and uses the
absolute paired mean. It requires independent readers and sign exchangeability
(symmetry of paired differences under the null). Word classes were not randomly
assigned. Holm correction controls familywise error across all 216 comparisons
without requiring independence between the tests. All comparisons are reported;
the code does not choose a best region/window after seeing the data.

**Resolution limit:** with 12 readers the smallest possible two-sided exact raw
p value is 2/4096 = 0.0004883. The first Holm threshold is 0.05/216 = 0.0002315.
Consequently this particular broad family cannot produce a rejection with 12
readers, even if every reader's difference points the same way. The report states
this explicitly. Its value is estimating sizes and consistency, rather than
providing a route to claim significance. The primary family has a feasible
first Holm threshold of 0.05/18, although power remains limited. We should not
switch corrections or narrow either family after inspecting test outcomes.

The estimates also include a reproducible 10,000-resample reader-bootstrap
percentile 95% interval. It is pointwise and conditional on the fixed word
collection, not adjusted for multiplicity, and cannot establish significance
across the family. The primary reader-dot plots display these intervals.

Reader dot plots make the practical question visible: is a difference shared
across readers, small relative to their variability, or strongly influenced by
one reader? The result remains exploratory because descriptive plots have
already been inspected. It conditions on this fixed set of annotated sentences
and does not model uncertainty across new words/items.

All outputs are inside `analysis/paired_reader_contrasts/`. Original files are
read-only inputs; their checksums are recorded in the analysis protocol. The
source EEG is not loaded, and words/exclusions/preprocessing remain unchanged.

## Neutral words in neutral sentences

This is a useful additional control, but not something the pooled means can
recover. Sentence-level affect must first be annotated across the complete
corpus, separately from target-word polarity. Existing SR review labels may be
retained as source metadata; they should not substitute for an explicit policy
across SR/NR/TSR. Uncertain/mixed sentences need a separate category.

During rebuilding, preserve the original analysis and save word-label by
sentence-context strata separately. Include neutral target words in neutral
sentences as one baseline. Also compare target classes within the same context
strata where coverage permits. For example, compare positive and neutral target
words within neutral sentences if both have enough examples. Comparing only
positive words in positive sentences with neutral words in neutral sentences
changes two things at once: the word and its sentence context. It cannot tell
which caused an EEG difference.

Do not require the whole sentence to be neutral for every analysis without
checking what this removes. A sentence containing an affective word may itself
be affective; strict filtering could leave few valenced targets. Report the
word-by-context counts before extracting new subsets.

## Next source-epoch analysis: compact word features and confounds

Before rebuilding any EEG, `scripts/summarize_zuco_covariates.py --run-dir
<existing-result-folder>` can already quantify several potential confounds
using the saved onset tables. It writes class counts, distinct source-item
counts, acceptance/exclusion balance, word length, sentence position/length,
first-fixation duration and next-located-fixation timing per reader/class.
Word-level EEG voltages, validated POS and external word frequency are absent.
These are descriptive balance checks rather than adjusted ERP estimates.

The next-located-fixation field includes only successfully recovered fixations
within the sentence. Fractions with a located next fixation before 200/500/800
ms are observed lower bounds on overlap opportunities. A missing or late next
located fixation does not establish that the epoch was free of other eye
movements or overlapping responses. The reports preserve that distinction.

Alongside rebuilding strata, save one row per accepted fixation/word with
reader, task, sentence/item identity, both word-label layers, context label,
onset and duration, exclusion reason and regional/window mean voltages. Add
word length, frequency with a named lexicon, part of speech, sentence position
and preceding-fixation/saccade timing where recoverable. This compact table is
much smaller than exporting every EEG epoch and supports later models without
re-extracting EEG.

The neutral class contains many function words while affective targets often
are content words. A content-word neutral control and matching or regression
for length/frequency/part of speech/fixation behavior are higher-priority than
training a classifier on the present class means. Subject/item mixed-effects
models can then address repeated readers and repeated words/sentences. Pooled
reader averages cannot reconstruct item-level variance.

## Overlap and interpretation

Natural reading fixations are often closer together than an 800 ms epoch.
Activity aligned to one word can include responses to preceding/following
fixations and eye movements. Deconvolution/regression on continuous EEG with
fixation/saccade timing is required for stronger isolated-response claims;
class means cannot undo overlap. Scalp-region differences do not localize a
cortical generator. A 300–500 ms difference is not itself a sentiment-specific
N400 finding.

Relevant primary methodological sources:

- Ehinger & Dimigen (2019), regression-based EEG deconvolution and covariates:
  https://doi.org/10.7717/peerj.7838
- Dimigen & Ehinger (2021), regression-based analysis of combined EEG and eye
  tracking: https://doi.org/10.1167/jov.21.1.3
- Westfall, Kenny & Judd (2014), power and generalization with participants and
  stimuli: https://doi.org/10.1037/xge0000014
- Holm (1979), sequentially rejective multiple-test correction:
  https://www.jstor.org/stable/4615733
