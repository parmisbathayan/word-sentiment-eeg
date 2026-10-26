# FRP paper comparison and evidence audit

Checked 2026-10-26. This is a methods review, not a report of a new extraction.
No source EEG, package, model, runtime or environment was downloaded or installed
on the Mac for this review.

## Published study and code availability

[Pfeiffer et al. (2020), NeuroImage 218, 116934](https://doi.org/10.1016/j.neuroimage.2020.116934),
sections 2.4.2–2.4.5, describes SR-only word-linked fixation FRPs, not a
first-fixation-per-word restriction. Its exclusions include out-of-word events,
fixations under 100 ms, neutral words, nearby negation and excessive voltage.
Negation meant five English particles within the preceding three words.
“Matching” means stratified trial sampling to balance classes and linguistic/
oculomotor covariates, including word length and neighboring fixation timing;
it does not mean finding a waveform inside sentence EEG. The study additionally
uses regression deconvolution for overlapping fixation/saccade/blink/keypress
responses. Its main preprocessing includes a 1–50 Hz bandpass, resampling to
125 Hz, −600…1000 ms epochs and ±90 µV rejection. Of 13,027 matched epochs,
238 failed amplitude screening, leaving 12,789. The prose assigns positive
6,379 / negative 6,410; the supplied figure reverses these class labels.
The total agrees. Do not silently resolve that inconsistency.

I did not verify a public repository containing this paper's complete custom
analysis pipeline. The article, targeted title/author/code searches, public
`methlabUZH` repositories and public `cp3fr` repositories were checked. This is
an unsuccessful availability check, not proof that the code does not exist.
Do not represent toolbox availability as availability of the entire study.

The following public tools are reusable subject to their own licenses and
dependencies; GitHub repository metadata was checked read-only:

| Tool | Verified URL | Current repository license | Role |
| --- | --- | --- | --- |
| Automagic | <https://github.com/methlabUZH/automagic> | GPL-3.0 | EEG preprocessing |
| unfold (MATLAB) | <https://github.com/unfoldtoolbox/unfold> | GPL-3.0 | Overlap/deconvolution modeling |
| FieldTrip | <https://github.com/fieldtrip/fieldtrip> | GPL-3.0 | EEG analysis, including permutation testing |

The author's [generic cluster-permutation helper](https://github.com/cp3fr/Cluster_Permutation_Test)
is public, but GitHub did not identify a license. It was not established to be
the paper's study pipeline. No source was copied into this project.
If an exact study replication becomes necessary, the custom scripts and pinned
versions would need to be obtained from a verifiable release or the authors.

## What is already removed upstream?

The [original ZuCo data paper](https://www.nature.com/articles/sdata2018291),
eye-tracking preprocessing section, says engineered word data use within-word
fixations and exclude durations below 100 ms; raw eye tracking remains available.
Word boundaries were expanded horizontally, and line assignment was corrected.
Thus “within word bounds” follows the corpus's operational mapping, not a
guarantee of perfect gaze localization. The paper also documents upstream
ocular-artifact cleaning and ±90 µV EEG-feature rejection. Those are separate
from our added epoch checks. Its validation FRPs used first-pass fixations,
which is not identical to first fixation per word. These statements support
provenance, not a fresh audit of every stored event in our exact input copies.

For the current project, neither `src/word_sentiment_eeg/erp.py` nor
`src/word_sentiment_eeg/teco_erp.py` implements an explicit blink-predecessor or
100-ms threshold. Their absence is not evidence that upstream removal occurred.
The next source-based run should export and check actual first-segment lengths
and available fixation-duration metadata, then distinguish inherited removals
from removals applied by our code.

TeCo evidence is more limited:

- Existing `Thesis-Load & Explore data.ipynb`, cell 69, displays `allFixations`
  fields `duration`, `x`, `y`, `pupilSize`.
- Its saved cell 70 output contains durations **12 ms and 74 ms**. Therefore
  the native all-fixation table was not universally filtered to >=100 ms.
- This does **not** prove that those short fixations remain in each word's
  `rawEEG` list. That requires auditing word segments, source fixation indices
  and timing rather than equating the all-fixation table with word candidates.
- Available word `rawET` contains time/gaze/pupil samples; the current project
  has not established a validated full blink-event timeline from them.
- `fixPositions` are fixation-order indices, not EEG sample onsets. Alignment
  of these indices to `allFixations` must be checked before using predecessor
  information.

The existing notebook is at
`/Users/parmis/Documents/ClaudeCode/yaser_code_fixing/Thesis-Load & Explore data.ipynb`.
Native schema reference:
`/Users/parmis/Documents/thesis_data/TECO_DATA_STRUCTURE.md`.
No full native reader was loaded for this methods review.

## Recommended reporting and selection decisions

Our starting denominator is reader × source-word occurrence, not all eye events.
An exclusion figure should begin there and explicitly state that outside-word
events are absent by design. It must not claim a measured count of upstream
blink/out-of-word/<100-ms removals when only the postprocessed word export is
available. Report such quantities as “upstream; count unavailable.”

Use sequential, mutually exclusive final statuses so counts reconcile:

1. Inventory source sentences/tokens, and actual reader-token availability.
2. Apply the chosen sentiment selection policy and explicit linguistic flags.
3. Select the first stored fixation only; retain missing first slots.
4. Validate sentence signal and first waveform; recover a unique onset.
5. Check full epoch boundaries and source filtering requirements.
6. Baseline-correct and screen finite values and peak absolute amplitude.
7. Count accepted epochs by class, reader and task; export unique event IDs.

Keep independent boolean failure flags too, but never add their potentially
overlapping counts as if they were a sequential flow. Attach the exact input
hashes, filter settings, epoch window, baseline, inclusion policy and code commit.

User-requested future reconstruction uses **40 Hz** low-pass. This requires
source EEG, a new processing fingerprint and separately versioned outputs;
existing 30-Hz means cannot reconstruct the removed 30–40-Hz activity. Preserve
the completed results as 30-Hz results.

Do not drop neutral or downsample the larger class in the main run. Preserve
all eligible trials and report unequal counts. Trial matching, duration strata,
stopword/content-word subsets and a negation-excluded subset can be separate
sensitivity analyses with explicit attrition. A blanket negation rule should
not substitute for contextual annotation. Keep negation scope as a flag even
when the main selection accepts the contextual interpretation.

A word-linked first fixation can still follow a blink and its epoch can still
overlap another fixation or motor response. Being word-linked addresses spatial
assignment; it does not remove these temporal/physiological concerns. Estimate
their availability first. A blink check needs verified predecessor events, not
an invented count or zero-valued pupil heuristic without validation.

Our waveform-onset matching failures should be labeled **“onset recovery failed
or ambiguous”**, to distinguish them from the paper's covariate-balancing
matching exclusions.

## Other TeCo tasks: bounded local inventory

The existing local tree contains Task 2 sentence workbooks
`TeCo/sentences/Task_2/total_nr_1.xlsx` through `total_nr_3.xlsx`, plus
`total_question.xlsx`; Task 3 has six `total_tsr_ra_*.xlsx` workbooks.
Native EEG files discovered under `/Users/parmis/Documents/thesis_data` are
`TeCo/task1_with_total/<reader>/task1.pickle` and corresponding `.mat` files.
No native Task 2/3 `.pickle` or `.mat` was found in that tree in this review.
This does not establish whether those EEG files exist elsewhere or on Drive.
Do not claim that only Task 1 exists, or that Task 2/3 text is unavailable.

Before extending extraction, audit the actual Task 2/3 input location and schema,
then label their source texts and keep task identity explicit. This review did
not perform a new Drive-wide search or download missing task files.
