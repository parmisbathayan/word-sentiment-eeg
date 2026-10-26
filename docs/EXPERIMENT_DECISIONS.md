# Decisions and presentation notes — 2026-10-26

These are versioned methods decisions, not evidence that a new reconstruction has run.

## Labels and populations

Each word occurrence has lexical polarity, contextual word polarity (interpreted using the surrounding text, including negation/irony), and agent-reviewed sentence-context polarity. Sentence context is not automatically the source movie-review rating. Local context is not restricted to an arbitrary fixed-length phrase.

The first new experiment requires all three labels to agree. The three follow-ups select the class using only lexical, contextual-word, or sentence polarity respectively. Their populations overlap: never sum them as independent trials. Keep positive/negative/neutral and do not downsample neutral.

Primary new analyses use content words. Function and uncertain grammatical labels remain in the annotation table and are reported as exclusions, not deleted. Grammar is an operational agent/screen label, not validated POS gold. The English revision inherits existing lexical/contextual sentiment labels; it is not a fresh complete independent sentiment reannotation. Persian revision reread all165 sentences and3016 positions. All annotations are agent annotations, not human-consensus gold.

All-three content-only source counts: ZuCo SR positive274/negative291/neutral13; NR45/48/2207; TSR86/53/2503. TeCo116/77/11. These are source word occurrences, not accepted reader epochs. Sparse strict-neutral coverage must be reported without forced balancing. The user flagged 13 neutral word occurrences as potentially too few on 2026-10-26. Reconsider all-three agreement as the primary comparison before a new long run. Use “word occurrences” in presentation prose.

TeCo's15 CONTROL sentences are included for reading EEG in the new adapter. Never use answer_data for these FRPs. The historical TeCo run excluded controls; do not relabel its results as control-inclusive. Current native adapter covers task1 only; task2/3 EEG access remains to establish.

## Signal and exclusions

Use actual voltage waveforms, not time-collapsed frequency-band features. Locate the first stored word-fixation waveform in sentence reading EEG; never replace an invalid first slot with a later fixation. First stored is the documented export choice, not an independent certification of the first physical gaze event.

Missing/invalid first fixation can mean missing/nonmatrix/nonfinite waveform. Onset recovery separately rejects too-short/flat/too-long segments, nonunique matches and unmatched waveforms. Sentence EEG availability, nonfinite raw data, filter length, epoch boundaries, nonfinite epochs and amplitude are separate reasons. Preserve exact terminal reason per word event; count each once per experiment. Do not interpret export/alignment failure as proven eye-tracker noise.

Historical reconstruction: average reference, fourth-order zero-phase Butterworth30Hz low-pass of whole sentence, epoch−200…+800ms inclusive, baseline−200…0ms excluding0; reject epochs exceeding absolute100µV (TeCo native units assumedµV). ZuCo500Hz; TeCo documented1200Hz. New reconstructions use40Hz from source sentence EEG; old30Hz class means cannot recover30–40Hz information. All-channel signed means approach zero under average reference; regional curves/GFP are needed for interpretation.

ZuCo upstream publication describes word-bound assignment and short-fixation removal; exact upstream counts are unavailable in current word exports. TeCo allFixations includes short examples, but survival into word EEG is unverified. Do not claim blink-predecessor/outside-word/<100ms counts are zero, or add an unverified duration threshold. Do not blanket-exclude negation: contextual labels handle many reversals; a separately flagged sensitivity analysis can test them.

Old completed run accepted totals: ZuCo129008 across36reader/task recordings; TeCo18723 across12task1 readers. They are historical30Hz/old-policy totals, not new40Hz totals. Save source positions, sentence counts, theoretical word-reader pairs, actual observed unique pairs/presentations, fixation availability, accepted trials and reader rosters separately. TSR repeated presentations are observations, not new unique words.

## Reporting and next experiments

Provide per-reader/class/task counts, exclusive terminal exclusion totals, unknown upstream checks, code/label/source hashes, montage and filter settings, baseline/epoch settings, amplitude units, class comparison rosters and missing readers. Preserve old runs. New checkpoints validate content hashes and event identities; never add a reader twice. Overlapping neutral subsets are not separate trials.

Classification remains future work. Evaluate unseen-word/sentence and reader-held-out generalization separately from a diagnostic same-word/same-reader split. In the diagnostic, different occurrences can be in train/test, but never the same EEG event; correlated repeats and word identity can inflate apparent sentiment performance. Also examine within-reader repeated-word waveform/feature similarity, controlling occurrence context and overlap. Regional window features are compact inputs, not full trial time courses. No classifier result is claimed here.

Paper comparison and verified reusable toolbox links: PAPER_METHODS_COMPARISON.md. Its covariate-balanced sampling is different from our waveform onset matching. Its exclusion rules are not automatically appropriate for this project.
