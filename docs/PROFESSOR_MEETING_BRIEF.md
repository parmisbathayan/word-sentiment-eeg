# Professor meeting brief — completed runs and deferred work

Prepared 2026-10-26. Present the completed historical30Hz analyses. New40Hz/three-label experiments are proposals/prepared code, not completed results. Do not run a long reconstruction just for this meeting.

## What to open

- ZuCo: https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/zuco_plots.ipynb
- TeCo: https://colab.research.google.com/github/parmisbathayan/word-sentiment-eeg/blob/main/notebooks/teco_plots.ipynb

CPU → Run all only if the saved plots need regenerating. These notebooks read compact class means, not source EEG. No label/filter/extraction changes. They save small plotting outputs under each original run's analysis/all_plots_v1 folder. They can take minutes/Drive I/O time, not the full source-extraction workload. Completed extraction results remain read only.

ZuCo gallery: separate cells for overall reader means, original12-channel overview, regional means, GFP, and one individual-reader cell per task. Each reader is one horizontal row: whole head, original ROI, GFP and6regions. TeCo gallery: overall and regional cells, separate neutral comparisons, reader rows with whole head/GFP/6regions. Group shading is ±1SEM across readers, not standard deviation. Inspect displayed roster/count tables; strict-neutral subsets can use fewer readers.

## Methods to present as already done

- ZuCo1: SR, NR and TSR,12reader recordings per task. TeCo: task1,12readers. Task2/3 TeCo texts were located, but corresponding native EEG has not yet been located/validated for this pipeline. This is an access/schema scope limitation, not a scientific reason to omit those tasks permanently.
- Actual word-linked EEG voltage segments; no frequency-band summaries used as ERP time courses.
- The first stored fixation waveform for a word is located uniquely in sentence reading EEG. Missing/invalid first slots are not replaced by later fixations. This is not an independently verified first physical fixation.
- Historical selection: lexical polarity agrees with contextual-word polarity; uncertain/alignment-warning cases excluded. Whole-sentence polarity agreement was NOT required. Neutral included; function words included; classes not downsampled. TeCo CONTROL reading was excluded in this historical run.
- Average reference; whole-sentence fourth-order zero-phase Butterworth30Hz low-pass; epochs−200…+800ms; baseline−200…0 excluding0. ZuCo500Hz/105channels; TeCo documented1200Hz/126channels, voltage unit assumedµV.
- Reject unavailable/invalid fixation or sentence EEG, failed/ambiguous onset recovery, insufficient epoch/filter boundaries, nonfinite signals/epochs, and absolute epoch voltage above100µV. Original corpus preprocessing also applies upstream; do not invent its unavailable exclusion counts.
- Each accepted word occurrence from each reader is one EEG trial/instance. Repeated words and presentations count again when they are distinct recorded events. These instances are correlated within reader/sentence/word; do not call them independent statistical participants.
- Reader class means are averaged with equal reader weight within the displayed comparison roster. Group figures are descriptive, not a completed classification or statistically established sentiment effect.

## Actual accepted EEG instances, summed across all readers

These are old30Hz/old-selection results, not source vocabulary sizes and not newthree-label counts. Overlapping TeCo neutral subsets are NOT added again.

| Dataset | Positive instances | Negative instances | Neutral instances | Total |
|---|---:|---:|---:|---:|
| ZuCo SR |4325|3848|39306|47479|
| ZuCo NR |646|570|42718|43934|
| ZuCo TSR |975|414|36206|37595|
| ZuCo all tasks |5946|4832|118230|129008|
| TeCo task1 |1393|802|16528|18723|

Verified manifest accounting: ZuCo36completed recordings, TeCo12completed readers. This is not a fresh reread/checksum audit of every saved EEG artifact. The methods-report input-inventory subprocess failed; its displayed CalledProcessError only gives exit1, not the underlying cause. That reporting failure does not itself invalidate existing class means or plots. Defer diagnosis; capture the preceding child traceback when returning to it.

## What the plots can support

Show each task's group curves, regional patterns and reader variation. Report visual patterns only after inspecting the actual plots. Do not invent a significant effect, sentiment decoder or N400 interpretation from a curve difference alone. A whole-head signed mean is near zero by construction under average reference; it is not evidence of absent brain activity. GFP measures spatial spread of class-average voltage and depends partly on trial count/noise; it is not signed ERP voltage or single-trial power. Sensor regions do not precisely localize brain generators.

A reasonable meeting statement: “We recovered word-associated first-stored-fixation responses and compared positive, negative and neutral classes across tasks, readers and scalp regions. These are exploratory averages. We still need to evaluate label quality, overlapping events and linguistic/eye-movement confounds before attributing differences specifically to sentiment.”

## Paper comparison to mention

Pfeiffer etal2020: https://doi.org/10.1016/j.neuroimage.2020.116934

Their study also uses word-assigned fixations; it is not restricted to our first-stored-per-word rule. It uses ZuCo SR, excludes neutral/nearby negation, performs covariate-balanced trial sampling, and includes overlap/deconvolution modeling. Its preprocessing uses1–50Hz,125Hz sampling,−600…1000ms epochs and±90µV rejection. Our historical analysis uses all3ZuCo tasks plus TeCo task1, retains neutral, no new blanket negation filter or trial-balancing sampling, no deconvolution,30Hz/−200…800ms/±100µV. Our waveform matching recovers onsets; their matching procedure selects comparable trials. These differences must remain explicit; we have not reproduced their complete analysis. A complete public study-specific pipeline was not verified; public toolboxes are linked in PAPER_METHODS_COMPARISON.md.

## Backlog and professor questions — after this meeting

Priority1: audit/reporting without long EEG reconstruction:

- [ ] Diagnose methods_report input-inventory failure using its preceding stderr/traceback, not merely CalledProcessError. Do not change EEG results to fix a reporting error.
- [ ] Review English sentiment labels again if an independent new audit is desired. Existing English lexical/contextual labels are already agent-labeled; new three-level table joins them with existing sentence labels. Persian has had a new full165-sentence review. Neither is human consensus gold.
- [ ] Compare lexical, contextual-word and sentence labels; decide operational meaning of neutral/mixed and whether strict sentence neutrality is too restrictive.
- [ ] Question: should neutral targets be content words only? Historical results include function words. New grammar screen is operational, not validated POS gold.
- [ ] Verify native TeCo task2/3 EEG locations/schema, then label/extend tasks. Existing task1 is not the whole TeCo corpus.
- [ ] Review amplitude units (especially TeCo), montage/reference assumptions and source preprocessing documentation.
- [ ] Explain the exclusion categories to the user before final presentation/report preparation; this is a saved follow-up item, not a scheduled reminder.

Priority2: choices to agree with professor before expensive reconstruction:

- [ ] 30Hz versus40Hz: current results use30Hz. User requests40Hz in future source reconstruction. Ask whether a30/40 sensitivity comparison is scientifically useful;40Hz does not automatically make old results invalid.
- [ ] Blink-preceded events: should they be excluded or modeled? First verify blink timings/predecessor indices and upstream ocular cleaning. Word assignment is not blink removal.
- [ ] Short fixations: provisional working assumption of upstream short-fixation removal must remain explicitly UNVERIFIED for TeCo word EEG. Audit durations/segment conventions and word-to-allFixations mapping; do not assert allshort events removed. ZuCo publication describes upstream<100ms removal for engineered word data.
- [ ] Word bounds: document corpus assignment; upstream outside-word counts currently unknown. Do we need a further gaze-quality check?
- [ ] Overlap/deconvolution: rapid neighboring fixations produce overlapping responses within−200…800ms epochs. Ask whether descriptive FRPs suffice initially or whether overlap correction is required for inference. It is different from covariate matching, and may require a full event timeline.
- [ ] Covariates/confounds: compare word length, frequency, position, fixation duration, neighboring fixations and sentence context across sentiment classes. Example: if negative targets are longer, an EEG difference could reflect length/reading difficulty. Consider regression/stratified sensitivity checks; no automatic neutral downsampling.
- [ ] Negation and irony: audit contextual labels and negation scope. Ask whether a separate negation-excluded sensitivity analysis is needed; do not silently remove allsuch examples.
- [ ] First stored fixation versus first pass/allfixations: decide scientific target and document exporteddata limits. Never silently promote a later fixation when firstslot invalid.
- [ ] Epoch/baseline/±100µV policy: ask whether current windows/threshold and sentence-edge losses are acceptable; compare paper±90µV only as explicit sensitivity if needed.

Priority3: prepared/future experiments:

- [ ] Fourcontent-word policies: allthreeagree first; lexical-only; contextual-word-only; sentence-only. The latter policies select one target layer; they do not require the other two to disagree.
- [ ] IMPORTANT feasibility: allthreeagree+contentscreen leaves13neutral source positions inZuCo SR and11inTeCo BEFORE EEG quality rejection, not13/11totalreadertrials. With12readers, maximum156/132word-reader candidates respectively if allpresent, and fewer may survive. NR/TSR have larger strictneutral source pools. Decide before spending hours reconstructing tinyneutral groups.
- [ ] New40Hz/control-inclusive/content-only analysis must read source EEG again; oldclass means cannot recover discarded frequencies or previouslyexcluded controls. Keep separate versioned folders. Do not run it for this meeting merely because it is prepared.
- [ ] TeCo newlabel upload: connector quota error blockeddelivery. The new40Hz TeCo notebook asks for private word_labels_three_level.csv only ifmissing, verifiesSHA256, and saves inThesis/gpt uploaded files/word_sentiment_eeg/teco. Oldplot notebook needs no newCSV.
- [ ] Classification with reader/sentence/word-aware splits; avoid treating correlatedevents as independent participants. Report unseen-word generalization separately from same-word/same-reader diagnostic.
- [ ] Repeated-word within-reader correlation/similarity; distinctoccurrences may be compared, but neversameevent intrainandtest. Examinecontext changes andnoise.
- [ ] Statistical effect sizes/confidence intervals and suitable multiple-comparison control before claiming reliable sentiment effects. Regional analyses are exploratory until hypotheses/analysis plan fixed.
