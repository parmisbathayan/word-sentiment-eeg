# ZuCo paper figure reconstruction

This is an independent Python/Colab reconstruction of the four main figures in
Pfeiffer et al. (2020), *Neural dynamics of sentiment processing during naturalistic
sentence reading*, NeuroImage 221, 116934.
https://doi.org/10.1016/j.neuroimage.2020.116934

Open `notebooks/zuco_paper_figures.ipynb` in Colab. Run all; GPU recommended for
channel-shuffle permutations. Regression and mixed models run on CPU. Expect a
substantial source pass, potentially hours, rather than a quick old-plot redraw.
This workflow runs only ZuCo SR, not NR/TSR or TeCo. Published decoding Table 1
is outside the current four-figure request.

## Outputs

| Figure | Computed panels |
| --- | --- |
| 1 | Actual sentence/fixation example, reconciled selection flow, all-channel mean traces and electrode/time maps, E62 trial rasters sorted by fixation duration; raw versus cleaned |
| 2 | Raw positive/negative/difference electrode-time heatmaps, trial GFP, pointwise mixed models, corrected clusters, topographic consistency, four scalp maps |
| 3 | Same panels after continuous nuisance regression, using the same accepted event IDs |
| 4 | Native horizontal/vertical eye velocity, neighboring fixation probabilities, word length before/after selection, fixation duration and word frequency |

Each has PNG and SVG files; each notebook section displays one complete figure.
Color limits follow the computed data. Raw 224–280 ms and cleaned 224–304 ms maps
are **predeclared reference windows**, not a claim that this run detects the
paper's significant windows. No published signal or significance map is copied
into these computed figures. The layout test fixtures are prominently marked
SYNTHETIC and are never used in a real run.

## Data and alignment

The native 500-Hz, 105-channel preprocessed SR recordings and raw/corrected eye
exports come from the corpus OSF release. Original released sentence EEG is
matched as a full multichannel waveform inside each native block. Thus sentence
identity does not depend on assuming 50 trials per block. Ambiguous, unavailable
or unmatched sentences do not acquire guessed identities. File SHA256 hashes,
mapped sentence counts and trigger-fit residuals are saved per reader/block.

All eligible word-associated fixations enter this route, including repeat
fixations. This differs from our previous first-stored-fixation experiment.
Original SST five-level word labels are joined through the sentence tree, with
an explicit original dictionary-only fallback where needed. Unknown labels are
excluded. No agent-reviewed sentiment labels enter the paper reconstruction.
Neutral events and preceding-three-word negation events are excluded here to
follow the paper's binary selection; our older three-class data is untouched.

The corrected native exports have stored duration fields that can be physically
inconsistent. Durations are recomputed from recorded end minus start timestamps,
with positivity/plausibility validation and mismatch counts. No onset is invented.
Raw eye events provide the nuisance timeline; corrected gaze provides word mapping.

## Explicit methodological differences

- Corpus preprocessing is reused. Automagic, MARA, ICA and CleanLine are not rerun.
- Python centered 1–50 Hz FIR, 125-Hz resampling, independent spline/FIR regression
  and sparse least squares replace the MATLAB stack. No numerical parity claim.
- Trigger-sequence matching and a linear eye-to-EEG clock fit are required.
  Maximum accepted residual is 8 ms (one output sample), versus the published
  2-ms native-sample criterion. The observed residual and whether 2 ms was met
  are recorded for every block. A block failing the residual check is explicitly
  excluded with its observed fixation count. Missing/reordered triggers or other
  unsupported schema problems still stop the run. Inspect `source_coverage.csv`;
  do not describe the selected blocks as the complete published cohort.
- Original released word rectangles and an explicitly defined blink-predecessor
  rule are used. Earlier preprocessing exclusions cannot be reconstructed as
  newly counted exclusions from an already corrected export.
- Matching uses reader, exact word length 1–20 and tertiles of preceding/following
  fixation-onset gaps. The authors' exact sampling implementation was not located.
- Epochs are −600 to +1000 ms. No unstated baseline subtraction. Raw excursions
  beyond ±90 µV remove the same identity from raw and cleaned conditions.
- Gaussian identity mixed models use reader random intercept and sentiment slope;
  ML covariance estimates and asymptotic Wald tests are independently implemented.
  Failed fits remain missing and their counts are reported.
- GFP is calculated within each trial across electrodes before class aggregation.
  TCT independently shuffles channel identities within each ranked trial pair
  (5,000 draws). Cluster testing uses paired sign flips and maximum absolute mass
  (1,000 draws), distance-pruned Delaunay neighbors and at least five active neighbors.
  Temporal display bars require at least six consecutive samples.
- Control comparisons for neighbor bins, duration and frequency use reader-paired
  t-tests; eye-velocity curves use individual-event mixed models when coverage permits.
- Frequency is the count of case-folded, edge-punctuation-stripped occurrences in
  the released SR text. It is not an external-language Zipf frequency measure.

The output `METHOD_DIFFERENCES.md`, protocol, model-convergence audit, event
membership and exclusion tables preserve these choices with the figures. This
is a figure reconstruction with documented departures, not proof of exact
replication of the paper's published effects or trial counts.

## Storage and resume

Destination: `Thesis/Results/word_sentiment_eeg/zuco_paper_figures/`.
The fresh-run peak allowance is approximately 1.5 GB additional Drive space:
about 1.01 GB uncompressed permutation sums, plus checkpoints, statistical
summaries, figure files and margin. Colab verifies account quota before source
reads. Final outputs are estimated at 50–200 MB; the notebook reports actual size.
The final estimate is not a measured full-12-reader output size.

No second full-channel EEG archive is saved. Native downloads live temporarily
in Colab and are removed after conversion. Only the E62 trial raster is retained
at trial level for Figure 1. Completed-reader checksums prevent counting a reader
twice. Permutation chunks use atomic generations and reader membership; interrupted
chunks resume without adding completed readers again. An incomplete reader can
need source download/regression again. The original first-fixation archive,
classification features and earlier figures are not changed.

After inference and figure creation succeed, only this workflow's `null_work/`
folder is removed. Small final permutation statistics remain. A completed run
verifies reader/figure checksums and returns without source reads or refitting.
`--plots-only` redraws from saved statistics. Changed preparation fingerprints
stop rather than silently mix different methods. Never run two writers at once.
