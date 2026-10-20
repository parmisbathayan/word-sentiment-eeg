# Research and data evidence — 20 October 2026

## What was verified

The existing thesis code already reads the original ZuCo word structures:
`src/fusion/word_eeg.py`, `scripts/extract_word_eeg.py`, and
`src/followup/frp.py` in `parmis-thesis-eeg-sentiment`.
Its word-feature cache schema is words, per-trial offsets, sentence/sample IDs,
fixation counts, optional eye durations and `[words,8,105]` EEG features.
The new project validates this schema, preserving offsets and missing words.

Connected Drive listings on 20 October 2026 showed **12 subject NPZ files in
each of these caches**. Their binary contents were not downloaded to the Mac.

| Source | Verified Drive location |
|---|---|
| SR TRT cache | [TRT](https://drive.google.com/drive/folders/1fvMTjF-7z5JEI18LQhQXyzL-iZDi4Y3X) |
| NR TRT cache | [TRT_NR](https://drive.google.com/drive/folders/1zSQqIzNG0XhxEMvzDbnQpnRx4HOJbuWz) |
| TSR TRT cache | [TRT_TSR](https://drive.google.com/drive/folders/11yXF_QJy3n4i2PEbgi4Qk3IxDYePmb0a) |
| Existing original SR MATLAB files | [zuco_og_raw](https://drive.google.com/drive/folders/1l6JEal539PVjl6V05oSOczLE4tFE6G1l) |
| Existing FRP feature cache | [FRP](https://drive.google.com/drive/folders/1ci0gwZLE_0DvmxFD0Cf21LftfxdVc-oX) |

Directory listings establish availability, not correctness, usable coverage or
class counts. The first real Colab run validates the arrays, hashes all source
caches, records expected missing readers, and computes actual per-class counts.
The original raw files may lack `word.rawEEG` for some readers; this is checked
in cloud, not assumed from the existence of a MATLAB file.

The phrase “super dataset” in the project request was interpreted as ZuCo
(the corpus used in the prior experiments). This implementation uses ZuCo 1.0
SR, NR and TSR. It does not claim all worldwide word-aligned EEG corpora are
included, or that ZuCo 2.0 has already been inventoried.

## Directly relevant literature

- [Hollenstein et al., ZuCo 1.0 (2018)](https://doi.org/10.1038/sdata.2018.291):
  original corpus of simultaneously recorded EEG and eye tracking. Author data:
  [OSF ZuCo 1.0](https://osf.io/q3zws/). Word timing follows fixations during
  natural sentence reading, rather than isolated word presentation.
- [Pfeiffer et al., NeuroImage 218, 116934 (2020)](https://doi.org/10.1016/j.neuroimage.2020.116934):
  especially close prior work on word sentiment in ZuCo SR. It uses SST word
  annotations, excludes nearby negations, and analyzes deconvolved FRPs.
  Reported multi-trial-average decoding cannot be read as individual-fixation
  accuracy. It motivates a later FRP replication if the audit supports it.
- [Hollenstein et al., ZuCo 2.0 (2020)](https://aclanthology.org/2020.lrec-1.18/):
  additional normal/task-specific reading data. It needs its own acquisition
  and schema audit before inclusion; it is not automatically substituted for
  ZuCo 1.0 in this version.
- [Mohammad, NRC VAD (2018)](https://aclanthology.org/P18-1017/): human ratings
  for English lexical valence/arousal/dominance. The author's
  [resource page and terms](https://saifmohammad.com/WebPages/nrc-vad.html)
  provides v1 and v2.1 separately. The run explicitly uses **v1** and validates
  its `[0,1]` schema. Thresholds here are project decisions.

The paper on word sentiment also means this direction has a specific prior
baseline to discuss with the supervisors. Its results are not evidence that
the proposed simpler feature classifier will succeed. A faithful replication
would require its event selection, deconvolution, matched trial sampling and
averaging protocol; these are not claimed by this first implementation.

## Why NRC first, and what SST could add

SST supplies word/phrase/sentence annotations for the SR movie-review material,
making it valuable for a focused replication. NR and TSR contain other text.
NRC supplies a common lexical target across all three tasks without making
unlabeled sentences into sentiment classes. A later SR-only SST comparison
should preserve original tokenization, phrase-tree alignment and disagreements
with NRC. Single-word SST labels are not automatically contextual annotations
of every word occurrence.

## Prepared versus measured

Prepared: code, protocol, class exports, optional raw export, Colab runner,
offline synthetic checks and cloud integration checks.

Pending: real cache validation, actual class balance/coverage, raw-field
availability, real EEG scores, and scientific conclusions. The new workflow
does not overwrite or reinterpret the completed sentence-level experiments.
