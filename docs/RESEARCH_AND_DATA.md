# Data evidence and research notes — 20 October 2026

The prepared source uses existing local thesis outputs:
`eeg_to_text_v4_continuous/v4/sentences.csv` and the original SR word table
`followup_frp_analysis/frp_analysis/signal/word_targets.csv`.
The earlier code constructs sentence text by joining original word-cache
strings. SR was verified token-for-token against its 7,129-word source table.
NR/TSR IDs and word counts were checked against their connected Drive extraction
records across all subjects. No new EEG dataset was downloaded to the Mac.

The source contains 400 SR sentences and 637 distinct additional sentences.
NR has 300 distinct sentences. TSR has 407 presentations but 386 distinct
sentences, including repeated text; 49 sentences are shared by NR and TSR.
All unique sentence IDs in the existing task audits are represented.
One TSR sentence has a tokenization/encoding discrepancy and is flagged for
strict cache-alignment exclusion, not silently repaired.

The source corpus and exact agent decisions are frozen in annotations/.
All 1,037 sentences and 20,458 source tokens were reviewed in this chat.
Compilation is deterministic and records hashes. The whole-sentence sentiment
labels were not copied to words; all ordinary neutral tokens are explicitly
neutral within reviewed sentences. No dictionary/classifier produces the v2
labels. These are single-agent judgments, not independent human consensus.

Verified Drive source folders:
- [SR TRT](https://drive.google.com/drive/folders/1fvMTjF-7z5JEI18LQhQXyzL-iZDi4Y3X)
- [NR TRT](https://drive.google.com/drive/folders/1zSQqIzNG0XhxEMvzDbnQpnRx4HOJbuWz)
- [TSR TRT](https://drive.google.com/drive/folders/11yXF_QJy3n4i2PEbgi4Qk3IxDYePmb0a)
- [Original SR MATLAB files](https://drive.google.com/drive/folders/1l6JEal539PVjl6V05oSOczLE4tFE6G1l)

Each cache folder has 12 subject files. The actual binary EEG-to-annotation join
will be checked by the Colab run; source text coverage is complete but that
is not a claim of complete valid EEG coverage. Missing fixation/raw fields and
reader-dependent missing data stay explicit.

Research basis:
- [ZuCo 1.0](https://doi.org/10.1038/sdata.2018.291) and [author data](https://osf.io/q3zws/).
- [Pfeiffer et al. (2020), NeuroImage 218, 116934](https://doi.org/10.1016/j.neuroimage.2020.116934):
  close word-valence FRP precedent; includes negation/event selection,
  deconvolution and multi-trial averaging. This simpler band-power screen is
  not a reproduction of that decoding result.
- [ZuCo 2.0](https://aclanthology.org/2020.lrec-1.18/) is a separate extension,
  not silently included in the current claim of full ZuCo 1.0 task coverage.
- [NRC VAD](https://saifmohammad.com/WebPages/nrc-vad.html) remains available for
  an explicit later lexical comparison. Its data is not redistributed here.

Completed: source reconciliation, agent annotation, contextual/lexical separation,
annotation artifacts, strict join implementation, short Colab runner.
Pending: real EEG alignment audit/results and independent annotation validation.
