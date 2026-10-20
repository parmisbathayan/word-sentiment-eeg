# Channel order

`zuco_chanlocs.csv` is reused from the existing thesis project. It contains the
105 EEG channels in the authors' plotting order, including the original flat
vertex reference Cz. Source: `EEG.chanlocs` in `matlab/gip_ZAB_SR5_EEG.mat` in
[norahollenstein/zuco-benchmark](https://github.com/norahollenstein/zuco-benchmark),
source SHA-256 `c42eaa3d234f14d667a9fbded19a5d5cdcb78375e40ca093a471d3afc5a84960`.
ZuCo retains 104 EGI scalp channels plus Cz. Average reference is applied to all
105 channels before ERP epoching. The full-channel means are saved; the plot
uses the explicitly listed centro-parietal ROI in `erp.py` / run provenance.
