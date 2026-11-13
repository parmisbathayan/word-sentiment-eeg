"""Paired-reader statistical checks; uses no source EEG or external downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('followups', ROOT / 'scripts/analyze_zuco_followups.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module.exact_sign_flip(np.ones(12)) == 2 / 4096
assert module.exact_sign_flip(np.zeros(12)) == 1
assert module.exact_sign_flip([1, -1] * 6) == 1
d = np.array([.1, .8, -1, 3, -2, 4])
assert module.exact_sign_flip(d) == module.exact_sign_flip(-d)
assert module.exact_sign_flip(d) == module.exact_sign_flip(d * 1000)
np.testing.assert_allclose(module.holm_adjust([.01, .04, .03]), [.03, .06, .06])
np.testing.assert_allclose(module.holm_adjust([1, 0, .5]), [1, 0, 1])
try:
    module.exact_sign_flip([1, np.nan])
    raise AssertionError('Nonfinite difference was accepted')
except ValueError:
    pass
s = module.summarize_differences([1, 1, 1, -5])
assert not s['leave_one_reader_out_direction_stable']

with tempfile.TemporaryDirectory() as temp:
    run = Path(temp)
    (run / 'audit').mkdir()
    (run / 'audit/integrity.json').write_text(json.dumps({'status': 'passed'}))
    # Synthetic coordinates exercise every one of the existing fixed regions.
    positions = [(8, 0, 8), (0, 0, 8), (-4, 0, 8), (-7, 0, 8), (0, 6, 8), (0, -6, 8)]
    channel_names = list(module.ROI) + [f'OTHER{i}' for i in range(105-len(module.ROI))]
    montage = pd.DataFrame([{'index0': i, 'labels': channel_names[i], 'X': positions[i % 6][0],
                           'Y': positions[i % 6][1], 'Z': positions[i % 6][2]} for i in range(105)])
    montage.to_csv(run / 'zuco_chanlocs.csv', index=False)
    names = montage.labels.to_numpy(dtype=str)
    records = []
    rng = np.random.default_rng(472)
    readers = np.array([f'R{i:02}' for i in range(12)])
    for task in module.TASKS:
        values = rng.normal(size=(12, 3, 105, len(module.TIMES_MS)))
        # Large shared negative-positive offset in all regions/windows.
        values[:, 0] += 2
        np.savez_compressed(run / f'{task}_grand_average.npz', subject_means_uV=values,
                            subjects=readers, classes=module.CLASSES, times_ms=module.TIMES_MS,
                            channel_names=names)
        records.extend({'task': task, 'subject': reader, 'epochs_per_class': {c: 5 for c in module.CLASSES}}
                       for reader in readers)
    (run / 'manifest.json').write_text(json.dumps({'status': 'completed', 'files': records}))
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*') if p.is_file()}
    results, differences = module.analyze(run, make_plots=False)
    assert len(results) == 234
    assert len(differences) == 234 * 12
    assert np.all(results.p_holm_within_declared_family >= results.p_sign_flip_two_sided)
    # With 12 readers the smallest two-sided raw p is 2/4096. Holm's first
    # threshold is .05/216, so no test in this family can pass (even a huge one).
    assert not results.loc[results.family.eq('secondary_regions'), 'passes_holm_005'].any()
    assert results.loc[results.family.eq('primary_original_roi'), 'passes_holm_005'].any()
    assert (run / 'analysis/paired_reader_contrasts/RESULTS_INTERPRETATION.md').exists()
    for path, sha in originals.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    print('ZuCo paired contrasts: exact enumeration, Holm family, reader estimates, preservation passed')
