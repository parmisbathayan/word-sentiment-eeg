"""Exploratory paired-reader contrasts using existing ZuCo class averages.

No source EEG is loaded and the accepted epochs are not changed. The family of
18 primary tests uses the original ROI and two existing windows; a separate
secondary family contains 216 geometric-region/window/contrast tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from word_sentiment_eeg.erp import CLASSES, TIMES_MS, ROI
from review_regions import REGIONS, WINDOWS, assign_regions

TASKS = ('SR', 'NR', 'TSR')
CONTRASTS = (('negative', 'positive'), ('negative', 'neutral'), ('positive', 'neutral'))
PRIMARY_WINDOWS = ((300, 500), (500, 800))
ALPHA = .05


def exact_sign_flip(differences):
    """Two-sided exact sign-flip test of the paired mean, including all signs.

    Readers are independent units. Sign exchangeability requires symmetric
    paired reader differences under the null; observational word classes were
    not experimentally randomized. Zero differences are retained.
    """
    d = np.asarray(differences, dtype=np.float64)
    if d.ndim != 1 or not 2 <= len(d) <= 16 or not np.isfinite(d).all():
        raise ValueError('Need 2–16 finite independent reader differences')
    masks = np.arange(2 ** len(d), dtype=np.uint32)[:, None]
    signs = 2 * ((masks >> np.arange(len(d), dtype=np.uint32)) & 1).astype(np.float64) - 1
    null = np.abs(signs @ d / len(d))
    observed = abs(float(d.mean()))
    tolerance = 32 * np.finfo(float).eps * max(float(np.abs(d).mean()), observed)
    # This is full enumeration, not sampled permutations: no Monte Carlo +1.
    return float(np.count_nonzero(null >= observed - tolerance) / len(null))


def holm_adjust(pvalues):
    """Holm familywise-error correction, valid under arbitrary test dependence."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid p values')
    order = np.argsort(p, kind='stable')
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def summarize_differences(differences):
    d = np.asarray(differences, dtype=np.float64)
    p = exact_sign_flip(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    loso = (d.sum() - d) / (len(d) - 1)
    # Reader bootstrap, conditional on the fixed word collection. Pointwise
    # descriptive intervals are not simultaneous confidence bounds and should
    # not replace the corrected test. Same resampling indices in every cell.
    rng = np.random.default_rng(271828)
    bootstrap = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [.025, .975])
    direction = np.sign(mean)
    return {
        'readers': len(d), 'mean_difference_uV': mean,
        'median_difference_uV': float(np.median(d)),
        'sd_reader_difference_uV': sd, 'sem_reader_difference_uV': sd / np.sqrt(len(d)),
        'paired_dz': mean / sd if sd > 0 else None,
        'reader_bootstrap_pointwise_95_low_uV': float(ci_low),
        'reader_bootstrap_pointwise_95_high_uV': float(ci_high),
        'readers_above_zero': int((d > 0).sum()), 'readers_below_zero': int((d < 0).sum()),
        'readers_equal_zero': int((d == 0).sum()),
        'fraction_readers_matching_group_direction': float((np.sign(d) == direction).mean()) if direction else None,
        'leave_one_reader_out_min_uV': float(loso.min()),
        'leave_one_reader_out_max_uV': float(loso.max()),
        'leave_one_reader_out_direction_stable': bool(direction != 0 and np.all(np.sign(loso) == direction)),
        'p_sign_flip_two_sided': p, 'enumerated_sign_patterns': 2 ** len(d),
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(run_dir, make_plots=True):
    run = Path(run_dir)
    integrity_path = run / 'audit/integrity.json'
    if json.loads(integrity_path.read_text())['status'] != 'passed':
        raise ValueError('The saved-result integrity audit must pass first')
    manifest_path = run / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'completed':
        raise ValueError('Extraction manifest must be completed')
    records = manifest['files']
    identities = [(x['task'], x['subject']) for x in records]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate task/reader record')
    accepted = {(x['task'], x['subject']): x['epochs_per_class'] for x in records}
    montage_path = run / 'zuco_chanlocs.csv'
    mapping = assign_regions(pd.read_csv(montage_path))
    out = run / 'analysis/paired_reader_contrasts'
    out.mkdir(parents=True, exist_ok=True)
    hashes = {str(x.relative_to(run)): digest(x) for x in (integrity_path, manifest_path, montage_path)}
    rows = []; readers_rows = []; task_readers = {}
    for task in TASKS:
        source = run / f'{task}_grand_average.npz'
        hashes[source.name] = digest(source)
        with np.load(source, allow_pickle=False) as data:
            values = data['subject_means_uV'].astype(np.float64)
            readers = data['subjects'].astype(str).tolist()
            if data['channel_names'].astype(str).tolist() != mapping.labels.tolist():
                raise ValueError(f'{task}: inconsistent channel order')
            if not np.array_equal(data['times_ms'], TIMES_MS) or not np.array_equal(data['classes'], CLASSES):
                raise ValueError(f'{task}: unexpected class/time axes')
        if len(readers) != len(set(readers)) or not 2 <= len(readers) <= 16:
            raise ValueError(f'{task}: need 2–16 unique readers')
        if set(readers) != {s for t, s in identities if t == task and min(accepted[(t,s)].values())>=5}:
            raise ValueError(f'{task}: grand-average reader roster differs from manifest')
        if values.shape != (len(readers), len(CLASSES), len(mapping), len(TIMES_MS)) or not np.isfinite(values).all():
            raise ValueError(f'{task}: invalid subject means')
        if any(accepted[(task, subject)][cls] < 5 for subject in readers for cls in CLASSES):
            raise ValueError(f'{task}: a reader has fewer than five accepted epochs in a class')
        task_readers[task] = readers
        roi_indices = np.array([mapping.labels.tolist().index(name) for name in ROI])
        region_specs = [('primary_original_roi', 'original_roi', roi_indices, PRIMARY_WINDOWS)]
        region_specs += [('secondary_regions', region, np.flatnonzero(mapping.region.eq(region).to_numpy()), WINDOWS)
                         for region in REGIONS]
        for family, region, channels, windows_for_region in region_specs:
            regional = values[:, :, channels, :].mean(axis=2)
            for low, high in windows_for_region:
                mask = (TIMES_MS >= low) & (TIMES_MS < high)
                windows = regional[:, :, mask].mean(axis=2)
                for first, second in CONTRASTS:
                    label = f'{first}_minus_{second}'
                    d = windows[:, CLASSES.index(first)] - windows[:, CLASSES.index(second)]
                    key = {'family': family, 'task': task, 'region': region, 'contrast': label,
                           'window_start_ms': low, 'window_stop_ms_exclusive': high}
                    rows.append({**key, 'region_channels': len(channels), **summarize_differences(d)})
                    for i, subject in enumerate(readers):
                        readers_rows.append({**key, 'subject': subject, 'difference_uV': float(d[i]),
                            'first_class_mean_uV': float(windows[i, CLASSES.index(first)]),
                            'second_class_mean_uV': float(windows[i, CLASSES.index(second)]),
                            'first_class_epochs': accepted[(task, subject)][first],
                            'second_class_epochs': accepted[(task, subject)][second]})
        print(f'PAIRED CONTRASTS {task}: {len(readers)} readers; 6 original-ROI and 72 regional comparisons', flush=True)
    results = pd.DataFrame(rows)
    family_sizes = {'primary_original_roi': 18, 'secondary_regions': 216}
    if len(results) != sum(family_sizes.values()):
        raise ValueError('Incomplete planned family; refusing selective correction')
    results['p_holm_within_declared_family'] = np.nan
    for family, size in family_sizes.items():
        mask = results.family.eq(family)
        if int(mask.sum()) != size:
            raise ValueError('Incomplete planned family; refusing selective correction')
        results.loc[mask, 'p_holm_within_declared_family'] = holm_adjust(results.loc[mask, 'p_sign_flip_two_sided'].to_numpy())
    results['passes_holm_005'] = results.p_holm_within_declared_family <= ALPHA
    per_reader = pd.DataFrame(readers_rows)
    results.to_csv(out / 'paired_contrasts.csv', index=False)
    per_reader.to_csv(out / 'reader_differences.csv', index=False)
    protocol = {
        'status': 'completed', 'analysis_type': 'exploratory, conditional on the fixed annotated stimuli',
        'unit': 'reader, equally weighted; no trial/electrode pseudoreplication',
        'tasks': TASKS, 'regions': REGIONS, 'windows_ms_half_open': WINDOWS,
        'contrasts': [f'{a}_minus_{b}' for a, b in CONTRASTS],
        'correction': 'Holm separately within primary original-ROI family of 18 and secondary regional family of 216; no claim of joint FWER across both families',
        'family_sizes': family_sizes, 'primary_roi_channels': ROI, 'primary_windows_ms_half_open': PRIMARY_WINDOWS,
        'primary_label_is_exploratory': True,
        'bootstrap_interval': '10,000 reader bootstrap resamples, seed 271828, percentile pointwise 95%; not adjusted for multiplicity',
        'alpha': ALPHA, 'readers_by_task': task_readers,
        'test': 'two-sided absolute paired-mean statistic; exhaustive 2^n sign flips including observed signs',
        'assumptions': ['Independent readers', 'Paired reader differences symmetric around zero under the null',
                        'Classes are observational and were not randomly assigned'],
        'limitations': ['No generalization across a population of words/items: item-level variance is absent',
                       'Overlapping fixation responses are not deconvolved',
                       'Unequal class counts and lexical/sentence/fixation confounds remain',
                       'Small reader sample and multiplicity limit power; null results do not prove absence',
                       'Scalp regions do not identify cortical sources; amplitude units retain source assumptions',
                       'Regions/windows were fixed before these tests, but this analysis follows inspected descriptive plots; not confirmatory'],
        'source_sha256': hashes,
        'original_extraction_files_modified': False,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    write_interpretation(results, out, task_readers)
    if make_plots:
        plot_reader_differences(results, per_reader, out)
    print(f'FOLLOWUPS COMPLETE: {out}; {int(results.passes_holm_005.sum())}/{len(results)} contrasts pass their declared-family Holm correction', flush=True)
    return results, per_reader


def write_interpretation(results, out, task_readers):
    passed = results[results.passes_holm_005]
    lines = ['# Exploratory reader-level contrasts', '',
        'These analyses ask whether a class-average voltage difference is consistent across readers. '
        'They do not test whether sentiment itself caused the difference.', '',
        f"There are 18 primary original-ROI comparisons and 216 separate secondary regional comparisons. Reader counts: " +
        ', '.join(f'{task}: {len(readers)}' for task, readers in task_readers.items()) + '.', '',
        f'{int(passed.family.eq("primary_original_roi").sum())}/18 primary and '
        f'{int(passed.family.eq("secondary_regions").sum())}/216 secondary comparisons pass their separate Holm corrections at 0.05.', '',
        'With 12 readers, the smallest two-sided exact p is 2/4096 = 0.0004883. '
        'The first secondary Holm threshold is 0.05/216 = 0.0002315, so this broad secondary family cannot yield a rejection. '
        'The primary family uses the original 12-channel ROI and the two existing 300–500 and 500–800 ms windows, '
        'fixed before viewing these reader-test results. It remains exploratory because the descriptive data have already been inspected.', '']
    if len(passed):
        lines += ['Passing comparisons are listed completely, rather than choosing a best region:', '',
                  '|Family|Task|Region|Window (ms, end exclusive)|Contrast|Mean difference (µV)|Holm p|',
                  '|---|---|---|---|---|---:|---:|']
        for r in passed.itertuples():
            lines.append(f'|{r.family}|{r.task}|{r.region}|{r.window_start_ms}–{r.window_stop_ms_exclusive}|{r.contrast}|{r.mean_difference_uV:.5g}|{r.p_holm_within_declared_family:.5g}|')
    else:
        lines.append('The current saved averages do not establish a reliable class separation under this planned family of reader-level tests. This is not evidence that the brain has no sentiment response.')
    lines += ['', 'Inspect `reader_differences.csv` and the reader-dot figures to see whether a mean is shared across readers or driven by a small subset. '
              '`paired_contrasts.csv` gives every estimate, raw and adjusted p value, direction count, and leave-one-reader-out range. '
              'Leave-one-out ranges describe sensitivity; they are not confidence intervals or a separate significance test.', '',
              'The CSV also includes reader-bootstrap pointwise 95% intervals (10,000 resamples). These are descriptive intervals, '
              'not simultaneous bounds across comparisons and not a substitute for the Holm tests. Neither a CI nor a p value makes a class difference sentiment-specific.', '',
              'The sample is small, word labels are agent judgments, classes have unequal trial counts, and differences can reflect word length, '
              'frequency, part of speech, sentence context, fixation duration, or overlapping eye-movement responses. '
              'Readers read a fixed stimulus collection; this analysis does not estimate item-level uncertainty or generalize to new words.', '',
              '## Next analyses that answer stronger questions', '',
              '1. Add sentence-context labels and rebuild stratified word averages. Include neutral words in neutral sentences, '
              'and compare word classes within matched sentence-context strata where enough examples exist.',
              '2. Save compact per-word window features and covariates during that rebuild. Match or model word length, frequency, '
              'part of speech, sentence position, fixation duration, and preceding-fixation history; compare content-word controls.',
              '3. Fit subject/item mixed-effects models to those word-level features. The current pooled reader means cannot recover item variance.',
              '4. For claims about time courses/components, use continuous EEG and fixation/saccade timing to model overlapping responses '
              '(regression/deconvolution). Existing pooled means cannot undo overlap.', '',
              'These outputs reuse the original accepted word set and preprocessing. No source EEG was downloaded or re-extracted. '
              'No classifier was trained. All new files are confined to `analysis/paired_reader_contrasts/`.', '']
    (out / 'RESULTS_INTERPRETATION.md').write_text('\n'.join(lines))


def plot_reader_differences(results, readers, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Primary estimates and reader distributions, with pointwise bootstrap CIs.
    primary = results[results.family.eq('primary_original_roi')]
    for task in TASKS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
        for ax, contrast in zip(axes, [f'{a}_minus_{b}' for a, b in CONTRASTS]):
            for i, (low, high) in enumerate(PRIMARY_WINDOWS):
                part = readers[(readers.family == 'primary_original_roi') & (readers.task == task) &
                               (readers.contrast == contrast) & (readers.window_start_ms == low)].sort_values('subject')
                estimate = primary[(primary.task == task) & (primary.contrast == contrast) & (primary.window_start_ms == low)].iloc[0]
                ax.scatter(i + np.linspace(-.13, .13, len(part)), part.difference_uV, s=25, alpha=.7)
                ax.plot([i-.2, i+.2], [estimate.mean_difference_uV]*2, color='black', lw=2)
                ax.plot([i, i], [estimate.reader_bootstrap_pointwise_95_low_uV, estimate.reader_bootstrap_pointwise_95_high_uV], color='black', lw=1)
            ax.axhline(0, color='grey', lw=.8)
            ax.set_xticks(range(2), ['300–500', '500–800'])
            ax.set_title(contrast.replace('_minus_', ' − '))
        fig.suptitle(f'{task}: original 12-channel ROI; paired reader differences\nDots: readers; black: mean and pointwise bootstrap 95% interval (unadjusted)')
        fig.supxlabel('Time window from first fixation onset (ms, end exclusive)')
        fig.supylabel('Within-reader voltage difference (µV as stored)')
        fig.tight_layout(rect=(.02, .02, 1, .88))
        fig.savefig(out / f'{task}_original_roi_reader_differences.png', dpi=140)
        plt.close(fig)
    for task in TASKS:
        for contrast in [f'{a}_minus_{b}' for a, b in CONTRASTS]:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
            for ax, region in zip(axes.flat, REGIONS):
                for i, (low, high) in enumerate(WINDOWS):
                    part = readers[(readers.task == task) & (readers.region == region) &
                                   (readers.contrast == contrast) & (readers.window_start_ms == low)].sort_values('subject')
                    estimate = results[(results.task == task) & (results.region == region) &
                                       (results.contrast == contrast) & (results.window_start_ms == low)].iloc[0]
                    # Deterministic offsets, so a rerun produces the same display.
                    ax.scatter(i + np.linspace(-.13, .13, len(part)), part.difference_uV, s=20, alpha=.75)
                    ax.plot([i - .19, i + .19], [estimate.mean_difference_uV] * 2, color='black', lw=2)
                    if estimate.passes_holm_005:
                        ax.annotate('*', (i, estimate.mean_difference_uV), xytext=(0, 7), textcoords='offset points', ha='center')
                ax.axhline(0, color='grey', lw=.8)
                ax.set_xticks(range(len(WINDOWS)), [f'{low}–{high}' for low, high in WINDOWS])
                ax.set_title(region)
            fig.suptitle(f'{task}: {contrast.replace("_", " ")}\nEach dot is one reader; black bars are equal-reader means; * secondary-family Holm p ≤ .05 (216 tests)')
            fig.supxlabel('Time window from first fixation onset (ms, end exclusive)')
            fig.supylabel('Within-reader voltage difference (µV as stored)')
            fig.tight_layout(rect=(.02, .02, 1, .91))
            fig.savefig(out / f'{task}_{contrast}_reader_differences.png', dpi=130)
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    analyze(args.run_dir, not args.no_plots)
