"""Inspect real NR/TSR word EEG in cloud; retain only a small audit report."""
import gc
import json
from pathlib import Path
import sys
import tempfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_erp import cloud_download
from word_sentiment_eeg.erp import iter_sentences, locate
import numpy as np


def audit(record, directory):
    path = directory / record['filename']
    cloud_download(record, path)
    result = {'task':record['task'], 'subject':record['subject'], 'source_url':record['url'],
              'source_bytes':path.stat().st_size, 'scope':'first 10 readable sentences; not a complete reader audit',
              'sentences_inspected':0, 'words_inspected':0, 'finite_word_fixation_segments':0,
              'match_results':Counter(), 'examples':[]}
    reader = iter_sentences(path)
    try:
        for position, words, raw, segments in reader:
            result['sentences_inspected'] += 1
            result['words_inspected'] += len(words)
            if raw is not None and np.isfinite(raw).all():
                result.setdefault('sentence_eeg_shape',list(raw.shape))
                for j, cells in enumerate(segments):
                    for k, segment in enumerate(cells):
                        if segment is None or not np.isfinite(segment).all():
                            continue
                        result['finite_word_fixation_segments'] += 1
                        onset, method = locate(raw, segment)
                        result['match_results'][method] += 1
                        if onset is not None and len(result['examples']) < 3:
                            result['examples'].append({'sentence_position':position, 'word_index':j,
                                'word':words[j], 'fixation_index':k, 'word_eeg_shape':list(segment.shape),
                                'onset_sample_sentence_0based':onset, 'match_method':method})
            if result['sentences_inspected'] >= 10:
                break
    finally:
        reader.close()
        path.unlink(missing_ok=True)
        gc.collect()
    result['verified_word_segments_and_sentence_eeg'] = bool(result['examples'])
    return result


def main():
    if sys.platform == 'darwin':
        raise SystemExit('Run only in a cloud runtime. This audit downloads EEG.')
    records=json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files']
    selected=[r for r in records if (r['task'],r['subject']) in [('NR','ZJS'),('TSR','ZPH')]]
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        for record in selected:
            print(f"Inspecting {record['task']}/{record['subject']} in cloud", flush=True)
            item=audit(record,Path(tmp))
            results.append(item)
            print(json.dumps(item,indent=2),flush=True)
    target=ROOT/'source_availability_audit.json'
    target.write_text(json.dumps({'files':results,'all_selected_files_verified':all(r['verified_word_segments_and_sentence_eeg'] for r in results)},indent=2)+'\n')
    if not all(r['verified_word_segments_and_sentence_eeg'] for r in results):
        raise SystemExit('At least one sampled source did not provide verified word segments')


if __name__=='__main__':
    main()
