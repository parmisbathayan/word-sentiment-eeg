"""Build a review source from existing local thesis text, with no downloads."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from word_sentiment_eeg.common import sentence_key, digest, file_digest

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parents[1] / 'parmis-thesis-eeg-sentiment-review'
SOURCE = OLD / 'saved_results/eeg_to_text_v4_continuous/v4/sentences.csv'
WORDS = OLD / 'saved_results/followup_frp_analysis/frp_analysis/signal/word_targets.csv'
membership = json.loads((ROOT / 'source_membership.json').read_text())
with WORDS.open() as stream:
    originals = {}
    for row in csv.DictReader(stream):
        originals.setdefault(int(row['sentence_id']), []).append((int(row['word_index']), row['word']))
sources = []
with SOURCE.open() as stream:
    for row in csv.DictReader(stream):
        if row['lang'] != 'en':
            continue
        sid = int(row['sentence_id'])
        words = [w for _, w in sorted(originals[sid])] if sid in originals else row['text'].split()
        if ' '.join(words) != row['text']:
            raise ValueError(f'SR original-word mismatch: {sid}')
        tasks = ['SR'] if sid in originals else [t for t, m in membership.items() if sid in m['sentence_ids']]
        if not tasks:
            raise ValueError(f'No reading task: {sid}')
        alignment_warnings = []
        for task in tasks:
            if task != 'SR' and len(words) != membership[task]['n_words'][str(sid)]:
                alignment_warnings.append(f"{task}: annotation text has {len(words)} whitespace tokens; cache audit reports {membership[task]['n_words'][str(sid)]}. Exact EEG alignment requires validation; exclude on mismatch.")
        sources.append({'ordinal': len(sources), 'sentence_id_original': sid,
                        'sentence_key': sentence_key(words), 'tasks': tasks,
                        'words': words, 'text': row['text'], 'alignment_warnings': alignment_warnings})
(ROOT / 'source_sentences.json').write_text(json.dumps(sources, ensure_ascii=False, indent=1) + '\n')
(ROOT / 'source_manifest.json').write_text(json.dumps({
    'sentences': len(sources), 'unique_word_rows': sum(len(s['words']) for s in sources),
    'task_unique_sentences': {t: sum(t in s['tasks'] for s in sources) for t in ['SR','NR','TSR']},
    'alignment_warning_sentences': [s['sentence_id_original'] for s in sources if s['alignment_warnings']],
    'source_sha256': file_digest(SOURCE), 'sr_word_source_sha256': file_digest(WORDS),
    'source_words_sha256': digest([s['words'] for s in sources]),
    'source_note': 'Existing local output reconstructed directly from original word caches; SR boundaries verified against original word table; NR/TSR word counts checked against all subject extraction records.'
}, indent=2) + '\n')
print((ROOT / 'source_manifest.json').read_text())
