"""Expand explicit sentence-review decisions into every source token. No lexicon."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from word_sentiment_eeg.common import normalize_word, file_digest, write_json

ROOT = Path(__file__).resolve().parent
CLASS = {'p': 'positive', 'n': 'negative', 'u': 'uncertain', 'lp': 'positive', 'ln': 'negative', 'l0': 'neutral', 'lu': 'uncertain'}


def compile_reviews(partial=False):
    sentences = json.loads((ROOT / 'source_sentences.json').read_text())
    by_id = {s['ordinal']: s for s in sentences}
    decisions = {}
    for line in (ROOT / 'agent_decisions.tsv').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split('|')
        ordinal = int(fields[0])
        if ordinal in decisions or ordinal not in by_id:
            raise ValueError(f'Duplicate or unknown sentence review {ordinal}')
        record = dict(f.split('=', 1) for f in fields[1:])
        if not set(record) <= set(CLASS) | {'flags', 'note'}:
            raise ValueError(f'Unknown review field {ordinal}: {record}')
        decisions[ordinal] = record
    missing = sorted(set(by_id) - set(decisions))
    if missing and not partial:
        raise ValueError(f'{len(missing)} sentences unreviewed; first missing: {missing[:10]}')
    rows = []
    for ordinal, review in sorted(decisions.items()):
        source = by_id[ordinal]
        words = source['words']
        contextual = ['neutral'] * len(words)  # only after the sentence was explicitly reviewed
        lexical_overrides = {}
        used = set()
        for field, label in CLASS.items():
            for selector in review.get(field, '').split():
                token, _, position = selector.partition('@')
                matches = [i for i, word in enumerate(words) if normalize_word(word) == normalize_word(token)]
                if position:
                    matches = [i for i in matches if i == int(position)]
                if not matches:
                    raise ValueError(f'Review {ordinal}: token selector {selector!r} not found in {words}')
                for index in matches:
                    layer = 'lexical' if field.startswith('l') else 'contextual'
                    if (layer, index) in used:
                        raise ValueError(f'Review {ordinal}: conflicting {layer} labels for {words[index]}')
                    used.add((layer, index))
                    if layer == 'lexical':
                        lexical_overrides[index] = label
                    else:
                        contextual[index] = label
        for i, word in enumerate(words):
            rows.append({
                'sentence_ordinal': ordinal, 'sentence_id_original': source['sentence_id_original'],
                'sentence_key': source['sentence_key'], 'tasks': ';'.join(source['tasks']),
                'word_index': i, 'word': word, 'word_form': normalize_word(word), 'sentence': source['text'],
                'lexical_class': lexical_overrides.get(i, contextual[i]), 'contextual_class': contextual[i],
                'annotation_flags': review.get('flags', ''), 'annotation_note': review.get('note', ''),
                'alignment_warning': '; '.join(source['alignment_warnings']),
                'annotation_method': 'single_agent_sentence_and_token_review',
                'annotator': 'Codex', 'policy_version': 'v2',
                'annotation_status': 'agent_reviewed_not_human_gold',
            })
    output = ROOT / ('word_labels.PARTIAL.csv' if missing else 'word_labels.csv')
    if rows:
        with output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        if not missing:
            for name, subset in [
                ('uncertain_words.csv', [r for r in rows if 'uncertain' in (r['lexical_class'],r['contextual_class'])]),
                ('lexical_context_differences.csv', [r for r in rows if r['lexical_class'] != r['contextual_class']])]:
                with (ROOT / name).open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(subset)
    summary = {'status': 'partial' if missing else 'complete', 'reviewed_sentences': len(decisions),
               'total_sentences': len(sentences), 'reviewed_word_rows': len(rows),
               'missing_sentence_ordinals': missing, 'output': output.name,
               'source_sha256': file_digest(ROOT / 'source_sentences.json'),
               'decision_sha256': file_digest(ROOT / 'agent_decisions.tsv'),
               'policy_sha256': file_digest(ROOT / 'POLICY.md'),
               'task_unique_sentences': {t: sum(t in s['tasks'] for s in sentences) for t in ['SR','NR','TSR']},
               'task_contextual_word_counts': {t: {label: sum(t in r['tasks'].split(';') and r['contextual_class'] == label for r in rows)
                                                     for label in ['positive','negative','neutral','uncertain']}
                                               for t in ['SR','NR','TSR']},
               'labels': {layer: {label: sum(r[layer] == label for r in rows)
                                  for label in ['positive','negative','neutral','uncertain']}
                          for layer in ['lexical_class','contextual_class']}}
    write_json(ROOT / 'annotation_manifest.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'missing_sentence_ordinals'}, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--partial', action='store_true')
    compile_reviews(p.parse_args().partial)
