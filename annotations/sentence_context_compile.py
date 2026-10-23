"""Validate and expand directly reviewed ZuCo sentence-context decisions.

This compiler supplies no inferred/default labels. Every source ordinal must
have exactly one explicit decision in sentence_context_decisions.tsv.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
CLASSES = {'positive', 'negative', 'neutral', 'mixed', 'uncertain'}


def compile_context(base=BASE):
    base = Path(base)
    source_path = base / 'source_sentences.json'
    decision_path = base / 'sentence_context_decisions.tsv'
    source = json.loads(source_path.read_text())
    decisions = {}
    for line in decision_path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) != 3 or not parts[1].startswith('class=') or not parts[2].startswith('note='):
            raise ValueError(f'Malformed explicit context decision: {line}')
        ordinal, cls, note = int(parts[0]), parts[1][6:], parts[2][5:]
        if ordinal in decisions or cls not in CLASSES or not note.strip():
            raise ValueError(f'Duplicate/invalid decision: {ordinal}')
        decisions[ordinal] = (cls, note)
    expected = {s['ordinal'] for s in source}
    if set(decisions) != expected or len(source) != len(expected):
        raise ValueError('Every source sentence needs exactly one explicit context decision')
    if len({s['sentence_key'] for s in source}) != len(source):
        raise ValueError('Nonunique source sentence keys')
    rows = []
    for s in source:
        cls, note = decisions[s['ordinal']]
        rows.append({'sentence_key': s['sentence_key'], 'sentence_context_class': cls,
                     'sentence_context_note': note, 'annotation_status': 'agent_reviewed_not_human_gold'})
    output = base / 'sentence_context_labels.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        'status': 'all_source_sentences_explicitly_reviewed',
        'annotation_status': 'agent_reviewed_not_human_gold',
        'not_human_consensus': True,
        'sentence_count': len(source),
        'counts': dict(Counter(row['sentence_context_class'] for row in rows)),
        'task_counts': {task: dict(Counter(decisions[s['ordinal']][0] for s in source if task in s['tasks']))
                        for task in ('SR', 'NR', 'TSR')},
        'policy': 'sentence_context_POLICY.md',
        'policy_sha256': hashlib.sha256((base / 'sentence_context_POLICY.md').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'decisions_sha256': hashlib.sha256(decision_path.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'scope': 'Affective/evaluative context of the supplied sentence; not reviewer recommendation, discourse sentiment, or measured reader emotion',
        'coverage_note': 'Task counts overlap where a source sentence is shared by tasks',
    }
    (base / 'sentence_context_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['counts'], sort_keys=True))
    return rows


if __name__ == '__main__':
    compile_context()
