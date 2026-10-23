"""Preserve inherited original SR sentence ratings from the existing corpus export.

No ratings are inferred from word/context labels. NR/TSR remain unrated.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent


def prepare(corpus):
    corpus = Path(corpus)
    source = json.loads((BASE / 'source_sentences.json').read_text())
    manifest = json.loads((BASE / 'source_manifest.json').read_text())
    sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    if sha != manifest['source_sha256']:
        raise ValueError('Corpus differs from the already verified annotation text source')
    with corpus.open() as handle:
        exports = [r for r in csv.DictReader(handle) if r['lang'] == 'en']
    by_id = {}
    for row in exports:
        sid = int(row['sentence_id'])
        if sid in by_id:
            raise ValueError('Nonunique English source sentence ID')
        by_id[sid] = row
    ratings = []
    for sentence in source:
        if 'SR' not in sentence['tasks']:
            continue
        row = by_id[sentence['sentence_id_original']]
        if row['text'] != sentence['text'] or row['text'] != ' '.join(sentence['words']):
            raise ValueError('SR sentence text/word boundary differs from original export')
        mapped = int(row['label'])
        if mapped not in (0, 1, 2):
            raise ValueError('Unrated SR sentence')
        ratings.append({'sentence_key': sentence['sentence_key'], 'task': 'SR',
                        'sentence_id_original': sentence['sentence_id_original'],
                        'source_sentence_class': ('negative', 'neutral', 'positive')[mapped],
                        'original_label': mapped - 1, 'saved_corpus_mapped_label': mapped,
                        'rating_provenance': 'original_sentence_rating_preserved_in_existing_corpus_export'})
    if len(ratings) != 400:
        raise ValueError('Expected exactly 400 original SR sentence ratings')
    output = BASE / 'source_sentence_ratings.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ratings[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(ratings)
    provenance = {
        'status': 'preserved_existing_original_source_ratings', 'sentence_count': len(ratings),
        'class_counts': dict(Counter(r['source_sentence_class'] for r in ratings)),
        'source': 'parmis-thesis-eeg-sentiment-review/saved_results/eeg_to_text_v4_continuous/v4/sentences.csv',
        'source_sha256': sha, 'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'join_validation': 'All 400 English SR IDs, sentence texts and source word boundaries match exactly',
        'label_mapping': 'Old src/config.py LABEL_TO_ID={-1:0,0:1,1:2}; old src/eeg2text/data.py maps original trial labels to saved corpus labels',
        'scope': 'Original overall sentence/review judgment; not strict affect-free context, not word polarity, not measured reader emotion',
        'not_freshly_downloaded_or_human_readjudicated': True,
        'nr_tsr_rating': 'unknown; never inherited from SR or assumed neutral',
    }
    (BASE / 'source_sentence_ratings_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance['class_counts']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=BASE.parents[1] / 'parmis-thesis-eeg-sentiment-review/saved_results/eeg_to_text_v4_continuous/v4/sentences.csv')
    args = parser.parse_args()
    prepare(args.corpus)
