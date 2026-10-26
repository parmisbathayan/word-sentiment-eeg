"""Meaningful join/coverage/experiment invariants for three-level labels."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('three_level_compile', BASE/'annotations/three_level_compile.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for filename in ['word_labels.csv','sentence_context_labels.csv','source_sentence_ratings.csv','POLICY.md','sentence_context_POLICY.md','three_level_POLICY.md','three_level_compile.py']:
        (root/filename).write_bytes((BASE/'annotations'/filename).read_bytes())
    historical = {x:module.digest(root/x) for x in ['word_labels.csv','sentence_context_labels.csv','source_sentence_ratings.csv']}
    module.ROOT = root
    with contextlib.redirect_stdout(io.StringIO()):
        manifest = module.compile_labels()
    assert manifest['word_occurrences'] == 20458
    assert manifest['sentence_items'] == 1037
    assert manifest['new_full_sentiment_rereview_sentences_this_revision'] == 0
    assert all(module.digest(root/name)==sha for name,sha in historical.items())
    with (root/'word_labels_three_level.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    identity = {(int(x['sentence_ordinal']),int(x['word_index'])):x for x in rows}
    assert identity[(44,1)]['grammatical_class']=='content'  # I like...
    assert identity[(4,5)]['grammatical_class']=='function'  # comparison like...
    assert identity[(423,2)]['grammatical_class']=='content'  # May date
    assert identity[(57,1)]['grammatical_class']=='function'  # modal may
    assert identity[(482,4)]['grammatical_class']=='content'  # Will Robinson
    assert identity[(122,1)]['grammatical_class']=='content'  # well acted
    assert identity[(427,6)]['grammatical_class']=='function'  # as well as
    for row in rows:
        assert row['contextual_word_class']==row['contextual_class']
        if row['eligible_all_three']=='true':
            assert row['lexical_class']==row['contextual_word_class']==row['sentence_context_class']
            assert row['sentence_context_class'] in module.SENTIMENT
        if row['eligible_content_all_three']=='true':
            assert row['eligible_all_three']=='true' and row['grammatical_class']=='content'
        if row['grammatical_class']!='content':
            assert all(row['eligible_content_'+name]=='false' for name in ['all_three','lexical_only','contextual_only','sentence_only'])
        source = json.loads(row['source_sentence_class_by_task'])
        assert set(source)==set(row['tasks'].split(';'))
        assert all(value=='unknown' for task,value in source.items() if task!='SR')
    assert any(x['eligible_lexical_only']=='true' and x['eligible_all_three']=='false' for x in rows)
    # A missing sentence is an error, never a neutral fallback.
    path = root/'sentence_context_labels.csv'
    lines=path.read_text().splitlines(); path.write_text('\n'.join(lines[:-1])+'\n')
    try:
        module.compile_labels()
    except ValueError as error:
        assert 'coverage mismatch' in str(error)
    else:
        raise AssertionError('Missing sentence context accepted')
print('three-level label join, provenance, grammar examples and experiment invariants passed')
