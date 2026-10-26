"""Join reviewed sentiment layers without changing historical annotations.

Standard library only. Grammar screening is deliberately conservative and is
not advertised as a complete POS annotation or a new full sentiment rereview.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SENTIMENT = {'positive', 'negative', 'neutral'}
FUNCTION = set('''a an the and or but nor yet so if because although though whereas while unless whether than that which who whom whose what whatever whichever whoever where when whenever wherever why how as to of in on at by for from with without into onto upon through throughout within during until till between among against despite except per via versus vs is am are was were be been being i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves this these those such each every either neither both any some all another other else no not never here there hence thus therefore then'''.split())
# These forms have common lexical/auxiliary, particle/adverb/adjective or
# determiner/quantifier uses. Keep them uncertain unless individually reviewed.
AMBIGUOUS = set('''have has had having do does did done doing get gets got getting may will can must like well after before since over under about around near up down out off back still just only more most less least much many few little one first second last next own same enough rather once also ever even together along above below beyond across outside inside instead too away'''.split())
CONTRACTIONS = set("i'm you're he's she's it's we're they're i've you've we've they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll isn't aren't wasn't weren't hasn't haven't hadn't doesn't don't didn't can't couldn't shouldn't wouldn't won't mustn't there's that's what's who's".split())
MANUAL_CONTENT = {(44,1), (290,21), (321,3), (226,17), (48,4), (122,1), (342,5), (482,4), (483,16)}
MANUAL_FORMS = {'like', 'well', 'may', 'will', 'can', 'must'}


def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grammar(row):
    form = row['word_form']
    identity = (int(row['sentence_ordinal']), int(row['word_index']))
    if form in MANUAL_FORMS:
        if identity in MANUAL_CONTENT or (form == 'may' and row['word'].strip('("\'').startswith('May')):
            return 'content', 'targeted_agent_context_review', 'Lexical verb/adverb or named entity/date in this occurrence.'
        return 'function', 'targeted_agent_context_review', 'Comparison/conjunction, modal, or discourse construction in this occurrence.'
    if not any(c.isalpha() for c in form):
        return 'uncertain', 'conservative_grammar_screen', 'Numeric/punctuation/nonalphabetic token; excluded from content-only primary analysis.'
    if form in FUNCTION or form in CONTRACTIONS:
        return 'function', 'closed_class_screen', 'Conservative function-word list; includes negators and auxiliary/copular be.'
    if form in AMBIGUOUS:
        return 'uncertain', 'conservative_grammar_screen', 'Lexical versus grammatical role not individually resolved in this revision.'
    return 'content', 'open_class_candidate_screen', 'Open-class candidate by exclusion; not a full contextual POS annotation.'


def compile_labels():
    original = rows(ROOT / 'word_labels.csv')
    contexts = rows(ROOT / 'sentence_context_labels.csv')
    context_by_key = {x['sentence_key']: x for x in contexts}
    if len(context_by_key) != len(contexts):
        raise ValueError('Duplicate sentence-context key')
    ratings = rows(ROOT / 'source_sentence_ratings.csv')
    source_by_key = {}
    for row in ratings:
        key = (row['sentence_key'], row['task'])
        if key in source_by_key:
            raise ValueError('Duplicate source rating key')
        source_by_key[key] = row
    if {x['sentence_key'] for x in original} != set(context_by_key):
        raise ValueError('Word/context coverage mismatch')
    identities = [(x['sentence_key'], x['word_index']) for x in original]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate word identity')
    output = []
    for old in original:
        if old['annotation_status'] != 'agent_reviewed_not_human_gold':
            raise ValueError('Word annotation not reviewed')
        sentence = context_by_key[old['sentence_key']]
        if sentence['annotation_status'] != 'agent_reviewed_not_human_gold':
            raise ValueError('Sentence context not reviewed')
        if old['lexical_class'] not in SENTIMENT | {'uncertain'} or old['contextual_class'] not in SENTIMENT | {'uncertain'}:
            raise ValueError('Invalid word class')
        if sentence['sentence_context_class'] not in SENTIMENT | {'mixed','uncertain'}:
            raise ValueError('Invalid sentence class')
        new = dict(old)
        new['contextual_word_class'] = old['contextual_class']
        new['sentence_context_class'] = sentence['sentence_context_class']
        new['sentence_context_note'] = sentence['sentence_context_note']
        new['source_sentence_class_by_task'] = json.dumps({task: source_by_key.get((old['sentence_key'], task), {}).get('source_sentence_class','unknown') for task in old['tasks'].split(';')}, sort_keys=True)
        new['source_sentence_original_label_by_task'] = json.dumps({task: source_by_key.get((old['sentence_key'], task), {}).get('original_label','unknown') for task in old['tasks'].split(';')}, sort_keys=True)
        new['source_sentence_rating_provenance_by_task'] = json.dumps({task: source_by_key.get((old['sentence_key'], task), {}).get('rating_provenance','unavailable') for task in old['tasks'].split(';')}, sort_keys=True)
        new['grammatical_class'], new['grammatical_annotation_method'], new['grammatical_note'] = grammar(old)
        new['three_level_review_status'] = 'inherited_complete_sentiment_reviews_with_targeted_grammar_audit'
        new['three_level_policy_version'] = 'english_three_level_v1'
        valid = not old['alignment_warning']
        content = new['grammatical_class'] == 'content'
        lexical, contextual, whole = old['lexical_class'], old['contextual_class'], new['sentence_context_class']
        # Independent only means the chosen level supplies the class; it does
        # not require disagreement with the other two levels.
        for name, label in [('lexical_only',lexical),('contextual_only',contextual),('sentence_only',whole)]:
            new['eligible_'+name] = str(valid and label in SENTIMENT).lower()
            new['eligible_content_'+name] = str(valid and content and label in SENTIMENT).lower()
        agree = valid and lexical in SENTIMENT and lexical == contextual == whole
        new['eligible_all_three'] = str(agree).lower()
        new['eligible_content_all_three'] = str(agree and content).lower()
        output.append(new)
    output_path = ROOT / 'word_labels_three_level.csv'
    with output_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(output[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(output)
    count_fields = ['lexical_class','contextual_word_class','sentence_context_class','grammatical_class','grammatical_annotation_method']
    experiment_counts = {}
    for task in ['SR','NR','TSR']:
        subset = [x for x in output if task in x['tasks'].split(';')]
        experiment_counts[task] = {}
        for experiment in ['all_three','lexical_only','contextual_only','sentence_only']:
            label_field = {'all_three':'lexical_class','lexical_only':'lexical_class','contextual_only':'contextual_word_class','sentence_only':'sentence_context_class'}[experiment]
            experiment_counts[task][experiment] = {}
            for prefix in ['', 'content_']:
                eligible = [x for x in subset if x['eligible_'+prefix+experiment]=='true']
                experiment_counts[task][experiment][prefix+'candidate_words'] = {label:sum(x[label_field]==label for x in eligible) for label in sorted(SENTIMENT)}
    manifest = {
        'status':'complete_join_not_new_full_rereview', 'dataset':'ZuCo 1.0 English',
        'sentence_items':len(contexts), 'word_occurrences':len(output),
        'inherited_sentiment_review_sentences':len(contexts),
        'new_full_sentiment_rereview_sentences_this_revision':0,
        'targeted_grammar_review_forms':sorted(MANUAL_FORMS),
        'targeted_grammar_review_word_occurrences':sum(x['grammatical_annotation_method']=='targeted_agent_context_review' for x in output),
        'sentiment_changes_from_historical_word_labels':0,
        'new_independent_human_validation':False,
        'grammar_limitations':'Open-class candidate screening is not complete POS gold. Ambiguous unresolved roles explicitly uncertain. Function/content assignment is distinct from sentiment.',
        'counts':{field:dict(Counter(x[field] for x in output)) for field in count_fields},
        'task_experiment_source_word_counts':experiment_counts,
        'source_hashes':{name:digest(ROOT/name) for name in ['word_labels.csv','sentence_context_labels.csv','source_sentence_ratings.csv','POLICY.md','sentence_context_POLICY.md','three_level_POLICY.md','three_level_compile.py']},
        'output':output_path.name,'output_sha256':digest(output_path),
        'no_missing_labels_imputed':True,'original_files_modified':False,
        'count_unit':'Unique source word positions by task, before reader availability/fixation/EEG rejection; not accepted EEG counts.',
    }
    (ROOT/'three_level_review_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['status','sentence_items','word_occurrences','targeted_grammar_review_word_occurrences','counts','task_experiment_source_word_counts']},indent=2))
    return manifest


if __name__ == '__main__':
    compile_labels()
