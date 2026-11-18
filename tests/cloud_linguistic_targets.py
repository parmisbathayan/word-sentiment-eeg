"""Synthetic plus real Persian NLP smoke checks, all in cloud; no private data."""
from pathlib import Path
import sys,tempfile,json
from types import SimpleNamespace as Obj
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import linguistic_targets as l

if '--real-nlp' in sys.argv:
    # Download only PUBLIC NLP models on the cloud runner. No EEG or private text.
    with tempfile.TemporaryDirectory() as tmp:
        source=pd.DataFrame([dict(sentence_key='sample',word_index=i,word=w) for i,w in enumerate(['این','یک','کتاب','خوب','است','.'])]);source['normalized_word']=source.word.map(l.normalize)
        labels=l.linguistic_labels(source,l.ImmutableStore(Path(tmp)/'out',50_000_000),Path(tmp)/'models')
        assert labels.annotation_status.eq('automatic_stanza_wordnet').all()
        assert labels.upos.isin(['NOUN','VERB','ADJ']).any(),labels.to_string()
        assert labels.grammatical_class.eq('function').any()
        print(labels.to_string(index=False))
        # Verify public Persian noun mapping is present and labels/cache can resume without downloads.
        with patch.dict(sys.modules,{'stanza':None}):again=l.linguistic_labels(source,l.ImmutableStore(Path(tmp)/'out',50_000_000),Path(tmp)/'models')
        assert labels.fillna('').equals(again.fillna(''))
    print('Real Persian Stanza pretokenized POS/lemma plus OMW smoke check passed.');raise SystemExit

class Synset:
    def __init__(self,name,root):self.n=name;self.root=root
    def name(self):return self.n
    def hypernym_paths(self):return [[Obj(name=lambda:self.root),self]]
class WN:
    NOUN='n'
    def synsets(self,form,**kwargs):
        if form=='کتاب':return [Synset('book.n.01','physical_entity.n.01')]
        if form=='آزادی':return [Synset('freedom.n.01','abstraction.n.06')]
        if form=='چندمعنی':return [Synset('a','physical_entity.n.01'),Synset('b','abstraction.n.06')]
        return []
wn=WN()
assert l.noun_semantics('کتاب','کتاب',wn)[0]=='physical'
assert l.noun_semantics('آزادی','آزادی',wn)[0]=='abstract'
assert l.noun_semantics('چندمعنی','چندمعنی',wn)[0]=='unknown'
assert l.noun_semantics('missing','missing',wn)[0]=='unknown'
source=pd.DataFrame([dict(sentence_key='S0',word_index=i,word=w,normalized_word=l.normalize(w)) for i,w in enumerate(['کتاب','آزادی','است'])])
def factory():
    def nlp(sentences):
        tags=['NOUN','NOUN','AUX'];return Obj(sentences=[Obj(words=[Obj(text=w,upos=tags[i],lemma=w) for i,w in enumerate(sentences[0])])])
    return nlp
with tempfile.TemporaryDirectory() as tmp:
    store=l.ImmutableStore(Path(tmp),50_000_000)
    labels=l.linguistic_labels(source,store,Path(tmp)/'models',factory,wn)
    assert labels.semantic_class.tolist()==['physical','abstract','unknown']
    def forbidden():raise AssertionError('Reannotated completed labels')
    again=l.linguistic_labels(source,store,Path(tmp)/'models',forbidden,wn)
    assert labels.equals(again)
    meta=source.drop(columns='normalized_word').copy();meta['grammatical_class']='old';meta['annotation_status']='old'
    merged=l.attach(meta,labels);assert merged.nlp_valid.all() and merged.grammar_target.tolist()==['content','content','function']
    altered=meta.copy();altered.loc[0,'word']='different';assert not l.attach(altered,labels).nlp_valid.iloc[0]
# Source context includes unfixated words; duplicate text mismatches are excluded.
raw=pd.DataFrame([dict(dataset='teco',task='task1',subject='R',sentence_key='S0',sentence_position=0,word_index=i,word=w,event_id=f'e{i}',status='unfixated' if i==0 else 'archived') for i,w in enumerate(['کتاب','است','کتاب'])])
src,bad=l.source_tokens(raw);assert len(src)==3 and not bad
repeat=l.repetition_labels(raw);assert repeat['e0']=='first' and repeat['e2']=='repeat','Unfixated first presentation was ignored'
# Do not purge unrelated recordings that share sentence ordinal/onset numbers.
rng=np.random.default_rng(7)
meta=pd.DataFrame([dict(dataset='zuco',task=task,subject=f'R{r}',sentence_key=f'{task}_S{j//4}',sentence_position=j//4,word_index=j%4,word_identity=f'w{j%12}',word=f'w{j%12}',event_id=f'{task}-{r}-{j}',onset_sample_sentence_0based=(j%4)*1000) for task in ['SR','NR','TSR'] for r in range(12) for j in range(48)])
y=pd.Categorical(meta.task,categories=['NR','SR','TSR']).codes.astype(int)
for mode in ['familiar','reader','sentence']:
    for tr,te,audit in l.safe_splits(meta,y,np.arange(len(meta)),mode,3,18):
        assert not set(tr)&set(te)
        if mode=='familiar':assert all(v==1 for v in audit['test_seen_fraction'].values())
# Whole-word groups hold every occurrence of the same identity together.
py=np.where(meta.word_index%2==0,0,1)
for tr,te,_ in l.safe_splits(meta,py,np.arange(len(meta)),'word',3,18):assert not set(meta.word_identity.iloc[tr])&set(meta.word_identity.iloc[te])
# All new tasks, coverage gates and fixed vocabulary selection.
meta['upos']=np.array(['NOUN','VERB','ADJ','NOUN'])[meta.word_index];meta['nlp_valid']=True
meta['grammar_target']=np.where(meta.word_index%2,'function','content');meta['semantic_class']=np.where(meta.word_index%2,'abstract','physical')
post=rng.normal(size=(len(meta),48)).astype(np.float32);pre=rng.normal(size=(len(meta),12)).astype(np.float32)
features=l.t.regional_inputs(post,pre);features['metadata_control']=rng.normal(size=(len(meta),4))
src=pd.DataFrame([dict(word=f'w{i%12}') for i in range(120)])
for task in ['pos','content_function','semantic_nouns','word_identity5','word_identity10','zuco_reading_task']:
    rows,yy,xx,classes,plan,definition=l.target_data(task,meta,features,meta,src)
    assert len(rows)>0 and len(xx['regional_bins'])==len(rows)
    if task.startswith('word_identity'):assert 'word' not in plan['modes'] and len(classes)==int(task.removeprefix('word_identity'))
# Training-only cap is before preprocessing, all heldout examples retained.
tr,te,_=l.safe_splits(meta,y,np.arange(len(meta)),'reader',3,18)[0]
groups=(l.t.membership(meta,y,tr,1,2),l.t.membership(meta,y,te,1,3))
with patch.object(l,'CAP',40):
    result=l.limited_score(meta,y,post,groups,'lr_1',3,True)
assert result['actual_fit_examples']==40 and result['test_examples']==len(te)
with tempfile.TemporaryDirectory() as tmp:
    store=l.ImmutableStore(Path(tmp),50_000_000);small={k:features[k] for k in ['regional_bins','pre_control','metadata_control']}
    rows,dev,missing=l.t.evaluate(meta,y,small,['NR','SR','TSR'],dict(modes=['reader'],sizes=[1]),store,'zuco_reading_task',models=['lr_1','lda'],splitter=l.safe_splits,scorer=l.limited_score)
    assert len(rows)==15 and all(r['status']=='completed' for r in rows)
    summary=l.t.summarize(rows);assert summary.completed_folds.eq(3).all();l.t.figures(store,summary,rows)
print('Linguistic labels, ambiguous semantics, exact token alignment, full-source repetition, vocabulary, task grouping and bounded evaluator passed.')
