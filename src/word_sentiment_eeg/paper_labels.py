"""Conservative original-SST joins, independent of the project's agent labels.

No fuzzy matching, stemming, inferred sentiment, or silent punctuation merging.
The smallest containing phrase is exported as a candidate, not asserted to be
the exact phrase-label definition used by Pfeiffer's unavailable study scripts.
"""
from collections import defaultdict
import math
import re
import unicodedata
import zipfile


def canonical(text):
    text=unicodedata.normalize('NFKC',str(text))
    for old,new in [('-LRB-','('),('-RRB-',')'),('-LSB-','['),('-RSB-',']'),
                    ('-LCB-','{'),('-RCB-','}'),('``','"'),("''",'"'),
                    ('“','"'),('”','"'),('‘',"'"),('’',"'")]:
        text=text.replace(old,new)
    return re.sub(r'\s+','',text)


def five_class(score):
    score=float(score)
    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid original SST score')
    # Original CoreNLP converter: floor(score * 5). Explicitly handle the
    # theoretical closed upper endpoint as the very-positive class.
    return min(4,int(math.floor(score*5)))-2


def sentiment(label):
    return 'negative' if label<0 else 'positive' if label>0 else 'neutral'


class SST:
    def __init__(self,archive):
        with zipfile.ZipFile(archive) as z:
            def lines(name):
                paths=[p for p in z.namelist() if p.rsplit('/',1)[-1]==name and not p.startswith('__MACOSX/')]
                if len(paths)!=1:raise ValueError('Missing/ambiguous original SST file: '+name)
                if z.getinfo(paths[0]).file_size>50_000_000:raise ValueError('Unexpected SST member size')
                return z.read(paths[0]).decode('utf-8-sig').splitlines()
            self.scores={int(row.split('|')[0]):float(row.split('|')[1]) for row in lines('sentiment_labels.txt')[1:] if row}
            self.phrases={}
            for row in lines('dictionary.txt'):
                phrase,identifier=row.rsplit('|',1)
                key=tuple(phrase.split(' '));value=int(identifier)
                if key in self.phrases and self.phrases[key]!=value:raise ValueError('Ambiguous SST dictionary key')
                self.phrases[key]=value
            tokens=[row.split('|') for row in lines('SOStr.txt')]
            parents=[[int(v)-1 for v in row.split('|')] for row in lines('STree.txt')]
            if len(tokens)!=len(parents):raise ValueError('SST tokens/tree counts disagree')
        self.sentences=[dict(sst_sentence_index=i+1,tokens=t,parents=p) for i,(t,p) in enumerate(zip(tokens,parents))]
        self.index=defaultdict(list)
        for record in self.sentences:self.index[canonical(''.join(record['tokens']))].append(record)

    def lookup(self,tokens):
        escaped=tuple('-LRB-' if t=='(' else '-RRB-' if t==')' else t for t in tokens)
        key=escaped if escaped in self.phrases else tuple(tokens)
        identifier=self.phrases.get(key)
        if identifier is None or identifier not in self.scores:return None
        value=self.scores[identifier];label=five_class(value)
        return dict(phrase_id=identifier,score=value,label_5=label,sentiment=sentiment(label))

    def join(self,words):
        candidates=self.index.get(canonical(''.join(words)),[])
        if len(candidates)!=1:
            return dict(status='sentence_not_found' if not candidates else 'ambiguous_sentence',candidate_count=len(candidates),words=[])
        record=candidates[0];tokens=record['tokens']
        leaf_lengths=[len(canonical(t)) for t in tokens]
        offsets=[0]
        for n in leaf_lengths:offsets.append(offsets[-1]+n)
        node_spans=tree_spans(record['parents'],len(tokens))
        position=0;rows=[]
        for wi,word in enumerate(words):
            start=position;position+=len(canonical(word));end=position
            row=dict(word_index=wi,source_word=word)
            if start not in offsets or end not in offsets:
                rows.append(dict(row,status='source_word_splits_sst_token'));continue
            lo=offsets.index(start);hi=offsets.index(end)
            if hi<=lo:rows.append(dict(row,status='empty_word_span'));continue
            span=self.lookup(tokens[lo:hi])
            content=[i for i in range(lo,hi) if any(ch.isalnum() for ch in canonical(tokens[i]))]
            word_label=self.lookup([tokens[content[0]]]) if len(content)==1 else (self.lookup(tokens[lo:hi]) if hi-lo==1 else None)
            # Multiple lexical leaves (e.g. a split contraction) do not acquire
            # a fabricated word-alone label from their phrase score.
            containing=sorted((b-a,a,b) for a,b in node_spans if a<=lo and hi<=b and (a,b)!=(lo,hi))
            phrase=self.lookup(tokens[containing[0][1]:containing[0][2]]) if containing else None
            rows.append(dict(row,status='matched' if word_label else 'multi_leaf_or_missing_word_label',
                             sst_leaf_start=lo,sst_leaf_stop=hi,
                             word_label=word_label,target_span_label=span,
                             smallest_containing_phrase_label=phrase,
                             word_label_rule='single lexical leaf (punctuation-only companions permitted)'))
        return dict(status='matched',candidate_count=1,sst_sentence_index=record['sst_sentence_index'],
                    sentence_label=self.lookup(tokens),words=rows)


def tree_spans(parents,nleaves):
    if not 0<nleaves<=len(parents):raise ValueError('Invalid SST tree size')
    children=defaultdict(list)
    for i,parent in enumerate(parents):
        if parent>=len(parents) or parent==i or parent< -1:raise ValueError('Invalid parent pointer')
        if parent>=0:children[parent].append(i)
    roots=[i for i,p in enumerate(parents) if p==-1]
    if len(roots)!=1:raise ValueError('SST tree must have exactly one root')
    visiting=set();memo={}
    def visit(i):
        if i in visiting:raise ValueError('Cycle in SST tree')
        if i in memo:return memo[i]
        visiting.add(i)
        if i<nleaves:
            if children[i]:raise ValueError('Leaf has children')
            indices=[i]
        else:
            indices=sorted(j for child in children[i] for j in visit(child))
            if not indices:raise ValueError('Empty internal node')
        if indices!=list(range(indices[0],indices[-1]+1)):raise ValueError('Noncontiguous SST constituent')
        visiting.remove(i);memo[i]=indices;return indices
    if visit(roots[0])!=list(range(nleaves)) or len(memo)!=len(parents):raise ValueError('Disconnected SST tree')
    return sorted(set((v[0],v[-1]+1) for v in memo.values()))
