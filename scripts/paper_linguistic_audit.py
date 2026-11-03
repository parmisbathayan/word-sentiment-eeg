#!/usr/bin/env python3
"""Cloud-only audit of original ZuCo words/bounds and original SST labels.

Downloads only small text/bounds/ET files and the official ~6-MB SST ZIP.
No EEG downloads, models, re-extraction, Drive writes or agent relabeling.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.io import loadmat
from word_sentiment_eeg.paper_labels import SST
from paper_source_audit import request,fetch_sample
from paper_sst_source import acquire_sst

SOURCES=[dict(name='sentencesSR.mat',bytes=39657,download='https://osf.io/download/qdtvz/'),
         dict(name='wordbounds_SNR1_ZAB.mat',bytes=34529,download='https://osf.io/download/hg38s/'),
         dict(name='ZAB_SR1_ET.mat',bytes=813937,download='https://osf.io/download/h7d4s/'),
         dict(name='sentiment_normal_reading.csv',bytes=67617,download='https://osf.io/download/f2z4u/')]
SST_URL='https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'


def main():
    if sys.platform=='darwin':raise SystemExit('Cloud only: do not download datasets to the Mac')
    report={'purpose':'Original-source linguistic alignment audit, not reproduction results',
            'sources':[],'sst_source':SST_URL}
    with tempfile.TemporaryDirectory(prefix='paper-words-') as tmp:
        folder=Path(tmp);paths={}
        for source in SOURCES:
            path,sha=fetch_sample(source,folder);paths[source['name']]=path
            report['sources'].append(dict(**source,sha256=sha))
        print('Download official SST text/labels (no model)',flush=True)
        zip_path,report['sst_transport']=acquire_sst(folder)
        sst=SST(zip_path)
        report['sst_inventory']=dict(sentences=len(sst.sentences),phrases=len(sst.phrases),scores=len(sst.scores))
        sentences=loadmat(paths['sentencesSR.mat'],squeeze_me=True,struct_as_record=False)
        report['sentence_file_keys']=[k for k in sentences if not k.startswith('__')]
        text=[str(s) for s in np.asarray(sentences['sentences']).reshape(-1)]
        bounds=loadmat(paths['wordbounds_SNR1_ZAB.mat'],squeeze_me=True,struct_as_record=False)
        cells=np.asarray(bounds['wordbounds'],dtype=object).reshape(-1)
        sizes=[int(np.asarray(c).reshape(-1,4).shape[0]) for c in cells]
        report['word_bounds']=dict(entries=len(cells),first_12_row_counts=sizes[:12],
                                   first_rectangle=np.asarray(cells[0]).reshape(-1,4)[0].tolist())
        # Native firstLevel_SR.m drops five practice sentences, then pairs bounds
        # by order. Report count agreement before relying on that rule.
        report['sentences']=dict(total=len(text),first_12_word_counts=[len(t.split()) for t in text[:12]])
        report['bounds_count_alignment_candidates']=[]
        for offset in range(0,11):
            n=min(250,len(cells),len(text)-offset)
            matches=sum(sizes[i]==len(text[i+offset].split()) for i in range(n))
            report['bounds_count_alignment_candidates'].append(dict(sentence_offset=offset,compared=n,matching_word_counts=matches))
        sentence_status=Counter();word_status=Counter();labels=Counter();multi=0
        for sentence in text[5:405]:
            joined=sst.join(sentence.split());sentence_status[joined['status']]+=1
            for word in joined['words']:
                word_status[word['status']]+=1
                if word.get('word_label'):labels[word['word_label']['sentiment']]+=1
                if word.get('sst_leaf_stop',0)-word.get('sst_leaf_start',0)>1:multi+=1
        report['sst_join']=dict(scope='Native sentence entries 5:405, following corpus practice removal',
                                sentence_status=dict(sentence_status),word_status=dict(word_status),
                                original_word_label_counts=dict(labels),multi_leaf_source_words=multi,
                                note='Counts are source occurrences, not reader EEG trials. Unmatched/ambiguous rows have no invented label.')
        with paths['sentiment_normal_reading.csv'].open(encoding='utf-8-sig',newline='') as handle:
            sample=handle.read(2048);handle.seek(0)
            dialect=csv.Sniffer().sniff(sample,delimiters=',;\t')
            rows=list(csv.reader(handle,dialect=dialect))
        report['experiment_csv']=dict(rows=len(rows)-1,columns=rows[0])
        eye=loadmat(paths['ZAB_SR1_ET.mat'],squeeze_me=True,struct_as_record=False)
        report['eye_event_headers']={name:[str(v) for v in np.asarray(getattr(eye['eyeevent'],name).colheader).reshape(-1)]
                                     for name in ['fixations','saccades','blinks']}
        triggers=np.asarray(eye['event'])
        report['eye_trigger_preview']=triggers[:8].tolist()
        report['eye_trigger_counts']={str(k):int(v) for k,v in Counter(triggers[:,1]).items()}
        # Only short event/message templates and numeric IDs; no stimulus text.
        messages=[str(v) for v in np.asarray(eye['messages']).reshape(-1)]
        templates=Counter();id_examples=[]
        for message in messages:
            parts=message.split()
            templates[' '.join(re.sub(r'\d+','#',p) for p in parts[:3])]+=1
            if re.search(r'TRIALID|SENTENCE_ID|SENTENCEID|TRIAL_ID',message,re.I):
                id_examples.append(' '.join(parts[:5])[:100])
        report['eye_message_templates']=templates.most_common(20)
        report['eye_message_id_examples']=id_examples[:12]
    report['temporary_sources_removed']=True
    target=ROOT/'artifacts/paper_linguistic_audit.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
