"""Proof-bound clock recovery; old mapping restored and unverified data refused."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from word_sentiment_eeg.common import file_digest
from word_sentiment_eeg.paper_python import synchronize
from word_sentiment_eeg.paper_clock_audit import fit_candidates
from word_sentiment_eeg.paper_verified_recovery import verified_mapping,approved_clock
x=np.arange(100)*5000.+1e6;y=(x-1e6)*.001+.5;codes=np.tile([10,11],50)
ybad=y.copy();ybad[55]+=.060;samples=ybad*500+1;eye=np.c_[x,codes]
clock=fit_candidates(x,ybad)['theil_sen_candidate']
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);items={}
    for kind in ['eeg','eye','corrected']:
        p=root/(kind+'.mat');p.write_bytes(('SYNTHETIC '+kind).encode());items[kind]={'id':kind,'name':p.name}
    proof=dict(subject='TEST',block='1',candidate_clocks={'theil_sen_candidate':clock},
        source_checksums={k:file_digest(root/v['name']) for k,v in items.items()},
        validation={'theil_sen_candidate':dict(candidate_passes_audit=True,checked_fixations=120,checked_sentences=12,p99_error_ms=1.,maximum_error_ms=1.5)})
    module=SimpleNamespace(synchronize=synchronize)
    def fetch(item,folder):return root/item['name']
    def mapper(*args,**kwargs):return module.synchronize(codes,samples,eye,500,max_error_ms=8.)
    module.fetch=fetch;module.map_block=mapper
    inventory=[dict(block_id='1',available=True,**items)]
    try:mapper('TEST','1')
    except ValueError:pass
    else:raise AssertionError('Old maximum-trigger rule must reject the synthetic block')
    # An unrelated approved block may be omitted by an explicit pilot subset.
    unrelated=copy.deepcopy(proof);unrelated['block']='2'
    with verified_mapping(module,'TEST',inventory,[proof,unrelated]):
        try:module.map_block('TEST','1')
        except ValueError:pass
        else:raise AssertionError('Unverified source files must not be accepted')
        for item in items.values():module.fetch(item,root)
        result=module.map_block('TEST','1')
        assert result['acceptance']=='independent_word_waveform_validation'
        assert result['max_residual_ms']>50,'Do not hide the bad trigger'
        assert result['candidate_method']=='theil_sen_candidate'
    assert module.map_block is mapper and module.fetch is fetch and module.synchronize is synchronize
    # Scientific criteria cannot be bypassed with a lone successful boolean.
    bad=copy.deepcopy(proof);bad['validation']['theil_sen_candidate']['p99_error_ms']=3
    assert approved_clock(bad) is None
    # Native source tampering fails immediately, before waveform/time mapping.
    (root/'eye.mat').write_bytes(b'CHANGED')
    try:
        with verified_mapping(module,'TEST',inventory,[proof]):module.fetch(items['eye'],root)
    except ValueError:pass
    else:raise AssertionError('Changed native sources must be refused')
    assert module.fetch is fetch and module.map_block is mapper
print('PASS: validated clock recovery requires exact files, preserves failed markers, checks numeric evidence, and restores original extraction functions')
