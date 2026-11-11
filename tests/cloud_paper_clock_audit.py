"""Synthetic tests for clock audit independence and conservative correspondence."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from word_sentiment_eeg.paper_clock_audit import anchors,fit_candidates,errors,reading_intervals,summarize_validation
from audit_paper_clock_recovery import validate_block

x=np.arange(100)*5000.+1e6;y=(x-1e6)*.001+.5;bad=y.copy();bad[55]+=.060
fit=fit_candidates(x,bad)
assert fit['original_affine']['anchor_max_error_ms']>8
assert fit['theil_sen_candidate']['anchor_max_error_ms']>50,'Outlying anchor must remain reported'
assert np.max(abs(errors(fit['theil_sen_candidate'],x,y)))<1e-6
codes=np.tile([10,11],50);samples=bad*500+1;eye=np.c_[x,codes]
c,xx,yy=anchors(codes,samples,eye,500);assert len(c)==100
try:anchors(codes,samples,eye[:-1],500)
except ValueError:pass
else:raise AssertionError('Incomplete trigger sequence must not be guessed')

rng=np.random.default_rng(5);signal=rng.normal(size=(105,70000)).astype(np.float32)
sentences=[];fixes=[];codes=[];samples=[];eye_events=[]
for si in range(12):
    start=si*5000+500;stop=start+4000;eye_start=1e6+start*2;eye_stop=1e6+stop*2
    codes.extend([10,11]);samples.extend([start+1,stop]);eye_events.extend([[eye_start,10],[eye_stop-2,11]])
    bounds=np.array([[j*20,0,j*20+15,20] for j in range(10)])
    segments=[]
    for wi in range(10):
        local=100+wi*300;segments.append([signal[:,start+local:start+local+60]])
        onset=1e6+(start+local)*2
        fixes.append([onset,onset+120,120,wi*20+7,10])
    sentences.append(dict(position=si,words=['SYNTHETIC']*10,raw=signal[:,start:stop],bounds=bounds,segments=segments))
columns=np.array(['latency','endtime','duration','fix_avgpos_x','fix_avgpos_y'])
et=dict(event=np.array(eye_events),eyeevent=SimpleNamespace(fixations=SimpleNamespace(colheader=columns,data=np.array(fixes))))
# One bad trigger creates a max-error failure while stored word EEG remains valid.
samples=np.array(samples,float);samples[15]+=30
report=validate_block('SYNTHETIC','1',signal,sentences,et,et,500,codes,samples)
v=report['validation']['theil_sen_candidate']
assert v['checked_fixations']==120 and v['checked_sentences']==12 and v['candidate_passes_audit']
assert report['candidate_clocks']['theil_sen_candidate']['anchor_max_error_ms']>50
# Corrupt one stored waveform: it cannot become evidence via nearest clock time.
sentences[0]['segments'][0]=[rng.normal(size=(105,60)).astype(np.float32)]
r=validate_block('SYNTHETIC','1',signal,sentences,et,et,500,codes,samples)
assert r['validation']['theil_sen_candidate']['checked_fixations']==119
print('PASS: trigger mismatches rejected; sparse outliers preserved; word waveform validation independent of clock prediction; corrupt segments excluded')
