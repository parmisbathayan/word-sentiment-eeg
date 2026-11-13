from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from zuco_sliding_reader_maps import scan
from zuco_recovered_followups import scalp_test
if sys.platform=='darwin':raise SystemExit('Cloud only')
times=np.arange(-75,126)*8;rng=np.random.default_rng(22);maps=rng.normal(size=(2,5,105,201))
a=scan(maps,times,batch=3);b=scan(maps,times,batch=128)
assert a['windows']==191 and a['samples_per_window']==11 and a['assignments']==32
for name in ['p_uncorrected','p_familywise','statistic']:assert np.allclose(a[name],b[name])
assert (a['p_familywise']>=a['p_uncorrected']).all()
for r in range(2):
 for j in [0,40,103,190]:
  expected=scalp_test(a['reader_maps'][r,:,j,:])['p_exact']
  assert a['p_uncorrected'][r,j]==expected
# Sign invariant scale preserves marginal ranks; max-family test includes both reps.
zero=scan(np.zeros_like(maps),times)
assert (zero['p_familywise']==1).all() and (zero['p_uncorrected']==1).all()
strong=np.broadcast_to(np.arange(105)[None,None,:,None],(2,12,105,201)).copy()
s=scan(strong,times);assert np.all(s['p_familywise']==2/4096)
print('PASS exact marginal equivalence, full-family max correction, batch invariance, null and known effect')
