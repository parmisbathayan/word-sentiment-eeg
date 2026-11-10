"""Cloud-only diagnostic for native eye-event endpoint validation."""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from scipy.io import loadmat
from prepare_zuco_paper_figures import source_inventory,fetch,event_table,repair_eye_duration_columns
if sys.platform=='darwin':raise SystemExit('Cloud only')
with tempfile.TemporaryDirectory() as temporary:
    block=next(b for b in source_inventory('ZDM') if b['block_id']=='5')
    report={}
    for kind in ['eye','corrected']:
        eye=loadmat(fetch(block[kind],Path(temporary)),squeeze_me=True,struct_as_record=False)
        report[kind]={}
        for name in ['fixations','saccades','blinks']:
            table=event_table(eye,name);duration=table.endtime-table.latency
            bad=(~np.isfinite(duration))|(duration<=0)|(duration>10000)
            report[kind][name]=dict(count=len(table),bad_rows=table.loc[bad].to_dict('records'),duration_range=[float(duration.min()),float(duration.max())])
        report[kind]['repair_audit']=repair_eye_duration_columns(eye)
    print('ENDPOINT AUDIT',json.dumps(report,indent=2))
