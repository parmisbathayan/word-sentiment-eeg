"""Cloud-only eye endpoint audit, including all readers without EEG downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from scipy.io import loadmat
from prepare_zuco_paper_figures import source_inventory,fetch,event_table,repair_eye_duration_columns,ROOT


def audit_file(subject,block,kind,item,folder):
    path=fetch(item,folder)
    eye=loadmat(path,squeeze_me=True,struct_as_record=False);report={}
    for name in ['fixations','saccades','blinks']:
        table=event_table(eye,name);duration=table.endtime-table.latency
        bad=(~np.isfinite(duration))|(duration<=0)|(duration>10000)
        report[name]=dict(count=len(table),bad_rows=table.loc[bad,['latency','endtime','duration']].to_dict('records'),
                          duration_range=[float(duration.min()),float(duration.max())] if len(duration) else None)
    try:report['repair_audit']=repair_eye_duration_columns(eye)
    except ValueError as error:report['validation_error']=str(error)
    path.unlink()
    result=dict(subject=subject,block=block,kind=kind,file=item['name'],events=report)
    if any(v.get('bad_rows') for v in report.values() if isinstance(v,dict)) or report.get('validation_error'):
        print('ENDPOINT CASE '+json.dumps(result,allow_nan=True),flush=True)
    return result


def audit_subject(subject,requested_block=None):
    results=[]
    with tempfile.TemporaryDirectory(prefix='eye-endpoint-audit-') as temporary:
        blocks=source_inventory(subject)
        for block in blocks:
            if requested_block and block['block_id']!=requested_block:continue
            for kind in ['eye','corrected']:
                if block[kind]:results.append(audit_file(subject,block['block_id'],kind,block[kind],Path(temporary)))
    print('EYE AUDIT COMPLETE '+subject+' files='+str(len(results)),flush=True)
    return results


if __name__=='__main__':
    if sys.platform=='darwin':raise SystemExit('Cloud only: no eye datasets on Mac')
    parser=argparse.ArgumentParser();parser.add_argument('--all-readers',action='store_true');args=parser.parse_args()
    result=audit_subject('ZJM','6')+audit_subject('ZJS','1')
    if args.all_readers:
        subjects=[s['subject'] for s in json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files'] if s['task']=='SR']
        with ThreadPoolExecutor(max_workers=4) as pool:
            for rows in pool.map(audit_subject,subjects):result.extend(rows)
    path=Path('artifacts/eye_endpoint_audit.json');path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=True)+'\n')
    print('ENDPOINT AUDIT SAVED',path,flush=True)
