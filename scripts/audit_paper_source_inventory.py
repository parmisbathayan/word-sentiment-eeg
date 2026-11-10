"""Public metadata-only audit: no native EEG downloads."""
import json
from pathlib import Path
from prepare_zuco_paper_figures import source_inventory,ROOT
sources=[v for v in json.loads((ROOT/'configs/zuco1_sources.json').read_text())['files'] if v['task']=='SR']
report=[]
for source in sources:
    blocks=source_inventory(source['subject'])
    row=dict(subject=source['subject'],listed_blocks=len(blocks),available_blocks=[b['block_id'] for b in blocks if b['available']],
        missing_blocks=[dict(block=b['block_id'],missing=b['missing_sources']) for b in blocks if not b['available']])
    if not row['available_blocks']:raise ValueError('No usable source pairs for '+source['subject'])
    report.append(row);print('INVENTORY',json.dumps(row),flush=True)
path=Path('artifacts/source_inventory_audit.json');path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(report,indent=2)+'\n')
print('ALL READER SOURCE INVENTORIES AUDITED',len(report),flush=True)
