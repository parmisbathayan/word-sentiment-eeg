"""Execute actual notebook Git setup on fresh tag-free and existing shallow clones.

Cloud only. No Drive mount, dependency installation, EEG reads or extraction.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
if sys.platform=='darwin':raise SystemExit('Cloud only: no extra code clones on the Mac')
paths=sorted(list((ROOT/'notebooks').glob('zuco_paper_recover_*.ipynb'))+list((ROOT/'notebooks').glob('teco_paper_*.ipynb'))+list((ROOT/'notebooks').glob('zuco_recovered_followups*.ipynb'))+list((ROOT/'notebooks').glob('zuco_agent_paper_subset.ipynb')))
with tempfile.TemporaryDirectory() as tmp:
    for p in paths:
        cell=next(''.join(c['source']) for c in json.loads(p.read_text())['cells'] if c['cell_type']=='code' and 'CODE_REVISION=' in ''.join(c['source']))
        tree=ast.parse(cell);nodes=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE_REVISION' for t in node.targets):nodes.append(node)
            elif isinstance(node,ast.FunctionDef) and node.name=='git':nodes.append(node)
            elif isinstance(node,ast.If) and ('REPO' in ast.unparse(node.test) or 'CODE_REVISION' in ast.unparse(node.test)):nodes.append(node)
        ns={'os':os,'subprocess':subprocess,'REPO':Path(tmp)/p.stem}
        # Deliberately no local tags. This is the failure-prone shallow setup.
        subprocess.run(['git','clone','--depth','1','--no-tags','https://github.com/parmisbathayan/word-sentiment-eeg.git',str(ns['REPO'])],check=True,stdout=subprocess.DEVNULL)
        code=compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec')
        for repeat in range(2):
            exec(code,ns)
            assert subprocess.check_output(['git','-C',str(ns['REPO']),'rev-parse','HEAD'],text=True).strip()==ns['CODE_REVISION']
        if p.name=='zuco_paper_recover_prepare_cpu.ipynb':assert "'--budget-bytes','3600000000'" in p.read_text()
        if p.name.endswith('analyze_gpu.ipynb'):assert 'analyze_paper_small_storage.py' in p.read_text()
        print('PASS fresh tag-free checkout + unchanged resume:',p.name,flush=True)
print('All paper/follow-up notebook checkout paths verified; no scientific code or Drive files changed')
