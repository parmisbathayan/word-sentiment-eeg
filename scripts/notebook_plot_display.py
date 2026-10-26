"""Categorized display with one combined reader-by-plot matrix per task."""
import base64
import html
from pathlib import Path


def task_reader_rows_html(gallery,plots,task):
    items=[item for item in gallery if item['kind']=='subject_matrix' and item['task']==task]
    if not items:
        raise ValueError('Combined reader matrix missing for '+task+'. Rerun the notebook setup/plot cell.')
    if len(items)!=1:
        raise ValueError('Expected exactly one reader matrix per task')
    item=items[0];width=int(item['display_width_px'])
    data=base64.b64encode((Path(plots)/item['path']).read_bytes()).decode('ascii')
    columns=', '.join(html.escape(x.replace('_',' ')) for x in item['columns'])
    return (f'<p><b>{html.escape(task)}: one matrix, {item["reader_count"]} reader rows.</b> '
            'Columns: '+columns+'. Scroll within the matrix to inspect every row and column.</p>'
            '<div style="overflow:auto;width:100%;max-width:100%;max-height:1100px;border:1px solid #bbb">'
            f'<img alt="{html.escape(item["title"],quote=True)}" '
            f'style="display:block;width:{width}px;max-width:none;height:auto" src="data:image/png;base64,{data}"></div>')


def show_task_readers(gallery,plots,task):
    from IPython.display import HTML,display
    display(HTML(task_reader_rows_html(gallery,plots,task)))
    item=next(x for x in gallery if x['kind']=='subject_matrix' and x['task']==task)
    print('Combined PNG:',Path(plots)/item['path'])
    print('Combined PDF:',Path(plots)/item['pdf_path'])


def show_group_views(gallery,plots,views):
    from IPython.display import Image,Markdown,display
    order={'SR':0,'NR':1,'TSR':2,'task1':0}
    items=sorted((item for item in gallery if item['kind']=='group' and item['view'] in views),
                 key=lambda item:(order.get(item['task'],99),item['comparison'],views.index(item['view'])))
    for item in items:
        display(Markdown('### '+item['title'].replace('_',' ')))
        display(Image(filename=str(Path(plots)/item['path'])))
