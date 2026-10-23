"""Categorized notebook display; one reader per horizontally scrollable row."""
import base64
import html
from pathlib import Path


def task_reader_rows_html(gallery,plots,task):
    items=sorted((item for item in gallery if item['kind']=='subject' and item['task']==task),key=lambda item:item['subject'])
    if not items:
        return '<p>No reader figures are available for '+html.escape(task)+'.</p>'
    width=max(item['display_width_px'] for item in items)
    rows=[]
    for item in items:
        path=Path(plots)/item['path']
        data=base64.b64encode(path.read_bytes()).decode('ascii')
        caption=html.escape(item['subject'])
        rows.append(f'<figure style="margin:0 0 12px 0;width:{width}px">'
                    f'<figcaption style="font-weight:bold">Reader {caption}</figcaption>'
                    f'<img alt="{html.escape(item["title"],quote=True)}" '
                    f'style="display:block;width:{width}px;max-width:none;height:auto" src="data:image/png;base64,{data}"></figure>')
    columns=', '.join(html.escape(x.replace('_',' ')) for x in items[0]['columns'])
    return ('<p>One reader per row. Columns: '+columns+'. Scroll horizontally to see every column.</p>'
            '<div style="overflow-x:auto;width:100%;max-width:100%;border:1px solid #bbb">'
            f'<div style="width:{width}px">'+''.join(rows)+'</div></div>')


def show_task_readers(gallery,plots,task):
    from IPython.display import HTML,display
    display(HTML(task_reader_rows_html(gallery,plots,task)))


def show_group_views(gallery,plots,views):
    from IPython.display import Image,Markdown,display
    order={'SR':0,'NR':1,'TSR':2,'task1':0}
    items=sorted((item for item in gallery if item['kind']=='group' and item['view'] in views),
                 key=lambda item:(order.get(item['task'],99),item['comparison'],views.index(item['view'])))
    for item in items:
        display(Markdown('### '+item['title'].replace('_',' ')))
        display(Image(filename=str(Path(plots)/item['path'])))
