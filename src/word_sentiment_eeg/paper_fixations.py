"""Word-bound mapping and explicit selection flags for the reproduction route.

These functions require an independently validated sentence/word-bound join.
They never guess a missing sentence ID from fixation sequence order.
"""
import re
import numpy as np
import pandas as pd


def word_regions(rectangles,mode='original'):
    boxes=np.asarray(rectangles,dtype=float).copy()
    if boxes.ndim!=2 or boxes.shape[1]!=4 or not np.isfinite(boxes).all():
        raise ValueError('Expected finite [left,top,right,bottom] word rectangles')
    if np.any(boxes[:,2]<=boxes[:,0]) or np.any(boxes[:,3]<=boxes[:,1]):raise ValueError('Invalid word rectangles')
    if mode=='original':return boxes
    if mode!='horizontal_midpoints':raise ValueError('Unknown explicitly configured word-region policy')
    # Independent geometric implementation of horizontal gap sharing, not a
    # copy of the unpublished study mapper or a claim of pixel-for-pixel parity.
    # Group only rectangles with overlapping vertical extents; no magic line IDs.
    groups=[]
    for i in np.argsort(boxes[:,1],kind='stable'):
        candidates=[g for g in groups if min(boxes[j,3] for j in g)>boxes[i,1]
                    and max(boxes[j,1] for j in g)<boxes[i,3]]
        if len(candidates)>1:raise ValueError('Ambiguous line grouping')
        if candidates:candidates[0].append(int(i))
        else:groups.append([int(i)])
    for group in groups:
        group=sorted(group,key=lambda i:boxes[i,0])
        for a,b in zip(group,group[1:]):
            if boxes[b,0]<boxes[a,2]:raise ValueError('Overlapping original word boxes on a line')
            midpoint=(boxes[a,2]+boxes[b,0])/2
            boxes[a,2]=midpoint;boxes[b,0]=midpoint
        boxes[group[0],0]-=3;boxes[group[-1],2]+=3
    return boxes


def map_word(x,y,regions):
    if not np.isfinite([x,y]).all():return None,'invalid_gaze'
    boxes=np.asarray(regions)
    hit=np.flatnonzero((boxes[:,0]<=x)&(x<=boxes[:,2])&(boxes[:,1]<=y)&(y<=boxes[:,3]))
    if len(hit)==0:return None,'out_of_word_bounds'
    if len(hit)>1:return None,'ambiguous_word_bounds'
    return int(hit[0]),'matched'


def blink_predecessors(fixation_starts,fixation_durations_ms,blink_starts,blink_durations_ms):
    """Explicit operational flag, not asserted to be the authors' exact rule.

    A blink overlapping this fixation, or ending since the preceding fixation's
    end, flags the event. Input clocks are seconds; durations are milliseconds.
    This remains defined even when a saccade lies between blink and fixation.
    """
    starts=np.asarray(fixation_starts,float);dur=np.asarray(fixation_durations_ms,float)/1000
    blink=np.asarray(blink_starts,float);ends=blink+np.asarray(blink_durations_ms,float)/1000
    if len(starts)!=len(dur) or len(blink)!=len(ends):raise ValueError('Event timeline lengths disagree')
    if not np.isfinite(np.concatenate([starts,dur,blink,ends])).all() or np.any(dur<=0) or np.any(ends<blink):
        raise ValueError('Invalid event times or durations')
    if np.any(np.diff(starts)<0):raise ValueError('Fixations must be chronologically ordered')
    result=[]
    for i,start in enumerate(starts):
        previous_end=starts[i-1]+dur[i-1] if i else 0.
        between=(ends>=previous_end)&(ends<=start)
        overlap=(blink<start+dur[i])&(ends>start)
        result.append(bool(np.any(between|overlap)))
    return np.asarray(result)


def preceding_negation(words,index):
    if not 0<=index<len(words):raise ValueError('Target word index outside sentence')
    normalized=[re.sub(r"^[^\w]+|[^\w]+$",'',str(w).lower()) for w in words[max(0,index-3):index]]
    return any(w in {'no','not','without','nor','neither'} for w in normalized)


def selection_outcomes(events):
    """Sequential pre-matching outcomes; amplitude rejection belongs after matching.

    Keep original independent flags in the caller's event table. Unknown timing,
    gaze, labels or mappings never become a successful inclusion by default.
    """
    required={'mapping_status','duration_ms','blink_predecessor','word_label_status','sentiment','preceded_by_negation'}
    if not required.issubset(events.columns):raise ValueError('Missing explicit reproduction selection flags')
    duration=pd.to_numeric(events.duration_ms,errors='coerce')
    outcome=np.full(len(events),'eligible_before_matching',dtype=object)
    # First applicable rule wins so totals reconcile without double counting.
    conditions=[
        (~events.mapping_status.eq('matched'),'unavailable_word_mapping'),
        (~np.isfinite(duration)|(duration<=0),'invalid_duration'),
        (~events.blink_predecessor.isin([True,False]),'unknown_blink_predecessor'),
        (events.blink_predecessor.eq(True),'blink_predecessor'),
        (duration<100,'fixation_shorter_than_100ms'),
        (~events.word_label_status.eq('matched'),'unavailable_original_word_label'),
        (~events.sentiment.isin(['negative','neutral','positive']),'unavailable_original_word_label'),
        (events.sentiment.eq('neutral'),'neutral_excluded_in_paper_binary_route'),
        (~events.preceded_by_negation.isin([True,False]),'unknown_negation_context'),
        (events.preceded_by_negation.eq(True),'preceded_by_negation'),
    ]
    for condition,label in conditions:
        apply=(outcome=='eligible_before_matching')&condition.to_numpy(dtype=bool)
        outcome[apply]=label
    return outcome
