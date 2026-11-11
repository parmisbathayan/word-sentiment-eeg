"""Apply only independently validated clock candidates in a NEW reconstruction.

Original prepared results/protocols are untouched. Exact source-file hashes bind
published validation evidence to the three native files consumed by the mapper.
No failed trigger is deleted and no maximum-error cutoff is simply raised.
"""
from contextlib import contextmanager
import numpy as np
from .common import file_digest
from .paper_clock_audit import anchors,fit_candidates,errors


def approved_clock(proof):
    for name in ['theil_sen_candidate','original_affine']:
        v=proof.get('validation',{}).get(name,{})
        # Recheck actual metrics; don't trust a solitary success flag.
        if (v.get('candidate_passes_audit') is True and v.get('checked_fixations',0)>=100 and
            v.get('checked_sentences',0)>=10 and v.get('p99_error_ms',float('inf'))<=2 and
            v.get('maximum_error_ms',float('inf'))<=4):return name
    return None


@contextmanager
def verified_mapping(module,subject,inventory,proofs):
    """Scoped wrapper, used sequentially per reader; restored even on failure."""
    original_fetch=module.fetch;original_map=module.map_block;original_sync=module.synchronize
    selected={str(p['block']):p for p in proofs if p['subject']==subject and approved_clock(p)}
    items={str(b['block_id']):b for b in inventory};checked=set()
    by_id={}
    for block,proof in selected.items():
        if block not in items or not items[block].get('available'):raise ValueError('Validated block is no longer available')
        for kind in ['eeg','eye','corrected']:
            item=items[block][kind];sha=proof['source_checksums'][kind]
            if item['id'] in by_id and by_id[item['id']]!=sha:raise ValueError('Contradictory native source proofs')
            by_id[item['id']]=sha
    def checked_fetch(item,folder):
        path=original_fetch(item,folder)
        if item['id'] in by_id:
            if file_digest(path)!=by_id[item['id']]:raise ValueError('Native source changed since independent waveform validation: '+item['name'])
            checked.add(item['id'])
        return path
    def checked_map(*args,**kwargs):
        block=str(args[1]);proof=selected.get(block)
        if proof is None:return original_map(*args,**kwargs)
        if args[0]!=subject or any(items[block][k]['id'] not in checked for k in ['eeg','eye','corrected']):raise ValueError('Clock proof is not bound to current input files')
        method=approved_clock(proof)
        def validated_sync(codes,latencies,eye_events,rate,max_error_ms=None):
            allowed=(10,11) if proof.get('trigger_choice')=='reading_only_exact_sequence' else (10,11,12,13)
            c,x,y=anchors(codes,latencies,eye_events,rate,allowed);clock=fit_candidates(x,y)[method]
            saved=proof['candidate_clocks'][method]
            for key,tolerance in [('eye_origin_ms',1e-6),('slope_seconds_per_eye_ms',1e-12),('eeg_seconds_at_origin',1e-6)]:
                if not np.isclose(clock[key],saved[key],rtol=0,atol=tolerance):raise ValueError('Candidate clock differs from independently validated clock')
            residual=errors(clock,x,y)
            return dict(clock,triggers=len(c),max_residual_ms=float(abs(residual).max()),
                rms_residual_ms=float(np.sqrt(np.mean(residual**2))),native_one_sample_target_met=bool(abs(residual).max()<=1000/rate),
                configured_tolerance_ms=8.,acceptance='independent_word_waveform_validation',candidate_method=method,
                independent_validation=proof['validation'][method],trigger_choice=proof.get('trigger_choice','all_shared_exact_sequence'),
                note='Large marker residuals remain reported; clock accepted from independently matched word EEG, not from a relaxed trigger cutoff')
        module.synchronize=validated_sync
        try:
            result=original_map(*args,**kwargs)
            print(f'RECOVER VERIFIED CLOCK {subject}/{block}: {method}',flush=True)
            return result
        finally:module.synchronize=original_sync
    module.fetch=checked_fetch;module.map_block=checked_map
    try:yield
    finally:module.fetch=original_fetch;module.map_block=original_map;module.synchronize=original_sync
