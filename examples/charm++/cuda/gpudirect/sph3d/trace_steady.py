#!/usr/bin/env python3
"""trace_steady.py trace.json...: steady-state picture of each merged +gputrace: per GPU busy%, kernel count, gap time by
size bin (ms per second of trace), per-PE entry-busy%, step rate (forcesKernel per GPU per second), message latency p50."""
import json, sys, collections
def union(iv):
    iv.sort(); tot=0; cs,ce=None,None
    for s,e in iv:
        if cs is None or s>ce:
            if cs is not None: tot+=ce-cs
            cs,ce=s,e
        else: ce=max(ce,e)
    if cs is not None: tot+=ce-cs
    return tot
for f in sys.argv[1:]:
    ev=json.load(open(f))['traceEvents']; X=[e for e in ev if e['ph']=='X']
    t0=min(e['ts'] for e in X); t1=max(e['ts']+e['dur'] for e in X); W=(t1-t0)/1e6
    print(f'=== {f.split("/")[-2]}  window {W:.2f} s')
    bygpu=collections.defaultdict(list); forces=collections.Counter()
    for e in X:
        if e['cat']=='kernel':
            k=(e['pid'],e['args'].get('device',0)); bygpu[k].append((e['ts'],e['ts']+e['dur']))
            if e['name'].startswith('forcesKernel'): forces[k]+=1
    bins=[(0.2,1),(1,3),(3,8),(8,1e9)]
    agg=collections.defaultdict(float); nk=0; busy_all=[]
    print('  GPU        busy%  kernels/s  steps/s | gap ms per s of trace, by gap size:  0.2-1   1-3   3-8   >8   (count>3ms)')
    for k in sorted(bygpu):
        ks=sorted(bygpu[k]); b=union(list(ks))/1e6/W*100; busy_all.append(b)
        gb=collections.Counter(); cnt3=0; cur=ks[0][1]
        for s,e in ks[1:]:
            g=(s-cur)/1000
            if g>0:
                for lo,hi in bins:
                    if lo<=g<hi: gb[(lo,hi)]+=g
                if g>3: cnt3+=1
            cur=max(cur,e)
        for bb in bins: agg[bb]+=gb[bb]
        print(f'  r{k[0]} GPU{k[1]}   {b:5.1f}   {len(ks)/W:8.0f}   {forces[k]/W:6.1f} |' + ''.join(f' {gb[bb]/W:6.1f}' for bb in bins) + f'   ({cnt3/W:.1f}/s)')
    n=len(bygpu)
    print(f'  MEAN       {sum(busy_all)/n:5.1f}' + ' '*21 + '|' + ''.join(f' {agg[bb]/W/n:6.1f}' for bb in bins))
    pe=collections.defaultdict(list)
    for e in X:
        if e['cat']=='entry': pe[(e['pid'],e['tid'])].append((e['ts'],e['ts']+e['dur']))
    pb=[union(v)/1e6/W*100 for v in pe.values()]
    pb.sort(); print(f'  PE entry-busy%: min {pb[0]:.1f} median {pb[len(pb)//2]:.1f} max {pb[-1]:.1f} over {len(pb)} PEs')
    M=[e for e in ev if e['ph']=='i' and e.get('cat')=='msg']
    T={0:'same proc',1:'same node',2:'other node'}
    for kind in ('halo','migration'):
        row=[]
        for tier in (0,1,2):
            v=sorted(e['args']['lat_ms'] for e in M if e['name']==kind and e['args']['tier']==tier)
            if v: row.append(f'{T[tier]} n={len(v)/W:.0f}/s p50 {v[len(v)//2]:.2f} p90 {v[int(.9*len(v))]:.2f}')
        print(f'  msg {kind:9s} ' + ' | '.join(row))
