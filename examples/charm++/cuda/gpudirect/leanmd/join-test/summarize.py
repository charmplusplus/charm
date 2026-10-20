#!/usr/bin/env python3
"""Summarize local-duration force-join snapshots; no cross-process clock math."""
import json, re, statistics
from pathlib import Path
root=Path(__file__).resolve().parent

def stats(xs):
    xs=sorted(x for x in xs if x>=0)
    if not xs:return {}
    return dict(n=len(xs),mean=round(statistics.mean(xs),3),p50=round(xs[len(xs)//2],3),p95=round(xs[min(len(xs)-1,int(len(xs)*.95))],3),max=round(xs[-1],3))

out={}
for arm in ('nolb','sync'):
    d=root/'ranklogs'/f'weak_N2_{arm}'
    forces=[];acks=[];computes=[];inputs=[];times={};energy=[]
    for p in sorted(d.glob('rank_*.log')):
        for line in p.read_text(errors='replace').splitlines():
            if line.startswith('[join-'):
                fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
                for k,v in list(fields.items()):
                    if ',' not in v: fields[k]=float(v) if '.' in v else int(v)
                if line.startswith('[join-force]'):forces.append(fields)
                elif line.startswith('[join-ack]'):acks.append(fields)
                elif line.startswith('[join-compute]'):computes.append(fields)
                elif line.startswith('[join-input]'):inputs.append(fields)
            m=re.match(r'Step (\d+) Benchmark Time ([\d.]+)',line)
            if m:times[int(m[1])]=float(m[2])
            if 'Energy:' in line or 'SIMULATION' in line:energy.append(line)
    last=[x for x in forces if x['order']==26]
    input_index={(x['step'],x['c']):x for x in inputs}
    ack_index={(x['step'],x['c']):x for x in acks}
    for row in last:
        x,y,z=map(int,row['cell'].split(',')); n=row['ordinal']
        dx,dy,dz=n//9-1,(n%9)//3-1,n%3-1
        if n>=13: c=(x+1,y+1,z+1,x+1+dx,y+1+dy,z+1+dz)
        else:
            ax,ay,az=(x+dx)%16+1,(y+dy)%8+1,(z+dz)%8+1
            c=(ax,ay,az,ax-dx,ay-dy,az-dz)
        key=(row['step'],','.join(map(str,c)))
        row['input_ms']=input_index.get(key,{}).get('input_ms',-1)
        row['ack_ms']=ack_index.get(key,{}).get('ack_ms',-1)
    a={'steps':len(times),'mean42_ms':stats([v for k,v in times.items() if k>=42]),'energy':energy,
       'force_records':len(forces),'last_force_records':len(last),
       'compute_input_ms':stats([x['input_ms'] for x in inputs]),
       'compute_ready_ms':stats([x['ready_ms'] for x in computes]),
       'compute_ack_ms':stats([x['ack_ms'] for x in acks])}
    for name,rows in [('all',forces),('last',last)]:
        a[name]={key:stats([x[key] for x in rows]) for key in ('ready_ms','post_ms','elapsed_ms','span_ms')}
        a[name]['cross_node_fraction']=round(sum(x['pe']//32!=x['source']//32 for x in rows)/len(rows),4) if rows else None
    a['last_contributor_input_ms']=stats([x['input_ms'] for x in last])
    a['last_contributor_ack_ms']=stats([x['ack_ms'] for x in last])
    a['by_step']={str(step):{key:stats([x[key] for x in last if x['step']==step]) for key in ('ready_ms','post_ms','elapsed_ms','span_ms')} for step in (45,65,85,95)}
    a['last_by_receiver']={str(rank):{key:stats([x[key] for x in last if x['pe']//8==rank]) for key in ('ready_ms','post_ms','elapsed_ms','span_ms')} for rank in range(8)}
    a['last_source_counts']={str(rank):sum(x['source']//8==rank for x in last) for rank in range(8)}
    out[arm]=a
print(json.dumps(out,indent=2))
