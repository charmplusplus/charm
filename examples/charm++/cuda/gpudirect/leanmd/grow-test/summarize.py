#!/usr/bin/env python3
import json,re,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
out={}
for case in ('baseline','grow-any'):
    d=root/case/'ranklogs/weak_N2_sync'
    ranks=[p.read_text(errors='replace') for p in sorted(d.glob('rank_*.log'))]
    if not ranks:continue
    main=ranks[0]
    steps={int(k):float(v) for k,v in re.findall(r'Step (\d+) Benchmark Time ([\d.]+)',main)}
    audits=[]
    for rank in ranks:
        audits.append([dict((k,float(v)) for k,v in re.findall(r'(attributed_s|busy_s|window_s|occ)=([\d.]+)',line)) for line in rank.splitlines() if '[gpu-audit' in line])
    rounds=[]
    for i in range(min(map(len,audits))):
        v=[a[i]['attributed_s'] for a in audits]
        rounds.append({'gpu_seconds':v,'max_avg':max(v)/statistics.mean(v),'busy_fraction':[a[i]['occ'] for a in audits]})
    out[case]={'steps':len(steps),'mean42_ms':statistics.mean(v for k,v in steps.items() if k>=42) if any(k>=42 for k in steps) else None,
        'total_s':re.findall(r'Total application time ([\d.]+)',main),
        'migrations':list(map(int,re.findall(r'cross node migrations AFTER LB: (\d+)',main))),
        'energy':re.findall(r'Energy: [^\n]+',main),'success':'SIMULATION SUCCESSFULL' in main,
        'gpu_audit':rounds,
        'selection_by_rank':[[line for line in rank.splitlines() if 'move(s) accepted' in line] for rank in ranks]}
print(json.dumps(out,indent=2))
