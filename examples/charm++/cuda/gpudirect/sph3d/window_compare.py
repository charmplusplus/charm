#!/usr/bin/env python3
"""window_compare.py A.log B.log [first_step]: per-25-step window ms/step series from sph3d logs ('ms/step over the last 25'),
steady-state (step >= first_step, default half the run) mean / median / trimmed mean, and the paired per-window difference A-B."""
import re, sys, statistics as st
def series(f):
    out={}
    for l in open(f, errors='replace'):
        m=re.match(r'\s+step\s+(\d+):.*?([0-9.]+) ms/step over the last 25', l)
        if m: out[int(m.group(1))]=float(m.group(2))
    return out
A=series(sys.argv[1]); B=series(sys.argv[2]); last=max(A); first=int(sys.argv[3]) if len(sys.argv)>3 else last//2+25
ka=[s for s in sorted(A) if s>=first]; kb=[s for s in sorted(B) if s>=first]
def trim(v): v=sorted(v); k=max(1,len(v)//10); return st.mean(v[k:-k])
for name,S,keys in (('A',A,ka),('B',B,kb)):
    v=[S[s] for s in keys]
    print(f'{name} {sys.argv[1 if name=="A" else 2].split("/")[-1][:48]:48s} steps>={first}: n={len(v)} mean {st.mean(v):.2f} median {st.median(v):.2f} trimmed {trim(v):.2f} sd {st.pstdev(v):.2f} min {min(v):.2f} max {max(v):.2f}')
common=[s for s in ka if s in B]; d=[A[s]-B[s] for s in common]
print(f'paired A-B over {len(d)} windows: mean {st.mean(d):+.2f} median {st.median(d):+.2f} sd {st.pstdev(d):.2f}; windows A slower: {sum(1 for x in d if x>0)}/{len(d)}')
print('A-B by window: ' + ' '.join(f'{s}:{A[s]-B[s]:+.1f}' for s in common))
