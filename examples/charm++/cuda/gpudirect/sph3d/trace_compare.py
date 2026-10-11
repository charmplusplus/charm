#!/usr/bin/env python3
"""trace_compare.py <name=trace.json>... : side by side, for merged +gputrace traces of the same configuration:
GPU busy % (whole window and with gaps over 20 ms, the LB barrier, excluded), gap time by size, gap time by the
kernel that ended the gap (sph3d: packHalo = the step start, i.e. the migration round and the send drain; cellCount
= the halo round), the idle time series per 50 ms bin with the LoadBalance bins marked, and the message latency
distribution if the trace carries L records (envelope = transport; usable = consumed by the step)."""
import json, sys, collections
BIG = 20000.0   # us: a gap this long is an LB barrier, not an exchange wait
def load(f):
    ev = json.load(open(f))['traceEvents']
    K = collections.defaultdict(list); E = collections.defaultdict(list); M = []
    for e in ev:
        if e['ph'] == 'X':
            if e['cat'] == 'kernel': K[e['pid']].append((e['ts'], e['ts'] + e['dur'], e['name']))
            elif e['cat'] == 'entry': E[e['pid']].append((e['ts'], e['ts'] + e['dur'], e['name'].split('::')[-1]))
        elif e['ph'] == 'i' and e.get('cat') == 'msg': M.append(e)
    for v in K.values(): v.sort()
    return K, E, M
def merged(ks):
    m = []; cs, ce = ks[0][0], ks[0][1]
    for s, e, _ in ks[1:]:
        if s > ce: m.append((cs, ce)); cs, ce = s, e
        else: ce = max(ce, e)
    m.append((cs, ce)); return m
def main():
    runs = []
    for a in sys.argv[1:]:
        name, f = a.split('=', 1) if '=' in a else (a, a)
        runs.append((name, load(f)))
    for name, (K, E, M) in runs:
        print(f'===== {name}')
        span = busy = big = 0.0; nk = 0
        bysize = collections.Counter(); bynext = collections.Counter(); cnext = collections.Counter()
        t0 = min(v[0][0] for v in K.values()); t1 = max(v[-1][1] for v in K.values())
        nb = int((t1 - t0) / 50000) + 1; idle = [0.0] * nb
        for r, ks in K.items():
            nk += len(ks); m = merged(ks); span += m[-1][1] - m[0][0]; busy += sum(e - s for s, e in m)
            cur_end = ks[0][1]
            for s, e, nm in ks[1:]:
                if s > cur_end:
                    L = s - cur_end
                    if L >= BIG: big += L
                    for lo, hi, lab in ((300, 1000, '0.3-1 ms'), (1000, 5000, '1-5 ms'), (5000, BIG, '5-20 ms'), (BIG, 1e12, '>20 ms')):
                        if lo <= L < hi: bysize[lab] += L
                    if L > 300 and L < BIG:
                        key = nm.split('(')[0].replace('Kernel', ''); bynext[key] += L; cnext[key] += 1
                    a = cur_end
                    while a < s:
                        bi = int((a - t0) / 50000); be = t0 + (bi + 1) * 50000; seg = min(s, be) - a; idle[bi] += seg; a += seg
                cur_end = max(cur_end, e)
        n = len(K)
        print(f'  {nk} kernels on {n} GPUs; busy {100*busy/span:.1f}% of the window, {100*busy/(span-big):.1f}% with gaps >20 ms excluded')
        print('  gap time by size: ' + ', '.join(f'{lab} {100*bysize[lab]/span:.1f}%' for lab in ('0.3-1 ms', '1-5 ms', '5-20 ms', '>20 ms')))
        print('  gap time (0.3-20 ms) by the kernel that ended it: ' + ', '.join(f'{k} {100*v/span:.1f}% ({cnext[k]}x {v/cnext[k]/1000:.1f} ms)' for k, v in bynext.most_common(3)))
        lb = sorted(int((s - t0) / 50000) for v in E.values() for s, e, nm in v if nm == 'LoadBalance')
        print('  idle % per 50 ms bin: ' + ' '.join(f'{100*v/(n*50000):3.0f}' for v in idle) + (f'   LoadBalance at bin(s) {lb}' if lb else ''))
        if M:
            TIER = {0: 'same proc', 1: 'same node', 2: 'other node'}
            for kind in ('halo', 'migration'):
                row = []
                for tier in (0, 1, 2):
                    v = sorted(e['args']['lat_ms'] for e in M if e['name'] == kind and e['args']['tier'] == tier)
                    w = sorted(e['args']['env_ms'] for e in M if e['name'] == kind and e['args']['tier'] == tier)
                    if v: row.append(f"{TIER[tier]}: env p50 {w[len(w)//2]:.2f} p90 {w[int(.9*len(w))]:.2f} | usable p50 {v[len(v)//2]:.2f} p90 {v[int(.9*len(v))]:.2f}")
                if row: print(f'  {kind} latency ms  ' + ';  '.join(row))
main()
