#!/usr/bin/env python3
"""gap_report.py trace.json [min_ms]: per rank, the GPU's kernel gaps longer than min_ms (default 5) in a merged
+gputrace/nsys Perfetto trace, and what the rank's PEs were doing during the largest one (entries, idle, big copies).
A gap is a span with no kernel on any lane of that rank's GPU. A PE's idle is the part of the gap not covered by its
entry methods (the merge draws no idle events: empty space on a PE row is idle)."""
import json, sys, collections
f = sys.argv[1]; minms = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
ev = json.load(open(f))['traceEvents']
X = [e for e in ev if e['ph'] == 'X']
# application message latencies (instant events from the L records), by kind and tier
M = [e for e in ev if e['ph'] == 'i' and e.get('cat') == 'msg']
if M:
    TIER = {0: 'same process', 1: 'same node', 2: 'other node'}
    print('message latency, sent -> payload usable (envelope in brackets), ms:')
    for kind in ('halo', 'migration', 'halo2'):
        for tier in (0, 1, 2):
            v = sorted(e['args']['lat_ms'] for e in M if e['name'] == kind and e['args']['tier'] == tier)
            if not v: continue
            w = sorted(e['args']['env_ms'] for e in M if e['name'] == kind and e['args']['tier'] == tier)
            q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
            print(f'  {kind:9s} {TIER[tier]:12s} n={len(v):6d}  p50 {q(v, .5):6.2f} [{q(w, .5):5.2f}]  p90 {q(v, .9):6.2f} [{q(w, .9):5.2f}]  max {v[-1]:6.2f} [{w[-1]:5.2f}]')
byrank = collections.defaultdict(list)
for e in X: byrank[e['pid']].append(e)
big = []
for r in sorted(byrank):
    ks = sorted(((e['ts'], e['ts'] + e['dur']) for e in byrank[r] if e['cat'] == 'kernel'))
    if not ks: continue
    gs = []; cur_end = ks[0][1]
    for s, e in ks[1:]:
        if s > cur_end + minms * 1000: gs.append((cur_end, s))
        cur_end = max(cur_end, e)
    print(f'rank {r}: gaps >{minms:.0f} ms: ' + ', '.join(f'{s/1000:.0f}+{(e-s)/1000:.1f}' for s, e in gs) + (' (none)' if not gs else ''))
    if gs: big.append(max(gs, key=lambda g: g[1] - g[0]))
if big:
    lens = sorted((e - s) / 1000 for s, e in big)
    print(f'largest gap per rank: min {lens[0]:.1f} median {lens[len(lens)//2]:.1f} max {lens[-1]:.1f} ms')
    g0, g1 = max(big, key=lambda g: g[1] - g[0])
    print(f'during the largest ({g0/1000:.0f}..{g1/1000:.0f} ms), per rank: busiest entries / idle / copies >1 MB')
    for r in sorted(byrank):
        pes = collections.defaultdict(collections.Counter); busy = collections.Counter(); copies = collections.Counter()
        allpes = set(e['tid'] for e in byrank[r] if e['cat'] == 'entry')
        for e in byrank[r]:
            s, d = e['ts'], e['dur']; o = min(s + d, g1) - max(s, g0)
            if o <= 0: continue
            if e['cat'] == 'entry': pes[e['tid']][e['name'].split('::')[-1][:20]] += o; busy[e['tid']] += o
            elif e['cat'] == 'memcpy' and e.get('args', {}).get('bytes', 0) > 1 << 20: copies[e['name']] += 1
        for pe in sorted(allpes):
            top = ', '.join(f'{n} {v/1000:.1f}' for n, v in pes[pe].most_common(2))
            print(f'  rank {r} PE {pe}: idle {max(0.0, g1 - g0 - busy[pe])/1000:.1f} ms; {top}')
        if copies: print(f'  rank {r} copies: {dict(copies)}')
