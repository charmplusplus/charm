#!/usr/bin/env python3
"""flights.py <ranklog dir> [--steps A:B] [--expert E]: the example's per-message flight log (-F or
MOE_FLIGHT_LOG=1): for every output message a dispatcher consumed, the sender's send time, the time the
receiver's host saw the envelope (the post entry ran) and the time it was consumed (the entry ran, the
landing complete), CLOCK_REALTIME ns on each host. Prints, per step range, the distribution of
envelope - send (the sender's issue to the receiver's host) and consumed - envelope (the landing and the
delivery), split by same host / other host and by message size, and the hot expert's outputs on their own.
Clocks across hosts agree to NTP; same-host pairs are exact."""
import argparse, glob, os, re, sys
from collections import defaultdict
ap = argparse.ArgumentParser()
ap.add_argument('dir'); ap.add_argument('--steps', default='30:41'); ap.add_argument('--expert', type=int, default=None)
a = ap.parse_args()
A, B = map(int, a.steps.split(':'))
host_of = {}
rows = []
toks = []
for path in sorted(glob.glob(os.path.join(a.dir, 'rank_*.log'))):
    r = int(re.search(r'rank_(\d+)\.log', path).group(1))
    host = None
    for line in open(path, errors='replace'):
        if line.startswith('# rank') and 'host' in line:
            host = line.rsplit('host', 1)[1].strip().split('.')[0]
        if line.startswith('[FLIGHT] T '):
            f = line.split()
            step = int(f[3]); spe = int(f[5]); e = int(f[9]); nb = int(f[11]); ts, te = int(f[13]), int(f[15])
            if A <= step <= B: toks.append((step, spe, e, nb, te - ts))
        if line.startswith('[FLIGHT] O '):
            f = line.split()
            # [FLIGHT] O step S src SPE dst DPE expert E bytes N send T env T land T
            step = int(f[3]); spe = int(f[5]); dpe = int(f[7]); e = int(f[9]); nb = int(f[11])
            ts, te, tl = int(f[13]), int(f[15]), int(f[17])
            if A <= step <= B: rows.append((step, spe, dpe, e, nb, ts, te, tl, r))
    if host: host_of[r] = host
if toks:
    from collections import defaultdict as _dd
    g = _dd(list)
    for step, spe, e, nb, dt in toks:
        g['big (>=8 MB)' if nb >= 8 << 20 else ('mid (1-8 MB)' if nb >= 1 << 20 else 'small (<1 MB)')].append(dt)
    print(f'token puts, steps {A}-{B}: {len(toks)} puts; completion - issue, ms median / p90 / max')
    for k, v in sorted(g.items()):
        v.sort(); n = len(v)
        print(f'  {k:14s} n={n:5d}  {v[n//2]/1e6:.2f} / {v[int(n*0.9)]/1e6:.2f} / {v[-1]/1e6:.2f}')
    by = _dd(list)
    for step, spe, e, nb, dt in toks: by[(step, spe)].append(dt)
    firsts = sorted(min(v) for v in by.values()); lasts = sorted(max(v) for v in by.values())
    print(f'  per (step, dispatcher): first completion median {firsts[len(firsts)//2]/1e6:.2f} ms, last median {lasts[len(lasts)//2]/1e6:.2f} ms, last max {lasts[-1]/1e6:.2f}')
if not rows:
    sys.exit('no [FLIGHT] O records in the range (run with -F or MOE_FLIGHT_LOG=1)')
pe_host = {}
# rank r's log holds dispatcher r's consumption: dpe is its PE; the sender's host is the sender PE's process's
def host_pe(pe):
    # one PE per process in these runs: rank == PE; several PEs per process: assume contiguous blocks
    if pe in host_of: return host_of[pe]
    return host_of.get(min(host_of, key=lambda k: abs(k - pe)), '?')
def q(v):
    v = sorted(v); n = len(v)
    if n == 0: return '-'
    return f'{v[n//2]/1e6:.2f} / {v[int(n*0.9)]/1e6:.2f} / {v[-1]/1e6:.2f}'
def report(title, sel):
    print(f'{title}: {len(sel)} messages; ms median / p90 / max')
    groups = defaultdict(list)
    for step, spe, dpe, e, nb, ts, te, tl, r in sel:
        same = host_pe(spe) == host_pe(dpe)
        size = 'small (<1 MB)' if nb < 1 << 20 else ('mid (1-8 MB)' if nb < 8 << 20 else 'big (>=8 MB)')
        groups[(same, size)].append((te - ts, tl - te, tl - ts))
    for (same, size), v in sorted(groups.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
        print(f'  {"same host" if same else "other host":10s} {size:14s} n={len(v):5d}  envelope-send {q([x[0] for x in v]):>24s}   consumed-envelope {q([x[1] for x in v]):>24s}   consumed-send {q([x[2] for x in v]):>24s}')
report(f'all outputs, steps {A}-{B}', rows)
hot = a.expert
if hot is None:
    cnt = defaultdict(int)
    for row in rows: cnt[row[3]] += row[4]
    hot = max(cnt, key=cnt.get)
report(f'expert {hot} (the most bytes)', [r for r in rows if r[3] == hot])
# per step, the hot expert's outputs: when each dispatcher consumed them relative to the earliest send that step
print(f'\nexpert {hot} per step: dispatcher PE: consumed - send (ms) [bytes MB]')
by_step = defaultdict(list)
for row in rows:
    if row[3] == hot: by_step[row[0]].append(row)
for step in sorted(by_step):
    v = sorted(by_step[step], key=lambda r: r[2])
    print(f'  step {step}: ' + '  '.join(f'PE{r[2]}: {(r[7]-r[5])/1e6:.1f} [{r[4]/1048576:.0f}]' for r in v))
