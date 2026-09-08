#!/usr/bin/env python3
"""Aggregate the LB matrix from the per-run logs.

Timing comes from the "prev time" series, not from the application's own
"avg time": DataManager accumulates avgIterationRuntime from
prevIterationStart = 0.0, so its first term is the program's startup and its
last iteration is never added, and it then divides by the iteration count.

WARMUP is dropped by iteration index, the same count for every arm. The cut is
1: iteration 0 is the first tree build and the GPU warming up, and nothing
else is discarded. It cannot be larger. The balancer's cost lands in the
window BEFORE the iteration isBalancingIteration() names -- the tree pieces
join the step at the end of the preceding iteration -- so with -firstlb=2 the
first balancing step is paid for inside iteration 1, and a cut of 2 would
silently throw away the first and most expensive one. The windows are detected
from each log rather than computed, and the table prints how many landed
inside the average.
"""
import re, sys, glob, statistics as st
from collections import defaultdict

WARM = int(sys.argv[1]) if len(sys.argv) > 1 else 1

def parse(path):
    p, mig, cur, lb = [], 0, -1, []
    for line in open(path, errors='ignore'):
        m = re.search(r'\(0\) prev time ([\d.eE+-]+) s', line)
        if m: p.append(float(m.group(1)))
        m = re.search(r'\(0\) start iteration (\d+)', line)
        if m: cur = int(m.group(1))
        if 'Load balancing step' in line and cur >= 0: lb.append(cur)
        m = re.search(r'cross node migrations AFTER LB: (\d+)', line)
        if m: mig += int(m.group(1))
    if len(p) < 2: return None
    return {'setup': p[0], 'iters': p[1:], 'mig': mig, 'lb': sorted(set(lb))}

rows = defaultdict(list)
for f in sorted(glob.glob('check/lbm_*_*_[0-9].log')):
    m = re.match(r'check/lbm_(.+)_(noLB|sync|async)_(\d)\.log$', f)
    if not m: continue
    d = parse(f)
    if d: rows[(m.group(1), m.group(2))].append(d)

def lbsteps(d, warm):
    return sum(1 for i in d['lb'] if i >= warm)

for ds in dict.fromkeys(k[0] for k in rows):
    base = None
    print("\n%s   (warmup: iterations 0-%d dropped)" % (ds, WARM - 1))
    print("  %-6s %3s %9s %7s %8s %7s %10s" %
          ("arm", "n", "ms/iter", "sd", "LB steps", "moves", "vs noLB"))
    # A warmup cut of 1 drops only iteration 0, and the earliest window a
    # balancing step can land in is 1, so no cut this small can ever exclude
    # one -- true whatever -firstlb and -lbperiod were.
    for arm in ('noLB', 'sync', 'async'):
        v = rows.get((ds, arm))
        if not v: continue
        means = [st.mean(d['iters'][WARM:]) * 1000 for d in v]
        mean, sd = st.mean(means), (st.stdev(means) if len(means) > 1 else 0.0)
        steps = st.mean(lbsteps(d, WARM) for d in v)
        dropped = st.mean(len([i for i in d['lb'] if i < WARM]) for d in v)
        if arm == 'noLB': base = mean
        delta = "" if base is None or arm == 'noLB' else "%+.1f%%" % (100 * (mean - base) / base)
        detected = any(d['lb'] for d in v)
        cols = ("%8.0f %7.0f" % (steps, st.mean(d['mig'] for d in v))) if (arm == 'noLB' or detected) \
               else "%8s %7s" % ("n/a", "n/a")   # +LBDebug off: not printed by the balancer
        print("  %-6s %3d %9.1f %7.1f %s %10s%s" %
              (arm, len(means), mean, sd, cols, delta,
               "" if dropped == 0 else "   !! %.0f LB step(s) fell in the warmup and were dropped" % dropped))
