#!/usr/bin/env python3
"""Per-iteration timeline for barnes runs, with the balancing windows marked.

Timing source: the "prev time" lines. DataManager prints one just before each
"start iteration N", measuring the wall time since the previous such print, so
the value printed before "start iteration N" is the duration of iteration N-1.
The very first is the program's startup and is not an iteration at all.

Which window a balancing step lands in is DETECTED, not assumed. The balancer's
own output is interleaved with the iteration boundaries in the same stream, so
a "[DiffusionLB] Load balancing step" appearing after "start iteration N" and
before "start iteration N+1" was paid for inside iteration N. That is one lower
than isBalancingIteration() suggests -- the tree pieces join the step at the end
of the preceding iteration -- which is exactly the sort of off-by-one that makes
a warmup cut silently discard the first, most expensive balancing step.
"""
import re, sys, statistics as st

def parse(path):
    prev, cur, lb = [], -1, []
    for line in open(path, errors='ignore'):
        m = re.search(r'\(0\) prev time ([\d.eE+-]+) s', line)
        if m: prev.append(float(m.group(1)))
        m = re.search(r'\(0\) start iteration (\d+)', line)
        if m: cur = int(m.group(1))
        if 'Load balancing step' in line and cur >= 0: lb.append(cur)
    if len(prev) < 2: return None
    n = len(prev) - 1
    if not lb and re.search(r'lbperiod=(\d+)', open(path, errors='ignore').read() or ''):
        pass
    return {'setup': prev[0], 'iters': prev[1:], 'lb': sorted(set(lb)), 'n': n}

def lb_windows(d, firstlb=2, period=5):
    """Which duration windows paid for a balancing step.

    Preferred source is the balancer's own output interleaved with the
    iteration boundaries, which is only in the log when +LBDebug is on. Without
    it, fall back to the schedule: -firstlb F -lbperiod P balances at iterations
    F, F+P, ..., and the cost lands one window earlier because the tree pieces
    join the step at the end of the preceding iteration. That mapping was
    verified against a +LBDebug run of the same configuration, which reported
    windows 1, 6, 11, 16 for -firstlb=2 -lbperiod=5.
    """
    if d['lb']: return d['lb'], 'detected'
    start = firstlb - 1
    return [i for i in range(start, d['n']) if (i - start) % period == 0], 'from the schedule'

def sparkline(v, lo, hi, marks, width=None):
    blocks = "▁▂▃▄▅▆▇█"
    out = []
    for i, x in enumerate(v):
        f = 0.0 if hi <= lo else (x - lo) / (hi - lo)
        c = blocks[min(len(blocks) - 1, max(0, int(f * len(blocks))))]
        out.append("\033[1;31m%s\033[0m" % c if i in marks else c)
    return "".join(out)

if __name__ == '__main__':
    files = sys.argv[1:]
    data = [(f, parse(f)) for f in files]
    data = [(f, d) for f, d in data if d]
    allv = [x for _, d in data for x in d['iters'][1:]]
    lo, hi = min(allv), max(allv)
    print("scale: %.0f ms (lowest) to %.0f ms (highest); red = a window containing a balancing step\n"
          % (1000 * lo, 1000 * hi))
    for f, d in data:
        wins, how = lb_windows(d) if ('noLB' not in f) else ([], 'none')
        v, marks = d['iters'], set(wins)
        warm = v[1:]
        print("%-38s setup %.2fs  mean(it>=1) %6.1f ms  sd %5.1f  LB windows %s"
              % (f.split('/')[-1], d['setup'], 1000 * st.mean(warm),
                 1000 * (st.stdev(warm) if len(warm) > 1 else 0),
                 ("%s (%s)" % (wins[:12], how)) if wins else "none"))
        print("   " + sparkline(v, lo, hi, marks))
