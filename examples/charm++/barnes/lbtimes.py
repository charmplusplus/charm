#!/usr/bin/env python3
"""Per-iteration timings from a barnes log, with the warmup excluded.

The application's own "avg time" cannot be used for this. DataManager
accumulates avgIterationRuntime starting from prevIterationStart = 0.0, so its
first term is the wall clock from program start to the first iteration --
particle load, first tree build, GPU context -- and its last iteration is never
added. It then divides by the iteration count. So the printed number is
(setup + iterations 0..N-2)/N: it charges startup to every iteration.

The "prev time" lines carry what is actually wanted. The one printed before
iteration k is the duration of iteration k-1, and the very first is the setup.
"""
import re, sys, statistics as st

def parse(path):
    p = []
    for line in open(path, errors='ignore'):
        m = re.search(r'\(0\) prev time ([\d.eE+-]+) s', line)
        if m: p.append(float(m.group(1)))
    if len(p) < 2: return None
    return {'setup': p[0], 'iters': p[1:]}      # iters[k] is iteration k

def stats(d, warm):
    v = d['iters'][warm:]
    if not v: return None
    return {'n': len(v), 'mean': st.mean(v), 'sd': st.stdev(v) if len(v) > 1 else 0.0,
            'median': st.median(v), 'min': min(v), 'max': max(v)}

if __name__ == '__main__':
    warm = int(sys.argv[1]) if len(sys.argv) > 2 else 5
    for f in sys.argv[2 if len(sys.argv) > 2 else 1:]:
        d = parse(f)
        if not d: print("%-44s no timing lines" % f); continue
        a, b = stats(d, 0), stats(d, warm)
        if b is None: print("%-44s only %d iterations, fewer than the %d-iteration warmup" % (f, a['n'], warm)); continue
        print("%-44s setup %6.3f s | all %2d it %7.1f ms | warm>=%d %2d it %7.1f ms sd %5.1f med %7.1f" % (
            f, d['setup'], a['n'], 1000*a['mean'], warm, b['n'], 1000*b['mean'], 1000*b['sd'], 1000*b['median']))
