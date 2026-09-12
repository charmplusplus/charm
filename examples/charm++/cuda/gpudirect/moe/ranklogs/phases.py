import sys, re, glob, collections
# Phase breakdown of a moe step from CHARM_MOE_TRACE=2 rank logs.
#   phases.py <rank log directory>
# Dispatcher events come from the PEs that run a dispatcher (one per process
# under MOE_PPN > 1); expert events from every PE.
d = sys.argv[1]; ev = []
for f in glob.glob(d + '/rank_*.log'):
    for line in open(f, errors='replace'):
        m = re.match(r'\[T ([0-9.]+) pe(\d+)\] (.*)', line)
        if m: ev.append((float(m.group(1)), int(m.group(2)), m.group(3).strip()))
D = collections.defaultdict(dict)   # (pe, step) -> {begin, sent, in, end}
C = collections.defaultdict(list)   # (pe, step) -> compute issue times
O = collections.defaultdict(list)   # (pe, step) -> output send times
for t, p, x in ev:
    m = re.match(r'disp step (\d+) (begin|tokens sent|outputs in|end)', x)
    if m: D[(p, int(m.group(1)))][m.group(2)] = t; continue
    m = re.match(r'expert \d+ step (\d+) compute n=(\d+)', x)
    if m: C[(p, int(m.group(1)))].append(t); continue
    m = re.match(r'expert \d+ step (\d+) outputs sent', x)
    if m: O[(p, int(m.group(1)))].append(t)
dpes = sorted(set(p for (p, s) in D))                  # dispatcher PEs
epes = sorted(set(p for (p, s) in C) | set(p for (p, s) in O))  # PEs hosting experts
steps = sorted(set(s for (_, s) in D))
acc = collections.defaultdict(list)
print("  %d dispatcher PE(s), %d PE(s) with experts" % (len(dpes), len(epes)))
print("  step: route+gather | token exch (last send -> last compute issue) | compute span (first issue -> last output send, max PE) | output exch (last output send -> last outputs-in) | combine+end")
for s in steps:
    if s < 4 or any('end' not in D[(p, s)] for p in dpes): continue
    cp = [p for p in epes if C[(p, s)]]
    if not cp: continue
    tb = max(D[(p, s)]['begin'] for p in dpes); ts = max(D[(p, s)]['tokens sent'] for p in dpes)
    tc_last = max(max(C[(p, s)]) for p in cp)
    op = [p for p in cp if O[(p, s)]]
    span = max(max(O[(p, s)]) - min(C[(p, s)]) for p in op) if op else 0
    to_last = max(max(O[(p, s)]) for p in op) if op else ts
    tin = max(D[(p, s)]['outputs in'] for p in dpes); te = max(D[(p, s)]['end'] for p in dpes)
    r = ((ts - tb) * 1e3, (tc_last - ts) * 1e3, span * 1e3, (tin - to_last) * 1e3, (te - tin) * 1e3, (te - tb) * 1e3)
    for k, v in zip("rg tx cs ox ce tot".split(), r): acc[k].append(v)
    if s % 5 == 0: print("  %3d: %6.1f | %6.1f | %6.1f | %6.1f | %6.1f | total %6.1f" % ((s,) + r))
print("  mean: " + " | ".join("%s %.1f" % (k, sum(v) / len(v)) for k, v in acc.items()))
