import sys, re, glob, collections
d = sys.argv[1]
ev = []  # (t, pe, text)
immig = collections.defaultdict(list)
for f in glob.glob(d + '/rank_*.log'):
    for line in open(f, errors='replace'):
        m = re.match(r'\[T ([0-9.]+) pe(\d+)\] (.*)', line)
        if m: ev.append((float(m.group(1)), int(m.group(2)), m.group(3).strip())); continue
        m = re.match(r'\[IMMIG (\d+)\] id=\d+ unpack wait ([0-9.]+) ms', line)
        if m: immig[int(m.group(1))].append(float(m.group(2)))
ev.sort()
pes = sorted(set(p for _, p, _ in ev))
# dispatcher step durations per PE
steps = collections.defaultdict(dict)
for t, p, x in ev:
    m = re.match(r'disp step (\d+) (begin|end)', x)
    if m: steps[(p, int(m.group(1)))][m.group(2)] = t
maxstep = max((s for (_, s) in steps), default=0)
print("  per-PE dispatcher step ms (max over PEs), steps 4..%d:" % maxstep)
row = []
for s in range(4, maxstep + 1):
    durs = [(steps[(p, s)]['end'] - steps[(p, s)]['begin']) * 1e3 for p in pes if 'end' in steps.get((p, s), {}) and 'begin' in steps.get((p, s), {})]
    row.append("%d:%.0f" % (s, max(durs)) if durs else "%d:?" % s)
print("   " + " ".join(row))
# migration events per PE
for p in pes:
    packs = [float(m.group(1)) for _, q, x in ev if q == p for m in [re.search(r'pack step \d+ phase \d+ synced after ([0-9.]+) ms', x)] if m]
    pends = [float(m.group(1)) for _, q, x in ev if q == p for m in [re.search(r'pack end ([0-9.]+) ms', x)] if m]
    dtors = [float(m.group(1)) for _, q, x in ev if q == p for m in [re.search(r'dtor .* synced after ([0-9.]+) ms', x)] if m]
    dso = sum(1 for _, q, x in ev if q == p and 'stream-ordered' in x)
    unps = [float(m.group(1)) for _, q, x in ev if q == p for m in [re.search(r'unpack step \d+ phase \d+ ([0-9.]+) ms', x)] if m]
    print("  pe%d: packs=%d sync-wait sum %.0f ms (max %.0f) | pack-end sum %.0f ms | dtor sync sum %.0f ms (max %.0f) stream-ordered=%d | unpacks=%d app-unpack sum %.0f ms | runtime unpack waits n=%d sum %.0f ms (max %.0f)" % (
        p, max(len(packs), len(pends)), sum(packs), max(packs) if packs else 0, sum(pends), sum(dtors), max(dtors) if dtors else 0, dso,
        len(unps), sum(unps), len(immig[p]), sum(immig[p]), max(immig[p]) if immig[p] else 0))
# LB windows: from first atsync-type event of a step to last resume of that step, per PE
lb = collections.defaultdict(lambda: [None, None])
for t, p, x in ev:
    m = re.match(r'expert \d+ step (\d+) (atsync|atsyncstart|atsyncwait|resume)', x)
    if not m: continue
    s, kind = int(m.group(1)), m.group(2)
    key = (p, s)
    if kind != 'resume':
        if lb[key][0] is None or t < lb[key][0]: lb[key][0] = t
    else:
        if lb[key][1] is None or t > lb[key][1]: lb[key][1] = t
out = []
for (p, s), (a, b) in sorted(lb.items()):
    if a is not None and b is not None: out.append("pe%d s%d:%.0f" % (p, s, (b - a) * 1e3))
print("  LB windows (first join -> last resume, ms): " + " ".join(out[:24]))
# largest gaps between consecutive trace events on a PE (host stalls), top 3 per PE
for p in pes:
    ts = [(t, x) for t, q, x in ev if q == p]
    gaps = sorted(((ts[i+1][0] - ts[i][0]) * 1e3, ts[i][1][:38], ts[i+1][1][:38]) for i in range(len(ts) - 1))[-3:]
    print("  pe%d largest gaps: " % p + "; ".join("%.0f ms [%s -> %s]" % g for g in reversed(gaps)))
