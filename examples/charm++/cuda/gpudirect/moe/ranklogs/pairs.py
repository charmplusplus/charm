import sys, re, glob, collections
d = sys.argv[1]; ev = []
for f in glob.glob(d + '/rank_*.log'):
    for line in open(f, errors='replace'):
        m = re.match(r'\[T ([0-9.]+) pe(\d+)\] (.*)', line)
        if m: ev.append((float(m.group(1)), int(m.group(2)), m.group(3).strip()))
sent = {}; epe = {}; post = {}; cons = {}; osent = {}; opost = {}; ocons = {}
for t, p, x in ev:
    m = re.match(r'disp step (\d+) tokens sent', x)
    if m: sent[(p, int(m.group(1)))] = t; continue
    m = re.match(r'expert (\d+) step (\d+) slab src (\d+) n=(\d+) (posted|consumed)', x)
    if m:
        e, s, src, n, k = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
        epe[e] = p; (post if k == 'posted' else cons)[(e, s, src)] = (t, n); continue
    m = re.match(r'expert (\d+) step (\d+) outputs sent', x)
    if m: osent[(int(m.group(1)), int(m.group(2)))] = t; epe[int(m.group(1))] = p; continue
    m = re.match(r'disp step (\d+) output expert (\d+) n=(\d+) (posted|consumed)', x)
    if m:
        s, e, n, k = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        (opost if k == 'posted' else ocons)[(p, s, e)] = (t, n)
def q(v): v = sorted(v); return "%.1f/%.1f/%.1f" % (v[0], v[len(v)//2], v[-1]) if v else "-"
for s in (6, 7):
    print("step %d  token slabs, per (dst PE <- src PE): arrival(post) delay after src 'tokens sent' | consume-post delay  [min/med/max ms, n]" % s)
    for dst in range(4):
        row = []
        for src in range(4):
            arr = [(post[(e, s, src)][0] - sent[(src, s)]) * 1e3 for (e, ss, sr) in post if ss == s and sr == src and epe.get(e) == dst]
            cp = [(cons[(e, s, src)][0] - post[(e, s, src)][0]) * 1e3 for (e, ss, sr) in post if ss == s and sr == src and epe.get(e) == dst and (e, s, src) in cons]
            row.append("<-%d: %s | %s n=%d" % (src, q(arr), q(cp), len(arr)))
        print("   dst%d " % dst + "   ".join(row))
    print("step %d  outputs, per (dst dispatcher <- src expert PE): post delay after 'outputs sent' | consume-post [min/med/max ms]" % s)
    for dst in range(4):
        row = []
        for src in range(4):
            arr = [(opost[(dst, s, e)][0] - osent[(e, s)]) * 1e3 for (pp, ss, e) in opost if pp == dst and ss == s and epe.get(e) == src and (e, s) in osent]
            cp = [(ocons[(dst, s, e)][0] - opost[(dst, s, e)][0]) * 1e3 for (pp, ss, e) in opost if pp == dst and ss == s and epe.get(e) == src and (dst, s, e) in ocons]
            row.append("<-%d: %s | %s" % (src, q(arr), q(cp)))
        print("   dst%d " % dst + "   ".join(row))
