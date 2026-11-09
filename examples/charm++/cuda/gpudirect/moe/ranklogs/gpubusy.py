#!/usr/bin/env python3
"""gpubusy.py <gputrace dir> [<dir> ...]: each GPU's busy fraction (the union of its kernels, from the .proc files)
over windows of the dispatcher's steps on PE 0 (Dispatcher::runStep entries): steady 48-61, round 1 19-27,
round 2 40-48, and 5-62; the mean, min and max over the GPUs and each window's length."""
import glob, sys
def steps(run):
    pe0 = sorted(glob.glob(run + "/gputrace.*.pe0"))[0]; ep = None; st = []
    for l in open(pe0):
        x = l.split()
        if not x: continue
        if x[0] == "P" and " ".join(x[2:]) == "Dispatcher::runStep": ep = x[1]
        elif x[0] == "E" and ep is not None and len(x) >= 4 and x[3] == ep: st.append(int(x[1]))
    return sorted(st)
def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out
for run in sys.argv[1:]:
    st = steps(run); gp = []
    for p in sorted(glob.glob(run + "/gputrace.*.proc")):
        gp.append(union([(int(x[3]), int(x[4])) for x in (l.split() for l in open(p)) if x and x[0] == "K" and len(x) >= 5]))
    print(run)
    for name, (s0, s1) in (("steady 48-61", (48, 62)), ("round 1, 19-27", (19, 28)), ("round 2, 40-48", (40, 49)), ("steps 5-62", (5, 63))):
        if s1 > len(st): continue
        t0, t1 = st[s0 - 1], st[s1 - 1]
        fr = [sum(max(0, min(b, t1) - max(a, t0)) for a, b in u) / (t1 - t0) for u in gp]
        print("   %-15s GPU busy mean %3.0f%% (min %3.0f, max %3.0f) over %.0f ms" % (name, 100 * sum(fr) / len(fr), 100 * min(fr), 100 * max(fr), (t1 - t0) / 1e6))
