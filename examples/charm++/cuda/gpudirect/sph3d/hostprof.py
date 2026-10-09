#!/usr/bin/env python3
"""hostprof.py <log>...: per-PE host-time table from HOSTPROF lines, the hot PE against the
median PE by phase, and how total host time tracks objects and messages by tier."""
import re, sys, statistics as st
for path in sys.argv[1:]:
    rows = []
    for line in open(path, errors="replace"):
        if not line.startswith("HOSTPROF"): continue
        d = dict(re.findall(r"(\w+)=([-\d.]+)", line.split("|")[0]))
        ph = dict((m.group(1), (float(m.group(2)), int(m.group(3)))) for m in re.finditer(r"(\w+)=([\d.]+)/(\d+)", line.split("|")[1]))
        m = re.search(r"sent intra/node/xnode=(\d+)/(\d+)/(\d+) \(MB (\d+)/(\d+)/(\d+)\) recv=(\d+)/(\d+)/(\d+) \(MB (\d+)/(\d+)/(\d+)\)", line)
        s = [int(x) for x in m.groups()]
        rows.append(dict(pe=int(d["pe"]), proc=int(d["proc"]), objs=int(d["objs"]), total=float(d["total"]), ph=ph,
                         sent=s[0:3], sentMB=s[3:6], recv=s[6:9], recvMB=s[9:12]))
    if not rows: print(f"{path}: no HOSTPROF lines"); continue
    rows.sort(key=lambda r: -r["total"])
    tot = [r["total"] for r in rows]
    print(f"\n== {path}: {len(rows)} PEs, host total s: max {max(tot):.2f} median {st.median(tot):.2f} min {min(tot):.2f}")
    print(" pe prc objs total | intra  node xnode (msgs sent+recv, k) | xnode MB | top phases (s)")
    for r in rows[:6] + [None] + rows[len(rows)//2:len(rows)//2+2] + [None] + rows[-2:]:
        if r is None: print("  ..."); continue
        tiers = [(r["sent"][i] + r["recv"][i]) / 1e3 for i in range(3)]
        top = sorted(r["ph"].items(), key=lambda kv: -kv[1][0])[:4]
        print(f"{r['pe']:3d} {r['proc']:3d} {r['objs']:4d} {r['total']:5.2f} | {tiers[0]:5.1f} {tiers[1]:5.1f} {tiers[2]:5.1f} | {r['sentMB'][2]+r['recvMB'][2]:7d} | " +
              "  ".join(f"{k}={v[0]:.2f}" for k, v in top))
    # phase shares, hot vs median
    hot, med = rows[0], rows[len(rows)//2]
    print(" phase shares, hot PE vs median PE:")
    for k in hot["ph"]:
        a, b = hot["ph"][k][0], med["ph"].get(k, (0, 0))[0]
        if a + b > 0.05: print(f"   {k:13s} hot {a:6.2f} s ({100*a/hot['total']:4.1f}%)  median {b:6.2f} s  per-call hot {1e6*a/max(1,hot['ph'][k][1]):6.1f} us median {1e6*b/max(1,med['ph'].get(k,(0,1))[1]):6.1f} us")
    # correlations
    def corr(x, y):
        mx, my = st.mean(x), st.mean(y); sx, sy = st.pstdev(x), st.pstdev(y)
        return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (len(x)*sx*sy) if sx > 0 and sy > 0 else float("nan")
    feats = {"objs": [r["objs"] for r in rows], "msgs intra": [r["sent"][0]+r["recv"][0] for r in rows],
             "msgs same-node": [r["sent"][1]+r["recv"][1] for r in rows], "msgs cross-node": [r["sent"][2]+r["recv"][2] for r in rows],
             "msgs cross-process": [r["sent"][1]+r["recv"][1]+r["sent"][2]+r["recv"][2] for r in rows],
             "msgs all": [sum(r["sent"])+sum(r["recv"]) for r in rows], "MB cross-node": [r["sentMB"][2]+r["recvMB"][2] for r in rows],
             "MB cross-process": [r["sentMB"][1]+r["recvMB"][1]+r["sentMB"][2]+r["recvMB"][2] for r in rows]}
    print(" correlation of total host time with:", "  ".join(f"{k} {corr(v, tot):+.2f}" for k, v in feats.items()))
