#!/usr/bin/env python3
"""What a balancing step does to the SFC partition, and what that costs.

Four panels on a shared iteration axis:

  1. the placement itself -- tree piece index on y, colour = owning PE. Tree
     pieces are numbered in key order, so a solid horizontal band is a PE
     holding a contiguous stretch of the space-filling curve. Stripes mean the
     ordering has been cut up.
  2. fragmentation: how many contiguous runs the partition has. One run per PE
     is the ideal; more means PEs whose bounding boxes overlap each other.
  3. locally essential tree volume actually pushed, summed over PE pairs.
  4. the wall time of each iteration.

The argument is that 2 drives 3 drives 4, and that the balancer sees none of
it -- it equalises device time, which panel 4 shows is not what sets the step.

usage: lbmap.py <log> <out.png> "<title>"
"""
import re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse(path):
    it = -1
    place, runs, let, dur = {}, {}, {}, []
    npieces = 0
    for line in open(path, errors='ignore'):
        m = re.search(r'\(0\) prev time ([\d.eE+-]+) s', line)
        if m: dur.append(float(m.group(1)))
        m = re.search(r'\(0\) start iteration (\d+)', line)
        if m: it = int(m.group(1))
        m = re.search(r'\[TPMAP\] it (\d+) runs (\d+) \|(.*)', line)
        if m:
            k = int(m.group(1)); runs[k] = int(m.group(2))
            spans = []
            for pe, a, b in re.findall(r'(\d+):(\d+)-(\d+)', m.group(3)):
                spans.append((int(pe), int(a), int(b)))
                npieces = max(npieces, int(b) + 1)
            place[k] = spans
        m = re.search(r'\[LET\] pe \d+ -> pe \d+: \d+ cells, \d+ particles \(([\d.]+) KB\)', line)
        if m and it >= 0: let[it] = let.get(it, 0.0) + float(m.group(1))
    return dict(place=place, runs=runs, let=let, dur=dur, npieces=npieces)

def main(path, out, title):
    d = parse(path)
    its = sorted(d['place'])
    if not its:
        print("no [TPMAP] lines in %s -- was BARNES_TP_MAP set?" % path); return
    n = d['npieces']
    lo, hi = min(its), max(its)
    img = np.full((n, hi - lo + 1), np.nan)
    for k in its:
        for pe, a, b in d['place'][k]:
            img[a:b + 1, k - lo] = pe
    npe = int(np.nanmax(img)) + 1

    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1, 1, 1.2]})
    cmap = plt.get_cmap('tab10', max(npe, 2))
    ax[0].imshow(img, aspect='auto', origin='lower', interpolation='nearest',
                 cmap=cmap, vmin=-0.5, vmax=npe - 0.5,
                 extent=[lo - 0.5, hi + 0.5, -0.5, n - 0.5])
    ax[0].set_ylabel("tree piece index\n(= position on the SFC)")
    ax[0].set_title(title)

    ax[1].plot(its, [d['runs'][k] for k in its], color='#333333', lw=1.2)
    ax[1].axhline(npe, color='#2a9d8f', ls='--', lw=1,
                  label="%d runs = one interval per PE" % npe)
    ax[1].set_ylabel("contiguous\nruns")
    ax[1].legend(frameon=False, fontsize=8)

    lk = sorted(d['let'])
    if lk:
        ax[2].plot(lk, [d['let'][k] / 1024.0 for k in lk], color='#c1121f', lw=1.2)
    ax[2].set_ylabel("LET pushed\n(MB / iteration)")

    y = [1000 * t for t in d['dur']][1:]
    ax[3].plot(range(1, len(y) + 1), y, color='#0353a4', lw=1.0)
    ax[3].set_ylabel("wall time\n(ms / iteration)")
    ax[3].set_xlabel("iteration")
    for a in ax: a.grid(alpha=0.2, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote %s" % out)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
