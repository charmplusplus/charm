#!/usr/bin/env python3
"""Per-step timeline for leanmd: step on x, that step's wall time on y.

Timing source is the application's own line, "Step N Benchmark Time X ms/step",
emitted by cell (0,0,0). leanmd prints it and resets stepTime BEFORE the load
balancing block runs (leanmd.ci:131-137), so a balancing step at N is paid for
in the time reported for step N+1 -- the balancing windows drawn here are
shifted accordingly.

Scatter, one x per step, repetitions overplotted so run-to-run spread shows.

usage: lmdplot.py <out.png> "<title>" <firstLdb> <period> label=log[,log...] ...
"""
import sys, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOURS = {'noLB': '#444444', 'sync': '#c1121f', 'async': '#0353a4'}
EXTRA = ['#c1121f', '#0353a4', '#2a9d8f', '#e07a00']

def steps(path):
    out = []
    for line in open(path, errors='ignore'):
        m = re.search(r'Step (\d+) Benchmark Time ([\d.]+) ms/step', line)
        if m: out.append((int(m.group(1)), float(m.group(2))))
    return out

def main(out, title, firstLdb, period, series):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    maxstep = 0
    for n, (label, paths) in enumerate(series):
        c = COLOURS.get(label) or EXTRA[n % len(EXTRA)]
        first = True
        for p in paths.split(','):
            d = steps(p)
            if not d:
                print("no step lines in %s" % p); continue
            xs = [s for s, _ in d]; ys = [t for _, t in d]
            maxstep = max(maxstep, max(xs))
            ax.scatter(xs, ys, label=label if first else None, color=c,
                       marker='x', s=22, linewidths=0.9, alpha=0.85)
            first = False
    # +1: the balancing step's cost lands in the following step's report.
    for s in range(firstLdb, maxstep + 1, period):
        ax.axvline(s + 1, color='#999999', alpha=0.45, lw=0.8, ls=':', zorder=0)
    ax.set_xlabel("step"); ax.set_ylabel("wall time for the step (ms)")
    ax.set_title(title); ax.grid(alpha=0.22, lw=0.5)
    ax.legend(frameon=False, markerscale=1.6)
    ax.text(0.995, -0.16, "dotted lines: the step reporting each balancing step's cost",
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color='#666666')
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("wrote %s" % out)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
         [tuple(a.rsplit('=', 1)) for a in sys.argv[5:]])
