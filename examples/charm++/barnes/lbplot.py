#!/usr/bin/env python3
"""Timeline figure: iteration on x, that iteration's wall time on y.

Scatter, one x per measured iteration. Repetitions of the same arm are plotted
as separate points at the same x rather than averaged, so the run-to-run spread
is visible instead of being hidden inside a mean.

Balancing windows are drawn where they actually fall, detected from the log
when +LBDebug is on and otherwise derived from the schedule (see lbtimeline).

usage: lbplot.py <out.png> "<title>" label=log[,log,...] ...
       the label may contain '=' (e.g. lblag=40=a.log,b.log)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lbtimeline import parse, lb_windows

COLOURS = {'noLB': '#444444', 'sync': '#c1121f', 'async': '#0353a4'}
# Anything not one of the three standard arms (e.g. "lblag=40") gets a colour
# from here, so a comparison does not have to borrow an arm name to be drawn.
EXTRA = ['#c1121f', '#0353a4', '#2a9d8f', '#e07a00']

def main(out, title, series):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bands = []
    for n, (label, paths) in enumerate(series):
        c = COLOURS.get(label) or EXTRA[n % len(EXTRA)]
        first = True
        for path in paths.split(','):
            d = parse(path)
            if not d:
                print("no timing lines in %s" % path); continue
            y = [1000 * t for t in d['iters']][1:]     # iteration 0 is warmup
            x = list(range(1, len(d['iters'])))
            ax.scatter(x, y, label=label if first else None, color=c,
                       marker='x', s=20, linewidths=0.9, alpha=0.85)
            first = False
            if label != 'noLB':
                w, _ = lb_windows(d)
                bands.extend(i for i in w if i >= 1)
    for i in sorted(set(bands)):
        ax.axvline(i, color='#999999', alpha=0.45, lw=0.8, ls=':', zorder=0)
    ax.set_xlabel("iteration")
    ax.set_ylabel("wall time for the iteration (ms)")
    ax.set_title(title)
    ax.grid(alpha=0.22, lw=0.5)
    ax.legend(frameon=False, markerscale=1.6)
    ax.text(0.995, -0.16, "iteration 0 (tree build, GPU init) omitted; dotted lines are balancing windows",
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color='#666666')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote %s" % out)

if __name__ == '__main__':
    # Split on the LAST '=': a label may itself contain one ("lblag=40"), and
    # the paths never do.
    main(sys.argv[1], sys.argv[2], [tuple(a.rsplit('=', 1)) for a in sys.argv[3:]])
