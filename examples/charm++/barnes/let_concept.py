#!/usr/bin/env python3
"""Schematic: contiguous intervals of EQUAL LOAD have very unequal spatial size.

Synthetic points, not measurements.

DiffusionLB does not scatter a node across the ordering. With a 1-D key
registered (barnes registers thisIndex, which is the SFC position) only the
ends of an interval move, and only toward the key-neighbour on that side, so
every PE keeps one contiguous run. That is not the problem.

The problem is that "contiguous" says nothing about "compact". Work in a tree
code follows density: a piece in the core is expensive and spatially tiny, a
piece in the halo is cheap and spatially huge. Equalising load therefore has to
hand the halo PE a much longer stretch of the curve -- still one run, but a box
covering most of the domain. Once a box is that big the opening criterion fails
against nearly everything, and that PE needs fine detail from everywhere.

So the cost is interval WIDTH, not fragmentation, and diffusion to neighbours
produces it just as readily as a global repack would.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rng = np.random.default_rng(7)
NPE, NPIECE = 4, 64
COL = ['#c1121f', '#0353a4', '#2a9d8f', '#e07a00']

pts = np.vstack([
    rng.normal([-0.45, -0.15], 0.10, size=(3000, 2)),
    rng.normal([0.42, 0.18], 0.12, size=(3000, 2)),
    rng.normal([0, 0], 0.58, size=(1200, 2)),
])
pts = pts[(np.abs(pts) < 1).all(axis=1)]

def morton(p):
    q = ((p + 1) / 2 * 1023).astype(np.uint32)
    key = np.zeros(len(p), dtype=np.uint64)
    for b in range(10):
        key |= ((q[:, 0] >> b) & 1).astype(np.uint64) << np.uint64(2 * b + 1)
        key |= ((q[:, 1] >> b) & 1).astype(np.uint64) << np.uint64(2 * b)
    return key

pts = pts[np.argsort(morton(pts))]
piece = np.minimum((np.arange(len(pts)) * NPIECE) // len(pts), NPIECE - 1)
cent = np.array([pts[piece == i].mean(axis=0) for i in range(NPIECE)])
size = np.array([float(np.ptp(pts[piece == i], axis=0).mean()) for i in range(NPIECE)])
# Work per piece follows local density: a compact piece is a deep subtree with
# long interaction lists. Equal particle counts, very unequal cost.
work = 1.0 / (size + 0.02); work /= work.sum()

def contiguous_split(weight):
    """Cut the ordering into NPE contiguous runs with equal total weight --
    exactly what the key-adjacent phase converges to."""
    c = np.cumsum(weight); c /= c[-1]
    own = np.zeros(NPIECE, dtype=int)
    for i in range(NPIECE):
        own[i] = min(NPE - 1, int(c[i] * NPE - 1e-12))
    return own

own_a = contiguous_split(np.ones(NPIECE))   # equal pieces  (blockmap)
own_b = contiguous_split(work)              # equal load    (what LB converges to)

def boxes(own):
    return [np.array([pts[np.isin(piece, np.flatnonzero(own == p))].min(axis=0),
                      pts[np.isin(piece, np.flatnonzero(own == p))].max(axis=0)])
            for p in range(NPE)]

def let_proxy(own, theta=0.5):
    bx = boxes(own); tot = 0
    for b in range(NPE):
        lo, hi = bx[b]
        for i in range(NPIECE):
            if own[i] == b: continue
            d = np.maximum(np.maximum(lo - cent[i], cent[i] - hi), 0).sum()
            if size[i] / max(d, 1e-6) > theta: tot += 1
    return tot, bx

fig = plt.figure(figsize=(12.5, 5.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.62], wspace=0.28)
res = []
for k, (own, lab) in enumerate(((own_a, "(a) equal PARTICLES per PE"),
                                (own_b, "(b) equal LOAD per PE"))):
    ax = fig.add_subplot(gs[0, k])
    n, bx = let_proxy(own); res.append(n)
    for p in range(NPE):
        m = np.isin(piece, np.flatnonzero(own == p))
        ax.scatter(pts[m, 0], pts[m, 1], s=1.4, color=COL[p], alpha=0.55, linewidths=0)
        lo, hi = bx[p]
        ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec=COL[p], lw=1.7, alpha=0.9))
    runs = 1 + int((own[1:] != own[:-1]).sum())
    span = np.array([float(np.mean(b[1] - b[0])) for b in bx])
    ax.set_title("%s\n%d contiguous runs -- box width %.2f to %.2f"
                 % (lab, runs, span.min(), span.max()), fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

ax = fig.add_subplot(gs[0, 2])
ax.bar(["(a)", "(b)"], res, color=['#2a9d8f', '#c1121f'], width=0.55)
for i, v in enumerate(res): ax.text(i, v, " %d" % v, ha='center', va='bottom', fontsize=10)
ax.set_title("cells that must be shipped\n(opening criterion, theta=0.5)", fontsize=10)
ax.set_ylabel("LET entries"); ax.grid(alpha=0.2, axis='y', lw=0.5)
fig.suptitle("Both partitions are contiguous. Equalising load still inflates the boxes  "
             "(schematic, synthetic points)", fontsize=12)
fig.savefig("check/let_concept.png", dpi=140, bbox_inches='tight')
print("(a) %d LET entries, (b) %d" % tuple(res))
