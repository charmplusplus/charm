#!/usr/bin/env python3
"""Summarise lbdriver.json runs: locality and balance of the last phase
(after DiffusionLB) relative to the phase before it (after MetisLB).

cut%   : change in 4-neighbour edge cut, last phase vs previous, in percent
pcs+   : connected pieces (4-connectivity) in the last phase minus npes,
         i.e. how many PEs' regions fell apart
imb_bef: max/avg of the LAST phase's weights on the PREVIOUS phase's map --
         the imbalance the diffusion step actually saw
imb_aft: max/avg on the last phase's map
moves  : objects whose PE differs between the two phases
"""
import json
import sys
from collections import deque


def cut(m, nx, ny):
    c = 0
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            if i + 1 < nx and m[k] != m[k + 1]:
                c += 1
            if j + 1 < ny and m[k] != m[k + nx]:
                c += 1
    return c


def pieces(m, nx, ny):
    seen = [False] * (nx * ny)
    n = 0
    for s in range(nx * ny):
        if seen[s]:
            continue
        n += 1
        pe = m[s]
        q = deque([s])
        seen[s] = True
        while q:
            k = q.popleft()
            i, j = k % nx, k // nx
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    kk = jj * nx + ii
                    if not seen[kk] and m[kk] == pe:
                        seen[kk] = True
                        q.append(kk)
    return n


def imbalance(m, weights, npes):
    loads = [0.0] * npes
    for k, pe in enumerate(m):
        if 0 <= pe < npes:
            loads[pe] += weights[k]
    avg = sum(loads) / npes
    return max(loads) / avg if avg > 0 else 0.0


def summarise(path):
    with open(path) as f:
        d = json.load(f)
    nx, ny, npes = d["nx"], d["ny"], d["npes"]
    a, b = d["phases"][-2], d["phases"][-1]
    ca, cb = cut(a["map"], nx, ny), cut(b["map"], nx, ny)
    moves = sum(1 for x, y in zip(a["map"], b["map"]) if x != y)
    return dict(
        cut_incr=100.0 * (cb - ca) / ca if ca else 0.0,
        extra_pieces=pieces(b["map"], nx, ny) - npes,
        imb_before=imbalance(a["map"], b["weights"], npes),
        imb_after=imbalance(b["map"], b["weights"], npes),
        moves=moves,
    )


def main():
    rows = [(p, summarise(p)) for p in sys.argv[1:]]
    hdr = "%-24s %7s %5s %8s %8s %6s"
    fmt = "%-24s %+7.1f %5d %8.2f %8.2f %6d"
    print(hdr % ("file", "cut%", "pcs+", "imb_bef", "imb_aft", "moves"))
    for p, r in rows:
        print(fmt % (p.split("/")[-1], r["cut_incr"], r["extra_pieces"],
                     r["imb_before"], r["imb_after"], r["moves"]))
    if len(rows) > 1:
        n = len(rows)

        def mean(k):
            return sum(r[k] for _, r in rows) / n

        print("%-24s %+7.1f %5.1f %8.2f %8.2f %6.1f" % (
            "mean", mean("cut_incr"), mean("extra_pieces"),
            mean("imb_before"), mean("imb_after"), mean("moves")))


if __name__ == "__main__":
    main()
