#!/usr/bin/env python3
"""Render lbdriver.json as one grid per phase, each cell coloured by its PE.

Standard library only -- no matplotlib, no numpy. Writes a single self-contained
HTML file that opens locally in a browser and is also publishable as-is.

  python3 plot_map.py [lbdriver.json] [lbmap.html]
"""

import json
import random
import sys

# Categorical palette for PE identity. These four hues are the only 4-subset of
# the reference categorical palette that clears the all-pairs CVD and
# normal-vision separation floors in BOTH light and dark modes -- which is the
# pairlist that applies here, because after balancing any two PEs can end up
# adjacent on the grid. Yellow and magenta sit under 3:1 contrast on the light
# surface, so the table view below the grids is required, not optional.
PE_LIGHT = ["#2a78d6", "#eda100", "#e87ba4", "#008300"]
PE_DARK = ["#3987e5", "#c98500", "#d55181", "#008300"]

# Beyond four PEs no palette can name each one, and a map does not need it to:
# what has to be told apart is a region from the regions it touches. So the
# page is coloured as a map -- no two PEs that share a boundary in ANY phase
# get the same colour, one colouring for the whole page so a PE keeps its
# colour from panel to panel -- from the four hues above and then these, which
# a greedy colouring reaches only when the adjacency needs them. Identity is on
# hover and in the table; the boundary lines carry the partition regardless.
#
# Validated against the four above (scripts/validate_palette.js, all pairs).
# Light: the purple passes every check, the cyan passes with a CVD warning,
# and the two after them clear the normal-vision floor but not the CVD floor
# against the green -- no fifth hue does. Dark: purple and teal clear the
# normal-vision floor; the last two clear neither against the pink and the
# amber. They come last, so only the few PEs whose neighbourhoods force a
# seventh or eighth colour reach them, and for those pairs the boundary lines
# are the separation.
EXTRA_LIGHT = ["#7f3f98", "#17becf", "#e0521f", "#a52a2a"]
EXTRA_DARK = ["#7b4fc4", "#29a89a", "#a9463c", "#a86b2f"]

CELL = 15
PAD = 1


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def colour_pes(phases, nx, ny, npes):
    """Colour index per PE. The PE itself while the four-hue palette can name
    every PE; otherwise a DSATUR colouring of the union, over all phases, of
    the PE adjacency on the grid."""
    if npes <= len(PE_LIGHT):
        return list(range(npes))
    adj = [set() for _ in range(npes)]

    def link(a, b):
        if a != b and 0 <= a < npes and 0 <= b < npes:
            adj[a].add(b)
            adj[b].add(a)

    for ph in phases:
        m = ph["map"]
        for i in range(nx):
            for j in range(ny):
                a = m[i * ny + j]
                if i + 1 < nx:
                    link(a, m[(i + 1) * ny + j])
                if j + 1 < ny:
                    link(a, m[i * ny + j + 1])
    def dsatur():
        colour = [-1] * npes
        sat = [set() for _ in range(npes)]
        for _ in range(npes):
            best = max((pe for pe in range(npes) if colour[pe] < 0),
                       key=lambda pe: (len(sat[pe]), len(adj[pe])))
            c = 0
            while c in sat[best]:
                c += 1
            colour[best] = c
            for nb in adj[best]:
                sat[nb].add(c)
        return colour

    def greedy(order):
        colour = [-1] * npes
        for pe in order:
            used = set(colour[nb] for nb in adj[pe] if colour[nb] >= 0)
            c = 0
            while c in used:
                c += 1
            colour[pe] = c
        return colour

    # Fewer colours is fewer weakly-separated pairs, so a handful of restarts
    # is worth their milliseconds: largest-degree-first with random ties.
    best = dsatur()
    rnd = random.Random(1)
    order = list(range(npes))
    for _ in range(40):
        rnd.shuffle(order)
        order.sort(key=lambda pe: -len(adj[pe]))
        cand = greedy(order)
        if max(cand) < max(best):
            best = cand
    return best


def pieces(m, nx, ny, npes):
    """Connected pieces (4-connectivity) beyond one per PE: how many PEs'
    regions have fallen apart, and into how many extra bits."""
    seen = [False] * (nx * ny)
    count = [0] * npes
    for s in range(nx * ny):
        if seen[s] or not (0 <= m[s] < npes):
            continue
        count[m[s]] += 1
        stack = [s]
        seen[s] = True
        while stack:
            k = stack.pop()
            i, j = k // ny, k % ny
            for ii, jj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= ii < nx and 0 <= jj < ny:
                    kk = ii * ny + jj
                    if not seen[kk] and m[kk] == m[k]:
                        seen[kk] = True
                        stack.append(kk)
    return sum(c - 1 for c in count if c > 1)


def grid_svg(phase, nx, ny, npes, colour):
    """One phase as a grid of cells, plus partition and hot-region outlines."""
    m = phase["map"]
    w = phase["weights"]
    side = CELL
    W = nx * side
    H = ny * side

    def pe_at(i, j):
        return m[i * ny + j] if 0 <= i < nx and 0 <= j < ny else -2

    def hot_at(i, j):
        return w[i * ny + j] > 1.0 if 0 <= i < nx and 0 <= j < ny else False

    out = []
    out.append(
        '<svg class="grid" viewBox="-1 -1 %d %d" role="img" '
        'aria-label="Object to PE mapping, %s">' % (W + 2, H + 2, esc(phase["name"])))

    # Cells, as horizontal runs of one PE and one weight. x is the horizontal
    # axis, y runs downward. A run is one element instead of one per cell, which
    # is what keeps a 128x128 grid times several phases inside a page that a
    # browser will still open; the boundary paths below separate PEs.
    for j in range(ny):
        i = 0
        while i < nx:
            pe = m[i * ny + j]
            wt = w[i * ny + j]
            i2 = i
            while i2 + 1 < nx and m[(i2 + 1) * ny + j] == pe and w[(i2 + 1) * ny + j] == wt:
                i2 += 1
            cls = "c%d" % colour[pe] if 0 <= pe < npes else "cna"
            span = "(%d, %d)" % (i, j) if i2 == i else "(%d&#8211;%d, %d)" % (i, i2, j)
            out.append(
                '<rect class="cell %s" x="%d" y="%d" width="%d" height="%d">'
                '<title>%s &#183; PE %d &#183; weight %g</title></rect>'
                % (cls, i * side, j * side, (i2 - i + 1) * side - PAD, side - PAD, span,
                   pe, wt))
            i = i2 + 1

    # Partition boundaries: an edge wherever the neighbour belongs to another PE.
    # This is the secondary encoding -- the partition stays legible even where two
    # fills are hard to tell apart.
    seg = []
    for i in range(nx):
        for j in range(ny):
            pe = pe_at(i, j)
            x0, y0 = i * side, j * side
            if pe_at(i - 1, j) != pe:
                seg.append("M%d %dV%d" % (x0, y0, y0 + side))
            if pe_at(i, j - 1) != pe:
                seg.append("M%d %dH%d" % (x0, y0, x0 + side))
    out.append('<path class="bound" d="%s"/>' % "".join(seg))

    # The heavy region, where there is one, outlined so the imbalance can be read
    # against the partition rather than guessed at.
    if any(v > 1.0 for v in w):
        hseg = []
        for i in range(nx):
            for j in range(ny):
                if not hot_at(i, j):
                    continue
                x0, y0 = i * side, j * side
                if not hot_at(i - 1, j):
                    hseg.append("M%d %dV%d" % (x0, y0, y0 + side))
                if not hot_at(i + 1, j):
                    hseg.append("M%d %dV%d" % (x0 + side, y0, y0 + side))
                if not hot_at(i, j - 1):
                    hseg.append("M%d %dH%d" % (x0, y0, x0 + side))
                if not hot_at(i, j + 1):
                    hseg.append("M%d %dH%d" % (x0, y0 + side, x0 + side))
        out.append('<path class="hot" d="%s"/>' % "".join(hseg))

    out.append("</svg>")
    return "\n".join(out)


def stats(phase, npes):
    counts = [0] * npes
    loads = [0.0] * npes
    for k, pe in enumerate(phase["map"]):
        if 0 <= pe < npes:
            counts[pe] += 1
            loads[pe] += phase["weights"][k]
    total = sum(loads)
    avg = total / npes if npes else 0.0
    mx = max(loads) if loads else 0.0
    return counts, loads, (mx / avg if avg > 0 else 0.0)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "lbdriver.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "lbmap.html"

    with open(src) as f:
        data = json.load(f)

    nx, ny, npes = data["nx"], data["ny"], data["npes"]
    phases = data["phases"]
    simulated = data.get("tool") == "lbsim"

    colour = colour_pes(phases, nx, ny, npes)
    ncol = max(colour) + 1
    light = PE_LIGHT + EXTRA_LIGHT
    dark = PE_DARK + EXTRA_DARK
    if ncol > len(light):
        print("plot_map: the map needs %d colours and the palette has %d; some "
              "neighbouring PEs will share a colour (the boundary lines still "
              "separate them)" % (ncol, len(light)))
    pe_light = "\n".join(
        "  --pe-%d: %s;" % (c, light[c % len(light)]) for c in range(ncol))
    pe_dark = "\n".join(
        "  --pe-%d: %s;" % (c, dark[c % len(dark)]) for c in range(ncol))
    cell_rules = "\n".join(
        ".c%d { fill: var(--pe-%d); }" % (c, c) for c in range(ncol))
    sw_rules = "\n".join(
        ".sw.c%d { background: var(--pe-%d); }" % (c, c) for c in range(ncol))

    panels = []
    for n, ph in enumerate(phases):
        counts, loads, ratio = stats(ph, npes)
        extra = pieces(ph["map"], nx, ny, npes)

        # What the balancer was handed. Between two steps the weights can change
        # under a fixed mapping, so the imbalance this step started from is the
        # PREVIOUS mapping scored with THIS phase's weights -- without it the
        # third panel shows a good final number and no sign of the problem it
        # solved.
        before = ""
        if n > 0:
            prev = {"map": phases[n - 1]["map"], "weights": ph["weights"]}
            _, _, pre_ratio = stats(prev, npes)
            moved = sum(1 for a, b in zip(phases[n - 1]["map"], ph["map"]) if a != b)
            before = ('      <div><dt>was</dt><dd>%.2f&#215;</dd></div>\n'
                      '      <div><dt>objects moved</dt><dd>%d</dd></div>\n'
                      % (pre_ratio, moved))

        panels.append(
            '<figure class="panel">\n'
            '  <figcaption>\n'
            '    <span class="step">Step %d</span>\n'
            '    <h2>%s</h2>\n'
            '    <dl class="facts">\n'
            '%s'
            '      <div><dt>max/avg load</dt><dd class="key">%.2f&#215;</dd></div>\n'
            '      <div><dt>objects per PE</dt><dd>%d&#8211;%d</dd></div>\n'
            '      <div><dt>detached pieces</dt><dd>%d</dd></div>\n'
            '    </dl>\n'
            '  </figcaption>\n'
            '%s\n'
            '</figure>' % (n, esc(ph["name"]), before, ratio, min(counts), max(counts),
                           extra, grid_svg(ph, nx, ny, npes, colour)))

    if npes <= len(PE_LIGHT):
        legend = "\n".join(
            '<li><span class="sw c%d"></span>PE %d</li>' % (colour[i], i)
            for i in range(npes))
    else:
        legend = ('<li>%d PEs, %d colours: a colour separates a PE from its '
                  'neighbours, it does not name it &#8212; hover a cell for the PE</li>'
                  % (npes, ncol))

    head = "".join("<th>%s</th>" % esc(p["name"]) for p in phases)
    per_phase = [stats(ph, npes) for ph in phases]
    rows = []
    for pe in range(npes):
        cells = []
        for counts, loads, _ in per_phase:
            cells.append("<td>%d obj &#183; %.0f load</td>" % (counts[pe], loads[pe]))
        rows.append('<tr><th scope="row"><span class="sw c%d"></span>PE %d</th>%s</tr>'
                    % (colour[pe], pe, "".join(cells)))

    # A long table folds; the summary line says what is inside.
    if npes > 16:
        table_open = ('<details class="tablewrap"><summary>Objects and total weight '
                      'per PE, %d rows</summary>' % npes)
        table_close = '</details>'
    else:
        table_open = '<div class="tablewrap">'
        table_close = '</div>'

    if simulated:
        where = ("%d virtual nodes, DiffusionLB run offline by <code>lbsim</code> on "
                 "its own decision code" % npes)
    else:
        where = "%d PEs" % npes

    html = TEMPLATE % {
        "nx": nx, "ny": ny, "nobj": nx * ny, "npes": npes, "where": where,
        "pe_light": pe_light, "pe_dark": pe_dark, "cell_rules": cell_rules,
        "sw_rules": sw_rules,
        "panels": "\n".join(panels), "legend": legend,
        "thead": head, "rows": "\n".join(rows),
        "table_open": table_open, "table_close": table_close,
    }

    with open(dst, "w") as f:
        f.write(html)
    print("wrote %s (%d phases, %dx%d on %d PEs, %d colours)"
          % (dst, len(phases), nx, ny, npes, ncol))


TEMPLATE = """<title>Stencil Partition Walk</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --surface: #ffffff;
  --ink: #12161c;
  --ink-2: #4a5464;
  --ink-3: #7b8598;
  --rule: #dfe3ea;
  --bound: #12161c;
  --hot: #12161c;
%(pe_light)s
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #10131a;
    --surface: #171b24;
    --ink: #f2f4f8;
    --ink-2: #a9b2c2;
    --ink-3: #6f7889;
    --rule: #262c38;
    --bound: #f2f4f8;
    --hot: #f2f4f8;
%(pe_dark)s
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #10131a;
  --surface: #171b24;
  --ink: #f2f4f8;
  --ink-2: #a9b2c2;
  --ink-3: #6f7889;
  --rule: #262c38;
  --bound: #f2f4f8;
  --hot: #f2f4f8;
%(pe_dark)s
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: "IBM Plex Sans", system-ui, -apple-system, sans-serif;
  font-size: 15px;
  line-height: 1.55;
}
.wrap { max-width: 1240px; margin: 0 auto; padding: 40px 28px 64px; }

header { border-bottom: 1px solid var(--rule); padding-bottom: 22px; }
.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--ink-3); margin: 0 0 10px;
}
h1 { font-size: 30px; font-weight: 600; margin: 0 0 10px; text-wrap: balance;
     letter-spacing: -.015em; }
.lede { margin: 0; color: var(--ink-2); max-width: 66ch; }

.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; list-style: none;
          margin: 24px 0 0; padding: 0;
          font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12.5px;
          color: var(--ink-2); align-items: center; }
.legend li { display: flex; align-items: center; gap: 7px; }
.sw { width: 11px; height: 11px; border-radius: 2px; display: inline-block;
      flex: none; }
%(cell_rules)s
%(sw_rules)s
.legend .marks { color: var(--ink-3); }

.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 26px; margin: 32px 0 0; }
.panel { margin: 0; background: var(--surface); border: 1px solid var(--rule);
         border-radius: 6px; padding: 18px 18px 20px;
         display: flex; flex-direction: column; gap: 14px; }
.panel figcaption { display: flex; flex-direction: column; gap: 8px; }
.step { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px;
        letter-spacing: .09em; text-transform: uppercase; color: var(--ink-3); }
.panel h2 { font-size: 15.5px; font-weight: 600; margin: 0; text-wrap: balance; }

.facts { display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 2px 0 0; }
.facts div { display: flex; flex-direction: column; }
.facts dt { font-size: 11px; color: var(--ink-3); text-transform: uppercase;
            letter-spacing: .07em;
            font-family: "IBM Plex Mono", ui-monospace, monospace; }
.facts dd { margin: 0; font-size: 17px; font-weight: 500;
            font-variant-numeric: tabular-nums; color: var(--ink-2); }
.facts dd.key { color: var(--ink); font-weight: 600; }

.grid { width: 100%%; height: auto; display: block; }
.cell { stroke: none; }
.cell:hover { stroke: var(--ink); stroke-width: 1.5px; }
.cna { fill: var(--rule); }
.bound { fill: none; stroke: var(--bound); stroke-width: 1.4px; opacity: .85; }
.hot { fill: none; stroke: var(--hot); stroke-width: 2px; stroke-dasharray: 4 3;
       opacity: .9; }

.tablewrap { margin: 34px 0 0; overflow-x: auto; }
details.tablewrap summary { cursor: pointer; color: var(--ink-2); font-size: 13.5px;
                            padding: 6px 0; }
details.tablewrap summary:focus-visible { outline: 2px solid var(--ink-3); }
table { border-collapse: collapse; width: 100%%; font-size: 13.5px;
        font-variant-numeric: tabular-nums; }
caption { text-align: left; color: var(--ink-2); font-size: 13px;
          padding-bottom: 10px; }
th, td { text-align: left; padding: 8px 14px 8px 0; border-bottom: 1px solid var(--rule);
         white-space: nowrap; }
thead th { font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
           color: var(--ink-3);
           font-family: "IBM Plex Mono", ui-monospace, monospace; font-weight: 500; }
tbody th { font-weight: 500; }
tbody th .sw { margin-right: 8px; vertical-align: -1px; }

footer { margin: 34px 0 0; padding-top: 18px; border-top: 1px solid var(--rule);
         color: var(--ink-3); font-size: 12.5px; max-width: 72ch; }
code { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .93em;
       color: var(--ink-2); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Charm++ &#183; ck-ldb &#183; lbdriver</p>
  <h1>Stencil Partition Walk</h1>
  <p class="lede">A %(nx)d&#215;%(ny)d stencil of %(nobj)d chares on %(where)s.
  Each square is one chare, coloured by the PE holding it. The partition is
  balanced once with the weights flat, then the load is concentrated into one
  region and DiffusionLB has to repair it.</p>
  <ul class="legend">
    %(legend)s
    <li class="marks">&#9472; partition boundary</li>
    <li class="marks">&#9476; heavy region (12&#215; weight)</li>
  </ul>
</header>

<div class="panels">
%(panels)s
</div>

%(table_open)s
<table>
  <caption>Objects and total weight per PE. Some fills sit under 3:1
  contrast on a light ground, so the figures are given here as well as in the
  grids.</caption>
  <thead><tr><th scope="col">PE</th>%(thead)s</tr></thead>
  <tbody>
%(rows)s
  </tbody>
</table>
%(table_close)s

<footer>Generated by <code>tests/charm++/load_balancing/lbdriver</code>. Chare
loads are declared with <code>setObjTime</code> rather than measured, so the
input is identical for every balancer and every run; the communication graph is
whatever the runtime recorded from the stencil's ghost exchanges.</footer>
</div>
"""

if __name__ == "__main__":
    main()
