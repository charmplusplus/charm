#!/usr/bin/env python3
"""Render lbdriver.json as one grid per phase, each cell coloured by its PE.

Standard library only -- no matplotlib, no numpy. Writes a single self-contained
HTML file that opens locally in a browser and is also publishable as-is.

  python3 plot_map.py [lbdriver.json] [lbmap.html]
"""

import json
import sys

# Categorical palette for PE identity. These four hues are the only 4-subset of
# the reference categorical palette that clears the all-pairs CVD and
# normal-vision separation floors in BOTH light and dark modes -- which is the
# pairlist that applies here, because after balancing any two PEs can end up
# adjacent on the grid. Yellow and magenta sit under 3:1 contrast on the light
# surface, so the table view below the grids is required, not optional.
PE_LIGHT = ["#2a78d6", "#eda100", "#e87ba4", "#008300"]
PE_DARK = ["#3987e5", "#c98500", "#d55181", "#008300"]

CELL = 15
PAD = 1


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def grid_svg(phase, nx, ny, npes, idx):
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

    # Cells. x is the horizontal axis, y runs downward.
    for i in range(nx):
        for j in range(ny):
            pe = m[i * ny + j]
            cls = "c%d" % (pe % len(PE_LIGHT)) if pe >= 0 else "cna"
            out.append(
                '<rect class="cell %s" x="%d" y="%d" width="%d" height="%d">'
                '<title>(%d, %d) &#183; PE %d &#183; weight %g</title></rect>'
                % (cls, i * side, j * side, side - PAD, side - PAD, i, j, pe,
                   w[i * ny + j]))

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

    pe_light = "\n".join(
        "  --pe-%d: %s;" % (i, PE_LIGHT[i % len(PE_LIGHT)]) for i in range(npes))
    pe_dark = "\n".join(
        "  --pe-%d: %s;" % (i, PE_DARK[i % len(PE_DARK)]) for i in range(npes))
    cell_rules = "\n".join(
        ".c%d { fill: var(--pe-%d); }" % (i, i) for i in range(npes))

    panels = []
    for n, ph in enumerate(phases):
        counts, loads, ratio = stats(ph, npes)

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
            '    </dl>\n'
            '  </figcaption>\n'
            '%s\n'
            '</figure>' % (n, esc(ph["name"]), before, ratio, min(counts), max(counts),
                           grid_svg(ph, nx, ny, npes, n)))

    legend = "\n".join(
        '<li><span class="sw c%d"></span>PE %d</li>' % (i, i) for i in range(npes))

    head = "".join("<th>%s</th>" % esc(p["name"]) for p in phases)
    rows = []
    for pe in range(npes):
        cells = []
        for ph in phases:
            counts, loads, _ = stats(ph, npes)
            cells.append("<td>%d obj &#183; %.0f load</td>" % (counts[pe], loads[pe]))
        rows.append('<tr><th scope="row"><span class="sw c%d"></span>PE %d</th>%s</tr>'
                    % (pe, pe, "".join(cells)))

    html = TEMPLATE % {
        "nx": nx, "ny": ny, "nobj": nx * ny, "npes": npes,
        "pe_light": pe_light, "pe_dark": pe_dark, "cell_rules": cell_rules,
        "panels": "\n".join(panels), "legend": legend,
        "thead": head, "rows": "\n".join(rows),
    }

    with open(dst, "w") as f:
        f.write(html)
    print("wrote %s (%d phases, %dx%d on %d PEs)" % (dst, len(phases), nx, ny, npes))


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
.sw.c0 { background: var(--pe-0); }
.sw.c1 { background: var(--pe-1); }
.sw.c2 { background: var(--pe-2); }
.sw.c3 { background: var(--pe-3); }
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
  <p class="lede">A %(nx)d&#215;%(ny)d stencil of %(nobj)d chares on %(npes)d PEs.
  Each square is one chare, coloured by the PE holding it. The partition is
  balanced once with the weights flat, then the load is concentrated into one
  region and a second balancer has to repair it.</p>
  <ul class="legend">
    %(legend)s
    <li class="marks">&#9472; partition boundary</li>
    <li class="marks">&#9476; heavy region (12&#215; weight)</li>
  </ul>
</header>

<div class="panels">
%(panels)s
</div>

<div class="tablewrap">
<table>
  <caption>Objects and total weight per PE. Two of the four fills sit under 3:1
  contrast on a light ground, so the figures are given here as well as in the
  grids.</caption>
  <thead><tr><th scope="col">PE</th>%(thead)s</tr></thead>
  <tbody>
%(rows)s
  </tbody>
</table>
</div>

<footer>Generated by <code>tests/charm++/load_balancing/lbdriver</code>. Chare
loads are declared with <code>setObjTime</code> rather than measured, so the
input is identical for every balancer and every run; the communication graph is
whatever the runtime recorded from the stencil's ghost exchanges.</footer>
</div>
"""

if __name__ == "__main__":
    main()
