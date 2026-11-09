#!/usr/bin/env python3
"""gaps.py <gputrace dir> [--rounds 23,44]: the per-step periods of the dispatcher on PE 0 from the runtime's trace,
and the gap after each named step (the balancing round's cost on the step): the time from that step's start to the
next step's start, against the median step around it."""
import glob, re, sys, statistics
d = sys.argv[1]
rounds = [int(x) for x in (sys.argv[sys.argv.index("--rounds") + 1] if "--rounds" in sys.argv else "23,44").split(",")]
pe0 = sorted(glob.glob(d + "/gputrace.*.pe0"))[0]
steps = []
ep = None   # Dispatcher::runStep's entry id, from the trace's own name lines (it moves when the .ci changes)
for line in open(pe0):
    if line.startswith("P ") and line.rstrip().endswith(" Dispatcher::runStep"):
        ep = line.split()[1]
    elif line.startswith("E") and ep is not None:
        f = line.split()
        if f[3] == ep: steps.append(int(f[1]))
periods = [(steps[i + 1] - steps[i]) / 1e6 for i in range(len(steps) - 1)]
print("steps on PE 0:", len(steps))
for r in rounds:
    if r - 1 < len(periods):
        before = statistics.median(periods[max(0, r - 6):r - 1]) if r > 1 else float("nan")
        after = statistics.median(periods[r + 2:r + 8]) if r + 8 <= len(periods) else float("nan")
        print("after step %d: %.1f ms to the next step (steps before: %.1f ms median, after: %.1f)" % (r, periods[r - 1], before, after))
big = [(i + 1, p) for i, p in enumerate(periods) if p > 3 * statistics.median(periods)]
print("steps followed by a gap over 3x the median:", ", ".join("%d (%.0f ms)" % b for b in big[:12]) or "none")
