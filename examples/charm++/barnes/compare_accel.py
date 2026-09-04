#!/usr/bin/env python3
"""Compare two BARNES_ACCEL_DUMP runs.

Each run writes one file per PE, <prefix>.<pe>, holding
    key ax ay az phi
for every particle that PE held on the last iteration. Which PE held what does
not have to match between the two runs -- only the force on a given key does --
so the files are folded into one dict keyed by SFC key.

Reports the relative acceleration error against the first run, which is taken
as the reference:

    err_i = |a_i - ref_i| / <|ref|>

normalised by the mean reference magnitude rather than per-particle, because
particles near the centre of mass have near-zero acceleration and a per-particle
relative error there is all noise. This is the convention the tree-code
literature reports.

Usage: compare_accel.py <reference-prefix> <test-prefix> [more prefixes...]
"""
import sys, glob, math


def load(prefix):
    out = {}
    # Numeric suffixes only: the run's own .log sits beside these.
    files = sorted(f for f in glob.glob(prefix + ".*")
                   if f.rsplit(".", 1)[-1].isdigit())
    if not files:
        sys.exit("no files matching %s.*" % prefix)
    for fn in files:
        with open(fn) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                key = int(parts[0])
                out[key] = tuple(float(x) for x in parts[1:])
    return out


def compare(ref, test, name):
    common = ref.keys() & test.keys()
    if not common:
        print("%-28s NO OVERLAPPING KEYS (%d vs %d particles)"
              % (name, len(ref), len(test)))
        return
    missing = len(ref) - len(common)

    mags = [math.sqrt(ref[k][0]**2 + ref[k][1]**2 + ref[k][2]**2) for k in common]
    mean_mag = sum(mags) / len(mags)
    if mean_mag == 0.0:
        print("%-28s reference accelerations are all zero" % name)
        return

    errs = []
    for k in common:
        r, t = ref[k], test[k]
        d = math.sqrt((t[0]-r[0])**2 + (t[1]-r[1])**2 + (t[2]-r[2])**2)
        errs.append(d / mean_mag)
    errs.sort()

    n = len(errs)
    mean = sum(errs) / n
    print("%-28s n=%-8d mean=%.3e  median=%.3e  99th=%.3e  max=%.3e%s"
          % (name, n, mean, errs[n//2], errs[int(0.99*n)], errs[-1],
             "  MISSING %d" % missing if missing else ""))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref = load(sys.argv[1])
    print("reference: %s (%d particles)" % (sys.argv[1], len(ref)))
    for prefix in sys.argv[2:]:
        compare(ref, load(prefix), prefix)
