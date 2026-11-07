#!/usr/bin/env python3
"""smartmoe_check.py <path to SmartMoE-AE/src/fastmoe/fmoe/smartmoe/mapping.py>

Checks moe_ref.py's --place smartmoe (place_smartmoe) against the function it
ports, greedy_gen_mapping_from_history of the SmartMoE artifact
(github.com/thu-pacman/SmartMoE-AE, commit e19b3a9), on random instances: Zipf
loads, small-integer loads with many ties, all-equal loads and uniform loads,
for 32 and 64 experts on 4 and 8 ranks and the artifact's own 8 and 16. The
artifact's mapping is a slot index (rank * experts_per_rank + local); the
check compares the rank of every expert and the accept-or-keep decision.
Prints the instance count, the mismatches (expected 0) and how many instances
changed the placement, so both outcomes of the acceptance rule are exercised.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_ref import place_smartmoe  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ns = {}
    exec(open(sys.argv[1]).read().replace("import torch", ""), ns)   # torch is unused by the greedy path
    theirs = ns["greedy_gen_mapping_from_history"]
    rng = np.random.default_rng(7)
    n = bad = changed = 0
    for (E, P) in [(32, 4), (32, 8), (64, 4), (64, 8), (8, 4), (8, 8), (16, 8)]:
        per = E // P
        for it in range(400):
            owner = rng.permutation(np.repeat(np.arange(P), per))
            kind = it % 4
            if kind == 0:
                hist = rng.zipf(1.3, E).astype(float)
            elif kind == 1:
                hist = rng.integers(0, 5, E).astype(float)
            elif kind == 2:
                hist = np.full(E, 100.0)
            else:
                hist = rng.random(E) * 1000
            slots = [0] * P
            cur_map = []
            for e in range(E):
                r = int(owner[e])
                cur_map.append(r * per + slots[r])
                slots[r] += 1
            th = np.array([m // per for m in theirs(per, P, cur_map, hist.copy())])
            mine = place_smartmoe(owner, hist, P)
            n += 1
            if not np.array_equal(th, mine):
                bad += 1
                if bad <= 3:
                    print("mismatch: E %d P %d kind %d\n  theirs %s\n  ours   %s" % (E, P, kind, th.tolist(), mine.tolist()))
            changed += int(not np.array_equal(mine, owner))
    print("instances %d, mismatches %d, placements changed %d" % (n, bad, changed))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
