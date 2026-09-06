#!/usr/bin/env python3
"""selftest.py: checks the Python ports in moe_ref.py against ref/hashcheck
(the example's own routing and init code), bit for bit, on the CPU.
   selftest.py <path to hashcheck binary>"""
import math
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from moe_ref import Router, hmix, init_uniform  # noqa: E402

bad = 0
for (E, T, K, z, drift, seed, pe, steps) in [(8, 64, 1, 1.0, 2, 12345, 0, 5),
                                              (16, 100, 2, 1.0, 3, 12345, 3, 7),
                                              (64, 8192, 1, 1.0, 10, 12345, 2, 12),
                                              (6, 50, 4, 0.0, 0, 99, 1, 3),
                                              (3, 40, 3, 2.0, 1, 7, 0, 4)]:
    out = subprocess.run([sys.argv[1]] + [str(v) for v in (E, T, K, z, drift, seed, pe, steps)],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    hm = [int(v) for v in out[0].split()[1:]]
    want = [hmix(seed, 1), hmix(hmix(seed, 1000), pe), hmix(hmix(hmix(hmix(hmix(seed, 3), pe), 7), 0), 2)]
    if hm != want:
        print("hmix mismatch", hm, want)
        bad += 1
    r = Router(E, T, K, z, drift, seed, pe)
    for line in out[1:1 + steps]:
        f = line.split()
        step = int(f[1])
        got = np.array([int(v) for v in f[2:]])
        se, counts = r.route(step)
        if not np.array_equal(se, got):
            print("route mismatch E=%d T=%d K=%d z=%g step %d: %d of %d slots differ" % (
                E, T, K, z, step, int((se != got).sum()), got.size))
            bad += 1
    f = out[1 + steps].split()
    assert f[0] == "init"
    bits = [int(v) for v in f[1:]]
    idx = [0, 1, 2, 3, 4, 5, 6, 7, 1000, 123456, 16777215, 16777216]
    n = 16777217
    for q, (sd, bound) in enumerate([(hmix(hmix(seed, 1), 0), 1.0 / math.sqrt(2048.0)),
                                     (hmix(hmix(seed, 1000), pe), math.sqrt(3.0))]):
        v = init_uniform(n, sd, bound, torch.device("cpu"))
        got = v[idx].view(torch.int32).numpy().astype(np.int64) & 0xFFFFFFFF
        want = np.array(bits[q * 12:(q + 1) * 12])
        if not np.array_equal(got, want):
            print("init mismatch (set %d): got %s want %s" % (q, got, want))
            bad += 1
print("selftest:", "FAILED %d" % bad if bad else "all routing, hashing and init values match")
sys.exit(1 if bad else 0)
