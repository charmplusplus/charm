# 64-bit object id redesign: performance record (charm #3994)

Benchmark: `benchmarks/charm++/objid/objid_bench` (one SUMMARY line per run; medians
of 5 timed repetitions, N = 20000 elements, microseconds per operation). BASE is the
reviewed line just before PR 1a (worktree wt-objid-pr0 build); NEW is the branch tip
named. Machine: Kale's Mac (8 cores, reconverse-darwin-arm8, --with-production), shapes
1 process x 8 PEs (1p; sub-microsecond phases vary 20-50% there) and 4 processes x 2
PEs (4p; ranges within a few percent). Bounded array: `setBounds(N)` with no
numInitial, so the default map is RRMap and home != host for 7/8 of the elements.

## 2026-10-01, PR 1a + 1b tip (after the hops >= 1 repair change, commit on objid-pr1b-allocator)

bounded (packed ids)

| phase | 1p BASE | 1p NEW | ratio | 4p BASE | 4p NEW | ratio |
|---|---|---|---|---|---|---|
| insert | 0.484 | 0.561 | 1.16 (noise) | 4.145 | 3.913 | 0.94 |
| cold | 0.158 | 0.295 | 1.87 | 6.963 | 9.815 | 1.41 |
| warm | 0.215 | 0.110 | 0.51 | 6.953 | 3.892 | 0.56 |
| migrate | 0.757 | 0.721 | 0.95 | 8.792 | 8.590 | 0.98 |
| stale (1 hop) | 0.169 | 0.259 | 1.53 | 6.308 | 9.668 | 1.53 |
| stalewarm | 0.154 | 0.108 | 0.70 | 6.309 | 3.785 | 0.60 |
| stale2 (2 hops) | 0.179 | 0.315 | 1.76 | 7.146 | 12.952 | 1.81 |
| stale2warm | 0.166 | 0.125 | 0.75 | 6.025 | 3.804 | 0.63 |
| bcast | 1568 | 1528 | 0.97 | 1148 | 1139 | 0.99 |

unbounded (hashed ids)

| phase | 1p BASE | 1p NEW | ratio | 4p BASE | 4p NEW | ratio |
|---|---|---|---|---|---|---|
| insert | 0.690 | 0.848 | 1.23 (noise) | 4.051 | 3.923 | 0.97 |
| cold | 0.340 | 0.552 | 1.62 | 10.905 | 11.975 | 1.10 |
| warm | 0.125 | 0.121 | 0.97 | 3.893 | 3.825 | 0.98 |
| migrate | 0.750 | 0.896 | 1.19 (borderline) | 8.627 | 8.302 | 0.96 |
| stale (1 hop) | 0.166 | 0.259 | 1.56 | 6.660 | 9.600 | 1.44 |
| stalewarm | 0.162 | 0.123 | 0.76 | 6.675 | 3.821 | 0.57 |
| stale2 (2 hops) | 0.256 | 0.390 | 1.52 | 11.508 | 12.398 | 1.08 |
| stale2warm | 0.129 | 0.150 | 1.16 (noise) | 4.802 | 3.891 | 0.81 |
| bcast | 1420 | 1714 | 1.21 (noise) | 1166 | 1162 | 1.00 |

Reading:
- Unchanged: insert (tranche minting costs no more than the per-PE counter), warm,
  migrate, bcast, and the pingpong benchmark (array phases within 1-3% over three
  alternating reruns).
- Cold and stale rounds cost more on NEW because every forwarded delivery now sends
  one repair message to the sender (ck.C fast path, hops >= 1); the rounds after them
  run at warm speed. BASE never repairs a one-hop stale cache and never learns the
  location of a bounded element whose home is not its host, so it pays the forward on
  every round. Break-even: the second send to an element.
- Unbounded cold on 1p (1.62x) is the known interim cost of every hashed home being
  PE 0 in a single process (design 3.3); on 4p it is 1.10x (homes on 4 rank-0 PEs).
  The process-level directory (design 7) has to erase it; this is its target number.

## Earlier run the same day, before the repair change (1a's explicit requestLocationOnce)

4p bounded: cold 7.26 -> 14.09 (1.94), warm 7.23 -> 4.05 (0.56), stale 6.56 -> 6.78,
stale2 7.35 -> 12.07. 4p unbounded: cold 10.86 -> 12.44 (1.15), stale 6.72 -> 6.73,
stale2 11.6 -> 11.4. The explicit request cost a request and a reply per cold element;
the repair message replaced them (bounded 4p cold 14.09 -> 9.82).

## To do
- Rerun at each PR of the series and at scale on Anvil (2 nodes, 64-128 PEs); add
  rows here with the branch tip.
- Add a shape where home == host for the bounded array (block map with numInitial), the
  common case, which should show neither the cold cost nor the warm gain.
