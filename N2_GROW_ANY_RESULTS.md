# Two-node test of Diffusion's connected-piece restriction

Job **22132598**, Delta A40 nodes **gpub006/gpub048**, 2026-09-16.
Default sync LB followed by sync LB with `CHARM_DIFFUSION_GROW_ANY=1`, using
the same binary, nodes, calibrated cost table, NUMA pinning, and diagnostic
settings. Both completed 100 steps, exited zero, and passed energy conservation.
Allocation completed and released after 5m07s.

| metric | default | grow-any |
| --- | ---: | ---: |
| mean steps 42–100, ms | 1530.525 | 1391.806 |
| total application time, s | 148.721 | 141.023 |
| final measured GPU maximum/average | 1.395 | 1.242 |
| moves per Diffusion round | 1471, 508, 865, 3 | 915, 898, 564, 287 |
| total moves | 2847 | 2664 |

Grow-any reduced the mean step time by **9.1%** and whole-application time by
**5.2%** in this single matched pair. These are observed improvements, not a
repeat-run confidence estimate. The override changed placement from the first
round; this is not a test of changing only the last round's decision.

## What it establishes

The local connected-piece candidate restriction is a real obstacle to balancing
this workload. It is not necessary to disable calibrated communication or
migration costs to get further moves accepted. The override changes only the
post-seed frontier filter; quotas and receiver/memory checks remain active.

In the baseline's final round, processes 0, 4, and 5 each accept just one move,
then leave respectively **5.294, 4.409, and 4.520 seconds** of desired load
unshed. All report zero cost rejections, receiver refusals, and failed memory
checks. The selector supplies no next candidate.

With grow-any, the final-round donors (now processes 0 and 3) accept **127 and
160 moves**, satisfy their shedding obligations, and likewise report zero cost
rejections. More useful moves happen late, but **total moves decrease**.

Measured GPU maximum/average by drain:

- Default: 1.852, 1.873, 1.567, 1.395, 1.395.
- Grow-any: 1.852, 1.875, 1.558, 1.310, 1.242.

The last drain measures the interval after the step-80 placement and before the
final step-100 moves. There is no measurement after those final 287 moves.

An independent standalone reproduction uses the actual MetricComm and selection
code, linked with the existing runtime stubs. With three unit-load movable
objects, positive net move benefit, and no local edges, default selection moves
one and stops; grow-any moves all three. Neither rejects a move on cost. This
isolates the empty-frontier stopping mechanism from GPU/runtime behavior.

## Interpretation and next fix

LeanMD's computes communicate through fixed cells. A moved compute can have both
cells remote and therefore no neighbors in the local object graph. After that
compute seeds a move, the selector's frontier is empty. The existing restart
rule handles immovable/zero-load frontiers but explicitly excludes empty ones.
That behavior is inappropriate for these disconnected movable computes.

The evidence supports fixing this candidate restriction, rather than changing
alpha/beta calibration to address the observed refusal. It does **not** prove
that the complete cost model is accurate or that all remaining runtime loss is
explained. Residual measured GPU imbalance is still 24% above average.

The broad grow-any switch is an experimental workaround, not a globally enabled
production fix: the frontier rule was introduced to protect locality on stencil
workloads. A narrower follow-up should allow cost-checked reseeding for an empty
frontier caused by remote/fixed endpoints, and verify that stencil placement
does not regress. That narrower implementation has not been made here.

## Artifacts

`examples/charm++/cuda/gpudirect/leanmd/grow-test/` contains the batch script,
launcher, separate baseline/grow-any rank logs, `job-22132598.out/.err`,
`summary.json`, `summarize.py`, and the standalone empty-frontier reproduction.
Both arms used `../join-test/leanmd-trace`, whose checksum and exact instrumented
source are retained in that directory. No production policy was changed.
