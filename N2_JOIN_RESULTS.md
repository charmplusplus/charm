# Two-node force-join experiment

Delta A40 job **22131761**, 2026-09-16, nodes gpub006/gpub062. Completed
in 5m19s; allocation released. Only two nodes were tested, as requested.
Both noLB and sync completed 100 steps, exited zero, and passed energy
conservation. This is a diagnostic experiment, not a balancing-policy fix.

## Main finding

**The handoff's premise that GPU work is already equalized is false in this
run.** Substantial GPU imbalance remains alongside long force-dependency waits.
The force joins are real, but these results do not establish an unavoidable
fan-in ceiling or exonerate the balancer's placement decisions.

Per-process GPU audit, last measured interval (steps 81–100, following the
step-80 placement change):

| process / GPU | attributed device seconds | kernel busy / audit wall interval |
| --- | ---: | ---: |
| 0 | 19.770 | 66.1% |
| 1 | 9.142 | 30.4% |
| 2 | 8.543 | 28.5% |
| 3 | 12.773 | 42.7% |
| 4 | 22.644 | 75.6% |
| 5 | 21.942 | 73.3% |
| 6 | 18.443 | 61.6% |
| 7 | 16.292 | 54.5% |

Mean is 16.194 seconds; maximum/mean is **1.398**, and maximum/minimum is
**2.65**. The audit reports attributed time essentially equal to kernel busy
time (`util=1.000`), with negligible unowned kernel time. The `T_g` line reports
the mean/lower bound, not the 22.644-second maximum. Busy fractions use each
process's own audit interval and exclude untraced device activities.

Maximum/mean across the five drains: **1.852, 1.872, 1.567, 1.398, 1.398**.
Diffusion moves **1471, 508, 865, 3** objects and reports convergence. The last
drain precedes the final three moves; there is no subsequent measured interval
to evaluate those moves. This does not identify why moves stop: communication
pricing, fixed work, candidate selection, capacity, and convergence criteria
still need inspection. It does establish that GPU balance cannot be assumed.

## What the force trace shows

Four complete snapshots: steps **45, 65, 85, 95**. Each arm contains 110,592
force records (1024 cells × 27 forces × 4), 4096 last-contributor records, and
57,344 records each for compute input, completion, and acknowledgment waits.
All last contributors matched their compute records; no missing local timers.

The following are means over the four snapshots, in milliseconds:

| measurement | noLB | sync |
| --- | ---: | ---: |
| cell position-send start → last force consumed | 1445.8 | 1191.3 |
| first → last force consumed by cell | 1001.9 | 879.6 |
| last contributor: first input posted → inputs ready | 971.5 | 724.4 |
| last contributor: inputs ready → force-send entry | 378.3 | 353.9 |
| last force: receive posted → consumed by cell | 75.2 | 115.7 |

These are **not additive phases of a shared clock**. Computes serve two cells,
their first input may precede the measured cell's send, and work overlaps.
Input wait includes copying, the partner input, scheduling, and potentially
progress behind the previous iteration. Completion wait includes launch,
queued GPU work, execution, and callback scheduling—not pure kernel duration.
Receiver wait includes copying and SDAG consumption delay—not pure transport.

For noLB, lightly loaded receiver processes have mean first-to-last spans
around 1.37–1.52 s, while the two heaviest have spans around 0.29–0.31 s but
longer compute-completion delays. This is consistent with dependency delays
propagating from heavily loaded parts of the domain.

Only **256/4096 (6.25%)** last forces cross physical nodes in noLB, and
**2/4096 (0.049%)** in sync. Thus most final force edges are local; that does
not exclude cross-node delays earlier in their dependency chains.
Each cell receives from 1–2 processes in noLB (mean 1.50), and 1–5 in sync
(mean 1.88). The 27 contributors do not automatically span all eight processes.

## Runtime and limitations

| metric | noLB | sync |
| --- | ---: | ---: |
| reported mean steps 42–100, ms | 1578.0 | 1558.4 |
| total application time, s | 150.079 | 151.056 |

This reproduces the weak LB benefit; the small timing difference is not a
speedup claim from a single instrumented run. Logging is limited to four steps,
but can perturb those steps. NoLB agrees closely with the preceding 1581 ms
result. Mean cell force-join improvements are not equivalent to application
speedup: cells overlap, other phases remain, and the printed step timer belongs
to cell (0,0,0).

## Next action supported by this experiment

Audit **why Diffusion stops with measured GPU maximum/mean near 1.40**. Log
reasons for rejecting candidate moves, the selected transport cost tier,
predicted source/destination loads, fixed versus movable GPU work, and capacity
constraints. Compare these predictions with the next interval's per-GPU audit.
Do not add another communication threshold before inspecting the existing one.

The join trace alone cannot separate pure kernel execution from callback and
stream queueing. A device-event/activity timeline would be needed if that
distinction becomes the remaining question. No placement-independent ceiling
has been demonstrated.

## Reproduction and artifacts

`examples/charm++/cuda/gpudirect/leanmd/join-test/` contains:

- `run.sbatch`, isolated `scale.sh`, `leanmd-trace`, and `binary.sha256`;
- the instrumented source snapshot in `source/`;
- `job-22131761.out/.err` and separate noLB/sync rank logs;
- `summarize.py`, `summary.json`, and `gpu-audit.json`;
- `README.md` defining timer boundaries and limitations.

The instrumentation is opt-in through `LEANMD_JOIN_TRACE=1`. It adds no new
application barriers, CUDA synchronizations, or runtime messages. The force
message carries source PE and a local elapsed duration; no subtraction of
timestamps from different processes is used. Compute trace state is serialized
through migration. Cells are pinned by the existing application.

Built the application successfully and passed `git diff --check`. Existing
runtime changes and earlier scaling logs were preserved. Changes are uncommitted.
