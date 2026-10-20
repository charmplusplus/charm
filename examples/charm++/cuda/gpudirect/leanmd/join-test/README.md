# Two-node force-join diagnostic

Opt in with `LEANMD_JOIN_TRACE=1`. Snapshots at steps 45, 65, 85, 95,
across all cells and computes. No new barriers, CUDA synchronizations, or
additional runtime messages. Force messages carry source PE and locally
measured input-ready-to-completion-callback duration.

- `join-input input_ms`: first position receive-post to both inputs consumed
  by the compute's SDAG. Includes device copy queueing, the other input, and
  scheduling. Not a pure transport measurement.
- `join-compute ready_ms`: input-ready/launch entry to forcesReady callback
  consumed and sendForces entered. Includes launch, GPU queueing/execution,
  and callback scheduling. Not pure kernel time.
- `join-force post_ms`: receiver's force receive-post to SDAG consumption.
- `join-force elapsed_ms`: cell position-send start to this force consumption.
- `join-force span_ms`: first force consumption to this force consumption.
  `order=26` is the last of 27, including the self compute.
- `join-ack ack_ms`: force-send start to all sender acknowledgments consumed.

All durations use the same process clock. Cross-process timestamp differences
are deliberately absent. Compute migration across PEs invalidates its local
interval (reported -1). Cells are pinned in this application. Compute trace
state is serialized with migration state. Source physical node is source/32
for this two-node, four-process/node, eight-PE/process launch.

Sparse logging can perturb the four sampled steps. Do not use these runs as
precise performance comparisons to uninstrumented runs; check neighboring
unsampled steps as well. Post-to-consumption includes SDAG waiting, so a large
value alone is not evidence of a slow copy.

`run.sbatch` runs noLB then sync with all eight GPUs and NUMA pinning. The
launcher and binary are isolated here, and previous scaling rank logs remain
untouched. `summarize.py` reads these logs and emits JSON statistics.

The sync arm also enables `CHARM_GPU_LOAD_AUDIT=1`. Its busy-time union and
attribution totals should be inspected per process, with matching drain
windows; do not add host/launch/device fractions as a wall-time decomposition.
The source snapshot was built against the current libraries; no runtime
library changes were made for this experiment.
