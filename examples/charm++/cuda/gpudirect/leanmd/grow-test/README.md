# Matched two-node grow-any experiment

Job 22132598 runs baseline sync followed by sync with
`CHARM_DIFFUSION_GROW_ANY=1` on the same two A40 nodes. Both use the immutable
`../join-test/leanmd-trace` binary, four sampled force-trace steps,
`CHARM_GPU_LOAD_AUDIT=1`, `+LBDebug 2`, and the same calibrated cost table.
Only the grow-any environment switch differs between the two arms.

The switch disables the local connected-piece candidate filter after the
first accepted move. It does not remove the cost score, per-neighbor quota,
receiver checks, or memory admission checks. It is an experimental ablation,
not an installed production policy.

`empty-frontier.C` separately exercises the actual MetricComm and selection
loop with the existing lbsim runtime stubs. It makes no GPU/runtime startup
calls. Three objects with unit device load and amortized migration cost 0.1,
with no local communication edges, reproduce the selection stop:

```
grow=0 accepted=1 priced_out=0 unshed=2.0
grow=1 accepted=3 priced_out=0 unshed=0.0
```

This synthetic case proves the empty-frontier mechanism, not the performance
impact of altering it. `summarize.py` extracts real application outcomes,
per-GPU load, migration counts, and selection diagnostics for that comparison.
