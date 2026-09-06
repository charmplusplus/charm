# ref — PyTorch expert-parallel reference for the moe example

`moe_ref.py` is the same MoE layer as `../moe.C`, written the way an MoE
training framework writes it, so the Charm++ example can be compared against
current practice on the same node with the same work:

- one process per GPU, `torch.distributed` over NCCL;
- dispatch and combine are one `all_to_all_single` each, every token bundled
  per destination rank (the example sends one message per expert);
- expert compute with cuBLAS through `torch.mm`, the example's chunked SGD in
  the example's order (per source, chunks of `-c` tokens, `dh` reads the
  pre-update `W2`), on `-n` CUDA streams per rank;
- the host runs a step ahead of the device: slabs are double-buffered and the
  next step's token counts are exchanged (one `all_gather`) before this
  step's combine is issued, so routing and launch overhead hide under the
  previous step;
- load balancing by expert re-placement, the way EPLB does it, in two flavours:
  `--lb sync` recomputes the placement at an LB step from that step's token
  counts and moves the weights with NCCL point-to-point sends before the step
  runs (a full stop); `--lb async` makes the same decision but moves the
  weights after the old owner's compute of that step, overlapped with the rest
  of the step, and the placement takes effect at the next step. `--place
  greedy` moves as few experts as needed (pairwise transfers from the most to
  the least loaded rank, the same objective as DiffusionLB); `--place lpt` is
  EPLB's packing: every expert re-placed largest-first with an equal count per
  rank, with rank labels permuted afterwards to keep as many experts in place
  as possible.

The routing hash, the init hash, the loss and the update are ports of the
example's code; `selftest.py` checks the routing, the hashing and the init
values bit for bit against `hashcheck` (a copy of the example's functions):

    g++ -O1 -ffp-contract=off -o hashcheck hashcheck.C
    python selftest.py ./hashcheck

The Step lines use the example's format, and `out2` / `w2` are the example's
checksums (sums of squares of the combined outputs and of every expert's
weights after the step's update), so the two implementations can be checked
against each other: with the same arguments they agree to the fp32
accumulation-order noise of cuBLAS (about 1e-6 relative), and the
`tokens/PE` and `expert` imbalance ratios are identical.

## Options

The example's letters where they exist (`-e -m -t -k -i -u -z -p -c -n -L -C
-f -b -S -A -T -P`; `-H` for d_ff since argparse owns `-h`), plus

    --lb none|sync|async     re-placement mode (default none)
    --place greedy|lpt       placement algorithm (default greedy)
    --dtype fp32|bf16        bf16 weights and activations, throughput only
    --fuse                   one chunked pass per expert over all sources
    --no-pipeline            synchronize the host at every step end

## Run

    ref/run_ref.sh <jobid> <tag> [arguments]      # one A40 node, four ranks, logs in ranklogs/ref_<tag>/
    ref/compare.sh <jobid> [reps]                 # app noLB/sync/async vs ref none/sync/async (greedy, lpt)

`refwrap.sh` pins rank r to GPU r and the cores and memory of NUMA node 3-r
(the A40 nodes' GPU order is the reverse of the NUMA order) and sets
`NCCL_P2P_LEVEL=SYS` so NCCL uses PCIe peer-to-peer between the sockets
instead of bouncing through host memory. PyTorch 2.12 from
`/sw/rh9.4/user/python/conda-env/pytorch-2.12.1-cu130`, no module needed.

## Measured against the example (2026-09-06, one A40 node, four GPUs)

Same node, same allocation, same arguments; the example runs on `+gpupool`
with DiffusionLB, the reference on NCCL with greedy re-placement. Step times
are the mean over the timed steps.

Defaults: 64 experts of 2048 x 8192, 8192 tokens per rank per step, top-1,
fp32, 30 steps, LB every 5 steps from step 5.

| implementation                  | no LB   | sync LB | async LB |
|---------------------------------|--------:|--------:|---------:|
| Charm++ example, per source (-U) | 204-206 |     215 |  195-206 |
| Charm++ example, fused (default) |     191 |     199 |      185 |
| reference, per-source chunks    |     219 |     190 |      193 |
| reference, one pass per expert  | 189-199 | 153-155 |      159 |

Llama-sized FFN (4096 x 14336), 20 steps:

| implementation                   | no LB | sync LB | async LB |
|----------------------------------|------:|--------:|---------:|
| Charm++ example, per source (-U) |   631 |     623 |      605 |
| Charm++ example, fused (default) |   560 |     545 |      530 |
| reference, one pass per expert   |   568 |     448 |      498 |

Correctness: with identical arguments the two agree to fp32 accumulation
noise. At step 30 of the defaults the example gives `out2
2.625314025759872e+06`, the reference `2.625314019798698e+06` (2e-9 relative);
`w2` agrees to 7e-12. The `tokens/PE` and `expert max/avg` imbalance ratios
are identical step for step, which is the check that the routing really is the
same function.

### Where the difference is

**Not the migration mechanism.** The example's async LB beats its own sync LB
(605 against 623 at Llama size, 195 against 215 at the defaults); the
reference's async loses to its own sync (498 against 448, 159 against 155).
Overlapping the weight transfer with NCCL does not pay, for two reasons: the
placement only takes effect at the next step, so one more step runs
unbalanced, and the point-to-point transfer shares the PCIe links with the
next dispatch. The Charm++ runtime's overlap is the better mechanism.

**The placement decision.** Between reshuffles the reference's greedy
re-placement holds `tokens/PE max/avg` at 1.00 to 1.02; DiffusionLB holds 1.27
to 1.29 at the defaults and 1.52 at Llama size, moving 6 to 8 experts where
greedy moves 22. That is most of the reference's advantage, and it is a
balancer-quality gap, not a runtime gap.

**The GEMM structure.** Summing every rank's compute time gives the GPU
seconds spent in GEMMs for a known number of FLOPs (10 x d_model x d_ff per
token, forward and backward):

| variant                        | GPU ms/step, all ranks | TFLOPS/GPU | of peak |
|--------------------------------|-----------------------:|-----------:|--------:|
| reference, one pass per expert |                    278 |       19.8 |     53% |
| reference, per-source chunks   |                    397 |       13.8 |     37% |
| reference, TF32                |                    148 |       37.2 |     50% |
| reference, bf16                |                     72 |       76.5 |     51% |

The example chunked per source, as the reference does with `--fuse` off, and
its busiest rank took 153 ms against the reference's 150: the same 37%. Fusing
was worth 30% of the GEMM time and needed only a copy into contiguous scratch,
so the example now does it by default and its compute span is 131 ms. That
closes the no-LB gap: 191 ms against the fused reference's 189-199, and 560
against 568 at Llama size. A real framework goes further with a grouped GEMM
over all local experts.

**Precision.** Reference without LB: fp32 199 ms, TF32 120 ms, bf16 59 ms per
step. The example at TF32 takes 148 ms against the reference's 120.

**Exchange.** The reference moves 189 MB in and out per rank in an 18 ms
dispatch, 10.5 GB/s. The example's token exchange moves 191 MB in 33 ms, 5.8
GB/s, because it sends one message per expert (64 per dispatcher) at about
0.35 ms of host cost each, where the reference sends one per destination rank.
Both are far from the 24 GB/s the links can carry.

Two caveats on the tables. The example's `sync` and `async` include the LB
step's own cost, and its first LB step is expensive; the reference's greedy
decision is a few milliseconds of NumPy on data it already has. And the
reference keeps the host a step ahead of the device, which the example does
not need to do because its scheduler already overlaps; `--no-pipeline` costs
the reference nothing here (190 against 189 ms), so this is not what separates
them.
