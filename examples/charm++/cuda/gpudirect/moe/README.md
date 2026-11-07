# moe — mixture-of-experts layer miniApp for GPU load balancing

A single MoE feed-forward layer, written so that the experts are migratable
Charm++ objects with their state on the GPU, and the load moves the way it does
in a real MoE: by routing. Every step each PE routes its batch of tokens across
all experts, the experts run their feed-forward on what they were sent, the
outputs go back to the tokens' owners, and the experts update their weights.
The popularity of the experts is skewed (Zipf) and the hot set is reshuffled
periodically, so which GPU is overloaded changes during the run and no static
placement can fix it.

Unlike the halo-exchange miniApps (jacobi2d, leanmd, pic2d, barnes) this
exercises

- all-to-all traffic every step (every dispatcher to every expert and back),
- objects with no position key, so DiffusionLB takes its heap path rather
  than the interval cut,
- migration payloads of 100 MB to 1.6 GB per object,
- application-declared loads instead of CUPTI measurement: an expert knows its
  cost before it runs (token count times its measured seconds per token), so
  the balancer needs no kernel tracing at all.

## Model

Expert e holds `W1 [d_model x d_ff]` and `W2 [d_ff x d_model]` in fp32 (plus
Adam moments with `-A`, tripling the payload). For a run of n tokens it runs

    h  = relu(x W1)        y  = h W2                      forward
    dy = y                                                loss = 1/2 |y|^2
    dh = dy W2^T           dz = dh * relu'(z)             backward
    dW2 += h^T dy          dW1 += x^T dz                  gradient

five GEMMs per chunk of `-c` tokens (cuBLAS, fp32; `-T` for TF32), and once
per step the update `W -= (lr/n) dW` (SGD) or Adam on `dW/n`, so the cost of
an expert is linear in the tokens it receives with a 3:1 backward to forward
ratio like real training, plus one pass over the weights per step. The
weights are read only within a step: the gradient is accumulated in two
parts, one per dispatcher partition (the even dispatchers' tokens and the
odd dispatchers'), each its own run with its own chunk grid, and the two
parts are added once in a fixed order. That is the structure two copies of
an expert compute in (below), so the bits agree whether or not an expert
was replicated. The loss is self-similar so no labels are needed; it only
exists to make the weights change every step in a way that depends on
exactly which tokens each expert saw and in which order, which is what the
checksums verify.

## Decomposition and communication

- `Dispatcher` group, one per PE, never migrates. Holds `-t` tokens of
  `d_model` floats on the device (fixed for the run), routes them each step on
  the host with a hashed Zipf draw, sorts them by expert with a gather kernel,
  and ships one slab per expert with a zerocopy device send (`CkDeviceBuffer`
  and a post entry method on the expert). Expert outputs land in a receive
  slab by expert offset and a combine kernel scatters them back to token
  order (weighted `1/k` for top-k). It also owns the PE's GPU context: the one
  stream every kernel and transfer on the PE goes through, the cuBLAS handle
  with a fixed workspace, and the activation scratch.
- `Expert` 2-D chare array `(expert, copy)`, `usesAtSync`. Created as
  `(e, 0)` for every expert; `(e, 1)` exists while the balancer keeps a
  second copy (below). A copy receives one slab per dispatcher it serves per
  step into a per-source segment (arrival order does not matter: the post
  entry method knows the source), runs the chunked step partition by
  partition in source order, sends each dispatcher its outputs from the
  matching output segment, and waits for those sends to complete before the
  next step.

Routing is a pure function of (seed, step, PE, token, slot). Slab order is
(token, slot) order within a source, sources are processed in PE order, chunk
boundaries fall at fixed offsets, GEMMs use a fixed workspace with atomics
disallowed, and the checksums are fixed-order double reductions. So every run
with the same arguments produces the same `out2` (sum of squares of the
combined outputs) and `w2` (sum of squares of all weights) whatever the
placement, and a migration that loses or reorders anything shows up as a
different checksum.

## Load balancing

The balancer is ShedLB, linked in (`-balancer ShedLB` in the Makefile; a
`+balancer` on the command line still overrides it). Every expert holds the
same bytes but the loads differ by routing (the hottest draws about 13x the
average at z 1.0), migration cost is bytes, and the communication graph
carries no placement information (each expert receives equal shares from
every dispatcher), so the objective that matters is ShedLB's: clear each
overloaded GPU's excess for the fewest bytes, and leave a drift-sized
imbalance (`+LBShedEps`, 0.10) alone instead of chasing it with a 128 MB
move. The runtime's default chain, MetisLB then DiffusionLB, repartitions
(46 of 64 experts proposed at one step on two nodes) or under-moves
(tokens/PE 1.28 against 1.01 for a greedy re-placement).

### Replication

A placement balancer stops at the heaviest object: at z 1.0 the hottest
expert draws ~21% of the tokens, so on 8 GPUs no placement gets below an
imbalance of 1.69 (mean share 12.5%). The experts therefore declare
themselves replicable (`setLBReplica`: up to four copies, the tree's limit,
with what a copy sends its peers and, once it has, what that measured), and
ShedLB asks for a copy of an expert whose load exceeds the per-GPU
threshold (`ckClone`), and takes it back once the expert is light again
(`ckRetire`). The runtime decides when, which expert and where, from the
measured loads, the memory ledger and its cost model; the application
implements the copy:

- The dispatchers are in four fixed partitions, `p = d mod 4`; copy r of k
  serves the contiguous block of partitions `[4r/k, 4(r+1)/k)`, a single
  partition or an aligned pair. The step's gradient is always the fixed
  tree `((P0 + P1) + (P2 + P3))` over the partitions' accumulators: a copy
  computes its block's leaves against the same weights, folds them to its
  block sum, sends it to every other copy (zero-copy device messages,
  landing in per-peer parity double-buffers so the next step's can land
  while this step's are read), assembles the two tree nodes from the block
  sums and applies the same update, so the copies' weights stay identical
  and equal an unreplicated expert's, which computes the four leaves itself
  and folds the same way. Per step that is k - 1 weight-sized exchanges per
  copy (128 MB each at the defaults) and `4 + 4 (R - 1)` weight-sized
  buffers per copy (`replicaExtraBytes`, declared to the ledger).
- The balancer's decision for a round reaches every copy of the expert
  before the round releases any of them: `ckClone` on the copy that will
  send the state, `ckRetire` on the copy that goes (always the highest
  rank), `ckCloneNotice` / `ckRetireNotice` on the others. All act at the
  end of the step the round released them in: the step's outputs announce
  the number of copies serving the next step (lookahead is off in the
  window, so every dispatcher routes the next step with it), the source
  inserts `(e, k)` where the balancer said and sends it the weights (and
  moments) behind the update, the copies whose block grew open channels for
  the dispatchers they newly serve, and a copy left alone frees its exchange
  buffers. A new copy waits for its state, opens channels for its
  dispatchers and starts at the next step. A retiring copy's dispatchers
  drop their channels at the start of the next step and say so, then it
  closes its own and goes.
- Only copy 0 reports to the main chare (the others' tokens, PEs and device
  times ride in with their block sums), so the per-step report count never
  changes; `w2` is copy 0's weights, identical to every copy's.

`-X` starts the hottest expert of the first hot set as two copies (both
initialised from the same seed, no transfer) with no balancer involved: the
replicated arithmetic alone, whose checksums must equal the plain run's bit
for bit. With `-f 99999` there is no balancing step, so no copy is ever made;
how many copies to keep is the balancer's decision, priced from the
calibrated transfer-cost table and the copies' own measurements (there is no
option for it since 2026-11-05; `-R` is gone).

The join point is inside the expert's step, after it has consumed every slab
of the step and before it computes (see `runStep` in `moe.ci`). Nothing can
be sent to an expert between those two events (the dispatchers are waiting
for its outputs), so a parked or migrating expert never has a device receive
landed or in flight; the runtime holds a migration until every such receive is
consumed, and an expert parked in `AtSync` with an unconsumed slab would
deadlock. `-a` uses the split barrier (`AtSyncStart` at an LB step,
`AtSyncWait` `-l` steps later) and needs `+LBAsync`.

The migratable state is the weights (and moments) plus whatever the step has
live at the moment of the move: the slabs received so far before compute, the
outputs after it. Each expert declares its load with `setObjGPUTime` every
step: its token count times an EMA of measured seconds per token (device
events around its own kernels), which is also rate-aware for free. With `-I`
the app instead switches CUPTI instrumentation on for three steps before each
LB step, as pic2d does, and the balancer uses measured loads.

## Options

    -e experts (64)        -m d_model (2048)      -h d_ff (8192)
    -t tokens per PE per step (8192)              -k top-k (1)
    -i steps (50)          -u warmup steps (5)
    -z zipf exponent (1.0; 0 = uniform)           -p hot-set reshuffle period (10; 0 = static)
    -c chunk tokens (1024) -r slab headroom (0 = auto from the zipf peak)
    -L learning rate (1e-4) -A Adam               -T TF32
    -B bf16 GEMM operands (tokens, activations, outputs, a bf16 copy of each
       weight), fp32 accumulation, fp32 master weights, moments, gradient
       accumulators and checksums; half the exchange bytes
    -G serving: the forward alone, no gradient, no update, no exchange between
       an expert's copies (w2 stays its initial value; out2 is the oracle)
    -C checksum frequency (5; 0 off)              -s stats frequency (1)
    -f first LB step (10)  -b LB frequency (9999) -a async LB  -l wait lag (3)
    -X start with the hottest expert as two copies (the arithmetic without a balancer)
    -I CUPTI-measured loads instead of estimates  -S seed  -P print placement
    -U one chunked pass per source instead of one per expert (slower; the
       fused default is described under Measured below)
    -E routing trace to replay instead of the hashed Zipf (ref/trace_routing.py
       records a real checkpoint's top-k per token on WikiText-103; token t of
       dispatcher d at step s is text token ((s-1)*n_disp + d)*n_tokens + t,
       modulo the trace; the slab capacity and -X's expert come from the trace)
    -w a transformer block: an attention sublayer (q/k/v/o projections, 32
       heads, causal, sequences of 1024 tokens) on each dispatcher's tokens
       before the MoE, trained on its own loss 1/2|h|^2 with SGD; the MoE routes
       its output. The same sublayer is the harnesses' --attn (ref/moe_ref.py).

## Build and run

    make                                   # needs cuBLAS: MATHLIBS=<nvhpc>/math_libs/<ver>
                                           # a HIP tree picks hipcc and hipBLAS itself (HIP_PORT.md)
                                           # device code for sm_80 and sm_86 (Delta's A100 and A40
                                           # nodes); CUDA_ARCHS="90" on the make line for another

On Delta (one A40 node, four PEs, one per GPU, NUMA-pinned by the wrapper,
which reads each GPU's NUMA node from sysfs through `ranklogs/gpunuma.sh`):

    export RANKLOG_DIR=ranklogs/x; mkdir -p $RANKLOG_DIR
    srun --jobid=$J --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=8 \
         --cpu-bind=none --exact --unbuffered ranklogs/moewrap.sh ./moe \
         -i 60 -u 3 -f 20 -b 20 -a \
         +gpushm +gpupool +gpupoolsize 1024 +gpuipceventpool 256

No balancer flags: ShedLB is linked in, and `-a` declares the split barrier
to the runtime (`CkLBSetAsync`), as sph3d, pic2d and leanmd do.

`ranklogs/lbcmp.sh <jobid> <reps> noLB_pool sync_pool async_pool` runs the
comparison and checks every run's checksums against the first.

Per-step output: `tokens/PE max/avg` is the routed-token imbalance across
PEs, `gpu/PE max/avg` the measured device-time imbalance (lagged one step:
an expert's time is sampled when its next step starts), `expert max/avg` the
skew across experts. With the defaults the hottest expert draws about 13x the
average.

## The load declaration (2026-10-31)

An expert declares its load as tokens times a rate times the steps of the
interval. The rate is job-wide: each PE keeps the union of its experts' kernel
spans on the device clock -- one interval set across steps, so a window's busy
time can never exceed its wall time -- and the tokens they covered, both with
a per-step decay; copy 0 of every expert reports its PE's two figures with its
step report, and the main chare broadcasts their ratio over all PEs to every
PE each step (`MoeCtx::global_spt`). Every expert does the same work per
token on identical GPUs, so this is the model, and every GPU's declared load
is its tokens on one scale.

What it replaced: a per-PE rate, busy time per step divided by that PE's
tokens. Experts' steps overlap in time under lookahead, so per-step interval
sets counted the same device time once per step, and a crowded GPU, whose
kernels run slower under contention, read a higher rate as well as more
tokens. In the async replication run below (job 23383169) two GPUs read
2.3-2.6e-5 s/token against 1.2-1.3e-5 on the other two with tokens within
15%, the balancer moved most of their experts away, and the run stayed at a
1.36 token imbalance; a round later the reading had flipped. The flat-load
diagnostic (`CHARM_MOE_FLATLOAD=1`) declares a constant rate and remains the
check of the model.

## Measured 2026-10-31: replication, one A40 node, four PEs

`-z 1.3 -p 0 -i 60 -u 3` (the hottest expert draws ~34% of the tokens, above
a GPU's share of 25% and the 27.5% threshold), 64 experts of 2048 x 8192,
8192 tokens per process, `+gpupoolsize 32768` (below), the job-wide rate.
Steady state = steps 33-63. Every arm's `out2` and `w2` are the same bits:
`out2 2.534902339438027e+06, w2 2.183175448111042e+05`.

| arm                                   | ms/step | what the balancer did                              |
|---------------------------------------|--------:|----------------------------------------------------|
| plain (`-R 1 -f 99999`)               |   179.7 | nothing                                            |
| `-X` (hottest expert as two copies)   |   143.1 | nothing; the replicated arithmetic alone           |
| ShedLB `-R 1`, sync / async           | 182.2 / 181.1 | nothing placeable: the floor (earlier run)     |
| ShedLB `-R 2`, sync                   |   122.1 | one copy decided at step 20, serving from 21; 7 moves |
| ShedLB `-R 2`, async                  |   115.3 | one copy decided at step 21, made at 22, serving from 23; 7 moves |

With a static hot set, replication under ShedLB takes a third off the step,
and the balancer's choice of destination beats `-X`'s fixed one. Both arms
end at tokens/PE 1.11 and a measured device imbalance of 1.27. With `-p 10`
(the hot set moves every 10 steps, balancing every 20) `-R 2` sync made 3
copies, retired 2 and moved 23 experts, checksums equal to its plain run, and
lost: 173.6 against 136.0 ms/step. Every decision there balances a hot set
that is gone the next step, as before, and a copy costs its exchange every
step until it is retired a round later.

Against the PyTorch and NCCL reference (`ref/`, same node class, same
arguments, `--fuse`, 4 lanes, fp32; `--replicate` is EPLB-style redundancy:
a second copy of the hottest expert, odd ranks' tokens to it, gradients
all-reduced between the copies; greedy re-placement for the rest):

| implementation                            | ms/step |
|-------------------------------------------|--------:|
| reference, no balancing                   |   289.2 |
| reference, greedy re-placement            |   188.8 |
| reference, greedy + redundancy, sync      |   171.1 |
| reference, greedy + redundancy, async     |   170.9 |
| this app, no balancing                    |   179.7 |
| this app, ShedLB with copies, sync        |   122.1 |
| this app, ShedLB with copies, async       |   115.3 |

The two agree on `out2`/`w2` to 4e-10 (same computation, different GEMM
library). The app's lead comes from the overlap work of 2026-10-25
(lookahead, token channels, streamed outputs), which the reference does not
have, and holds with replication on both sides.

Against Megatron-Core 0.20 (`ref/megatron_moe.py`: Megatron's own MoE layer
in expert parallelism with its all-to-all token dispatcher, the example's
hashed routing substituted for its top-k so the token skew per step is the
same, static expert placement as Megatron trains, SGD; same arguments, one
A40 node, `ref/run_mg.sh`). Megatron initializes its own weights, so only
throughput compares; steady state is steps 33-63 as above.

| Megatron-Core arm                                 | ms/step |
|---------------------------------------------------|--------:|
| fp32, one GEMM per expert, z 1.3                  |   294.1 |
| fp32, grouped GEMM, z 1.3                         |   294.1 |
| bf16, grouped GEMM, z 1.3                         |    95.5 |
| fp32, one GEMM per expert, uniform routing (z 0)  |   161.2 |
| bf16, grouped GEMM, uniform routing (z 0)         |    59.9 |

At the same precision Megatron runs at the reference's unbalanced number (294
against 289): the same strict sequence of dispatch all-to-all, the slowest
GPU's experts, combine all-to-all, and the backward of each, with the hottest
expert's GPU setting the pace (forward 248 ms on the slowest rank, 144 on the
fastest). The uniform-routing arm is the floor that perfect balancing inside
that structure could reach with the same token count: 161 ms/step, 1.4x the
app's 115 with the skew still in the data. Grouped GEMM buys nothing in fp32
on an A40, where one expert's GEMM already fills the GPU; bf16 tensor cores
buy 3x, which this fp32 example does not use. The same-node control on the
Megatron node ran plain 179.6 and ShedLB `-R 2` async 109.0 ms/step (one copy,
9 moves) with the checksums above.

Three things the runs showed:

- **The memory ledger needs a pre-sized pool.** With 1024 MB arenas mapped on
  demand, and even with one 16 GB arena, ShedLB refused every copy ("would
  need 2 copies, but no bin can hold one"): the buddy pool rounds each block
  to a power of two, so an expert's 521 MB of state occupies 896 MB, 16
  experts take 14.3 GB, and the plannable headroom is gone. A 32 GB arena per
  PE works on one node. Moves were refused for the same reason before.
- **The per-PE rate misled the balancer** (see "The load declaration" above).
  With it, the async arm ran 140.1 ms/step: at its second round two GPUs
  read twice the others' seconds per token with tokens within 15%, most of
  their experts were moved away, the run stayed at a 1.36 token imbalance,
  and at step 60 the copy of the still-hottest expert was retired on a load
  that read as cooled. With the job-wide rate the same arm runs 115.3 and
  every GPU declares on one scale.
- **The update scheme costs.** At the defaults (z 1.0, `-p 10`, no
  balancing) the previous binary, with its in-place per-chunk SGD update,
  ran 98.8 ms/step; the accumulate-then-apply scheme that bit-identity across
  copies requires ran 117.6, 19% more: one pass over every expert's weights
  per step (16 experts x 512 MiB of traffic per GPU, ~12 ms) and a second
  chunk grid for the second partition of every expert. The cost falls with
  model size and with fewer experts per GPU (estimated ~4% at the Llama size
  on two nodes); it was measured here, not there.

## Measured 2026-10-31, later: bf16, serving, real-model shapes, Megatron-Core

Two modes were added for this comparison. `-B` runs the GEMMs on bf16
operands (the tokens, activations and outputs, and a bf16 copy of each
weight) with fp32 accumulation, fp32 master weights, moments and gradient
accumulators: the mixed precision a framework trains in. The exchange bytes
halve; the checksums differ from fp32's and are placement-independent as
before, so every arm below agrees with its plain arm bit for bit. `-G`
serves: the forward alone, no gradient, no update and no exchange between an
expert's copies, which then cost one state send each; `w2` stays its initial
value and `out2` is still the oracle. The bf16 values are raw 16-bit words
with the conversions written by hand (`moe.cu`), so the CUDA and HIP builds
share one path; the mixed-precision GEMM is `hapiBlasGemmEx` in
`../hapi_blas.h`. The Kokkos variant has the same kernels and the same
smoke checksums bit for bit. The reference's `--dtype bf16` became the same
mixed precision (its `out2` agrees with `-B`'s to 3e-7); the Megatron-Core
harness gained `--forward-only`. The one-node arms below ran without the
balancer's helper thread (`CHARM_LB_HELPER=near`, which takes the strategy
and the CUPTI load build off the PE); the runner sets it from the two-node
arms on. It changes the balancing steps' own cost, not the steady state
between rounds that the tables report.

Default shape (64 experts of 2048 x 8192, 8192 tokens per process, `-z 1.3
-p 0`, steady steps 33-63, one A40 node), ms/step:

| arm                                            | fp32 training | bf16 training | bf16 serving |
|------------------------------------------------|--------------:|--------------:|-------------:|
| this app, plain                                |         179.7 |          62.4 |         22.9 |
| this app, `-X`                                 |         143.1 |          64.0 |         18.6 |
| this app, ShedLB `-R 2`, sync                  |         122.1 |          58.9 |         21.3 |
| this app, ShedLB `-R 2`, async                 |         115.3 |          52.8 |         20.2 |
| Megatron-Core, grouped GEMM                    |         294.1 |          95.5 |         54.5 |
| Megatron-Core, uniform routing (its balanced floor) |    161.2 |          59.9 |              |
| reference, no balancing                        |         289.2 |         156.8 |              |
| reference, greedy + redundancy, async          |         170.9 |         131.2 |              |

(Megatron-Core serving in fp32: 150.1.) In bf16 the step is a third of
fp32's and the copy's gain shrinks from 36% to 15%: the exchange of the
copies' fp32 partials (two 64 MB messages each way per step) and the second
apply pass do not shrink with the GEMMs, and `-X` is now slower than plain.
Serving, a copy exchanges nothing, and the balancer's copy and `-X` both
beat plain.

Real-model layer shapes, with the routing skew set to what the trained
models show per expert, a hottest expert 1.4-1.6x the mean (the per-domain
routing plots of the Mixtral and OLMoE papers; the loss-free balancing
paper's MaxVio of 0.04 for its method against 0.52-0.72 for its
auxiliary-loss baselines), steady steps 23-43, ms/step:

| shape                                                                  | app fp32 | app bf16 | Megatron fp32 | Megatron bf16 |
|------------------------------------------------------------------------|---------:|---------:|--------------:|--------------:|
| Mixtral-8x7B: 8 experts of 4096 x 14336, top-2, 8192 tokens, z 0.3     |    507.4 |    126.6 |         845.5 |         236.4 |
| DeepSeek-V3: 256 routed experts of 7168 x 2048, top-8, 4096 tokens, z 0.1 | 621.2 |    295.1 |        1136.6 |         494.6 |

The app leads 1.7-1.9x at both shapes, and the balancer has nothing to do at
either: a 1.5x hottest expert averages away over the 2 or 64 experts each of
4 GPUs holds (per-GPU token ratios 1.10 and 1.02, under the 10% threshold),
and the ShedLB arms cost their balancing steps and nothing else (519.0 /
130.1 and 672.1 / 311.0). Imbalance a balancer can act on comes from fewer
experts per GPU (Mixtral on 8 GPUs holds one each, and its hottest expert's
18% exceeds a GPU's 12.5% share; not run, two nodes) or from batch-level
skew: an auxiliary loss balances the load over the data mix, not within a
batch, so a batch of one language, one repository or mathematics alone
routes 2-3x the mean to that domain's specialists, and serving sees the
same, which is why DeepSeek-V3's deployment duplicates hot experts (EPLB).

The Mixtral-8x7B shape with such a router (`-z 1.0`: the hottest expert
takes 37% of the tokens, the per-GPU token ratio is 1.47; ShedLB decides one
copy at step 20, serving from step 23), ms/step:

| arm                                       | training fp32 | training bf16 | serving fp32 | serving bf16 |
|-------------------------------------------|--------------:|--------------:|-------------:|-------------:|
| this app, plain                           |         672.6 |         177.2 |        287.4 |        110.3 |
| this app, ShedLB `-R 2` async, one copy   |         602.9 |         197.1 |        219.6 |         85.8 |
| Megatron-Core, grouped GEMM               |        1108.6 |         302.1 |        536.4 |        163.7 |

Training in fp32 the copy takes 10% off the step; in bf16 it adds 11%: the
copies' partial exchange is 470 MB each way per step at this shape (4096 x
14336 in fp32) and a 177 ms step cannot hide it. Serving, the copy takes 24%
(fp32) and 22% (bf16) off, with nothing to exchange. The checksums are the
same bits with and without the copy in every arm.

### Two A40 nodes, eight GPUs (2026-10-31, job 23388275)

Same recipe as above (`-i 40 -u 3`, steady state steps 23-43, LB helper
thread on), one PE per GPU, `+gpupoolalloc vmm +gpupoolsize 16384
+gpupoolchunk 1536`. The two shapes are the Mixtral-8x7B expert (8 experts of
4096 x 14336, top-2, 8192 tokens per GPU, the trained skew `-z 0.3`: one
expert per GPU, the hottest 1.4x the mean) and the default shape at `-z 1.0`
(64 experts of 2048 x 8192, eight per GPU, the hottest expert 21% of the
tokens). "copies" is ShedLB (`-R 2 -f 20 -b 20 -a`); "plain" is `-R 1
-f 99999`. Megatron-Core is expert parallel over the 8 GPUs, grouped GEMM,
`--forward-only` for serving. Checksums (`out2`, `w2`) are bit-identical
between plain and copies in every pair.

| Mixtral-8x7B shape, trained skew       | plain | copies (what ShedLB did)        | Megatron-Core |
|----------------------------------------|------:|--------------------------------:|--------------:|
| fp32 training                          | 649.4 | 656.5 (2 copies)                |        1155.0 |
| bf16 training (`-B`)                   | 164.7 | 212.2 (2 copies)                |         337.7 |
| fp32 serving (`-G`)                    | 280.0 | 290.7 (2 copies)                |         570.0 |
| bf16 serving (`-B -G`)                 | 113.4 | 153.1 (2 copies)                |         201.3 |

| default shape, `-z 1.0`, fp32 training | plain | copies (what ShedLB did)        | Megatron-Core |
|----------------------------------------|------:|--------------------------------:|--------------:|
| 64 experts, 8 per GPU                  | 198.0 | 134.1 (1 copy + 6 moves)        |         385.0 |

The default shape is the case the design was for: with eight experts per
GPU the hottest expert (21% of the tokens, 1.69x a GPU's share) is split
over two GPUs and the rest re-placed around it, tokens per GPU max/mean
2.86 -> 1.69, the step -32%. The Mixtral shape on eight GPUs has one expert
per GPU, each 0.7-1.4x the mean: a copy of the hottest expert has to land
on a GPU that already holds a whole expert, which leaves that GPU as full
as the one the copy relieved, and the exchange is paid for nothing (the
balancer at the time took the emptiest bin when no bin cleared the target;
this run is why ShedLB now declines a copy that does not lower the fullest
bin it touches, `doc/internals/lb-multiplicity.md`; with that rule, same
nodes: bf16 training plain 166.2, ShedLB 170.9 with both copies declined,
165.5 against 165.5 over steps 33-43; the default shape keeps its copy and
its six moves, 197.4 -> 134.2). bf16 pays most: the
gradient partials are fp32 (470 MB per copy per step at this shape) and in
serving the copies only split work a single GPU did in 113 ms.

**The Mixtral shape with the unbalanced router (`-z 1.0`: hottest expert
37% of the tokens, lightest 5%), eight GPUs, one expert each** (job
23389495, same nodes and flags, `+gpupoolchunk 2046`). This is the case a
copy pays with one expert per GPU: the hottest expert is more than twice
the lightest, so half of it fits beside the lightest. ShedLB made the same
decision in every arm: expert 1 copied to PE 0, expert 4 copied to PE 5, one
move; tokens per GPU max/mean 2.53 -> 1.27. The fp32 slab block at this
skew is 2.4 GB with lookahead (capacity 9120 tokens per source), more than
any registrable chunk, so the fp32 arms ran with lookahead off (`-Y`, one
slab parity, both arms alike); the bf16 arms ran as usual, and once with
`-Y` for the comparison. Checksums identical plain vs ShedLB in every pair.
Steady state steps 23-43, and steps 33-43 in brackets (the copies land at
step 22).

| Mixtral-8x7B shape, unbalanced router, 8 GPUs | plain           | ShedLB (2 copies + 1 move) | change     | Megatron-Core EP8 |
|-----------------------------------------------|----------------:|---------------------------:|-----------:|------------------:|
| fp32 training, `-Y`                           | 1209.9 (1222.2) |              692.7 (687.2) | -43%       |            1882.1 |
| bf16 training                                 |   276.2 (276.2) |              219.1 (203.8) | -21% (-26%)|             522.6 |
| bf16 training, `-Y`                           |   294.4 (294.1) |              224.7 (217.6) | -24% (-26%)|                   |
| fp32 serving, `-Y`                            |   491.9 (495.9) |              282.0 (287.2) | -43%       |             908.8 |
| bf16 serving                                  |   121.8 (126.5) |               113.3 (98.9) | -7% (-22%) |             279.6 |

**FasterMoE on the same case** (job 23403736, gpub002+062, 2026-11-01, same
eight GPUs' worth of A40s on two nodes): FastMoE 1.1 as shipped ("plain", two
all-to-alls around the experts, static placement), with FasterMoE's smart
schedule, and with the schedule plus expert shadowing, which is the PPoPP'22
FasterMoE system: before every step its performance model picks the experts
that would be the critical path, broadcasts their weights to all eight GPUs
(each GPU then runs its own tokens for them) and reduces their gradients back
to the owner after the backward. Harness, launcher and the three patches the
shipped code needed are in `ref/README.md`; the policy's machine parameters
were measured on these ranks (470 MB expert broadcast to 8 ranks 117-154 ms,
GEMM 23.7 TFLOP/s fp32, 123 bf16). Steady state steps 23-43 (33-43 in
brackets); "shadows" is how many of the 8 experts FasterMoE shadowed every
timed step.

| Mixtral-8x7B shape, unbalanced router, 8 GPUs | FastMoE plain   | + smart schedule | FasterMoE (schedule + shadows) | example plain | example ShedLB | Megatron-Core |
|-----------------------------------------------|----------------:|-----------------:|-------------------------------:|--------------:|---------------:|--------------:|
| fp32 training                                 | 2102.5 (2106.8) |  2018.6 (2023.4) |   1195.0 (1195.8), 3 shadows   |        1209.9 |          692.7 |        1882.1 |
| bf16 training                                 |   631.2 (631.4) |    579.5 (579.8) |   402.9 (404.6), 2 shadows (a) |         276.2 |          219.1 |         522.6 |
| fp32 serving                                  |   901.9 (904.2) |          (b)     |   661.5, 3 shadows (b)         |         491.9 |          282.0 |         908.8 |
| bf16 serving                                  |   276.3 (276.3) |          (b)     |   306.7, 2 shadows (b)         |         121.8 |          113.3 |         279.6 |

| default shape, `-z 1.0`, fp32 training        | FastMoE plain   | + smart schedule | FasterMoE (schedule + shadows) | example plain | example ShedLB | Megatron-Core |
|-----------------------------------------------|----------------:|-----------------:|-------------------------------:|--------------:|---------------:|--------------:|
| 64 experts, 8 per GPU                         |   509.6 (507.9) |    449.1 (448.1) |   322.1 (322.9), 1.6 shadows   |         198.0 |          134.1 |         385.0 |

(a) FasterMoE's bf16 shadow path is numerically broken: the loss is inf or
NaN from step 2 in every bf16 shadow run while plain and schedule-only bf16
stay finite, the weights of every expert are still finite at the end (a
one-step probe, 2026-11-01), so the shadowed experts' outputs themselves are
wrong; the shipped code did not compile that path for bf16 at all (the
dispatch patch enabled it). The time is a throughput figure only. The fp32
shadow run reproduces FastMoE plain's loss to six digits at steps 1, 3 and
10, so the fp32 rows compare equal work. (b) FasterMoE's fused forward keeps
the experts' recorded sub-graphs until a backward consumes them, so a
forward-only run grows by the hot rank's activations every step: the
schedule-only serving arms ran out of 44 GB on the hot rank in every attempt
(under `no_grad`, with grad enabled, with a garbage collection per step),
and the shadow arms, whose hot rank holds a smaller share, reached step 30
with the collection but not step 43. The shadow serving rows are therefore
28-step runs (`-i 25 -u 3`), steady state steps 14-28, with 3 and 2 experts
shadowed every step (the fp32 forward sat at 650-720 ms throughout).

Shadowing takes FasterMoE from 43% slower than Megatron to the example's
plain arm (1195 against 1210 in fp32 training, 662 against 492 serving)
and no further: a shadow is eight
copies of a 470 MB expert re-broadcast and reduced every step, so the
model stops at three experts and the fp32 step keeps about 500 ms of that
traffic; ShedLB's second copy is one persistent replica with one 470 MB
partial exchange per step, 692.7. At the default shape (eight experts per
GPU) FasterMoE shadows one or two of 64 experts and lands at 322 against
the example's 198 plain and 134 with the copy and six moves. FlexMoE
(SIGMOD'23) has no public code; its paper reports 1.30x over FasterMoE on
average, which is not a measurement on this case.

What the invalid evening runs were: a VMM pool block bigger than a chunk
spans chunks, and the fabric registers one chunk at a time, so a remote
write into the part of a slab beyond its first chunk stopped at the chunk
end (`CHARM_MOE_SLABPROBE=1` showed exactly 9 MiB of each affected segment
landing, every one 9 MiB before a 256 MiB boundary). `+gpupoolchunk` above
the largest block fixes that, but not any value: a 2048 MB chunk registers
without error and then accepts no remote write at all (every rank, first
step, `ADDR_OUT_OF_RANGE`), while 1536, 1792 and 2046 MB carry every
transfer exactly (the probe: 0 of 64 slabs differ): the ceiling is 2^31
bytes. The runtime now refuses a VMM chunk of 2048 MB or more on more than
one node (`hapiDevPoolCheckRegistrable`).

## Measured 2026-11-01: real routing, two A40 nodes, eight GPUs

> **Superseded in part (2026-11-02):** the NCCL baselines in this section
> ran without the Slingshot plugin and the example's serving arms from
> "Copies up to four" on ran with a serving bug; the next section has the
> re-measured matrix and the corrected ratios. The example's training rows
> and the four-GPU rows stand, except (2026-11-03) the Megatron-Core rows,
> which ran sequential experts, and the FasterMoE serving rows, which paid
> a per-step garbage collection; see "Measured 2026-11-03".

Everything above routes a hashed Zipf. This section replays recorded routing
(`-E`, `--trace`; `ref/README.md` has the tools; the traces are release
`traces-2026-11-01` of <https://github.com/adityapb/charm-moe-traces>, 473 MB
packed, fetched by its `fetch_traces.sh`), bf16 throughout, lookahead
on, with the attention sublayer (`-w`, `--attn`) so a step is a transformer
block's, and every system on the same tokens. Three sources, from the mildest
to the strongest skew:

- **Mixtral-8x7B on WikiText-103, layer 12** (`traces/mixtral_L12.trace`, the
  checkpoint's own router): 8 experts, top-2; the hottest expert takes 15.9%
  of the (token, slot) pairs and the lightest 3.1%, so one GPU of eight holds
  1.3x the mean. Mixtral's inference routing is close to balanced (13-17% in
  every layer); this is its most unequal layer.
- **OLMoE-1B-7B on WikiText-103, layer 12** (`traces/olmoe_L12.trace`): 64
  experts, top-8; the hottest 5.7% (3.6x the mean), but 0.46 of a GPU's share
  with eight experts per GPU: placement balances it, no copy is needed.
- **SmartMoE's training routing, layer 2** (`traces/smartmoe_L02.trace`,
  thu-pacman/SmartMoE-AE: the tokens each of 32 experts received per
  micro-batch while a GPT-MoE trained without a balancing loss, top-1): the
  hottest expert takes 47.9% of the tokens, 3.8x a GPU's share with four
  experts per GPU, and the hot set drifts over the training run.

Arms (each in training and serving): the example plain (`-R 1`) and with
ShedLB (`-R 2 -f 20 -b 20 -a`); the PyTorch reference static, with EPLB's
packing recomputed every 10 steps (`--lb sync --place lpt`), and with one
copy of the hottest expert from the start (`--replicate`); Megatron-Core
grouped GEMM, dropless and with capacity factor 1.25; FastMoE as shipped and
FasterMoE (smart schedule and shadowing, the patched build). Steady state
steps 23-43 of 40 + 3 warmup; repeats where a band is given. Scale is 4 and
8 GPUs: the experiments are capped at two nodes.

*Serving rows of this section (every `-G` column below): measured with the
serving branch running two of the four dispatcher partitions, so half the
dispatchers' tokens were never served. Found on Frontier on 2026-11-02 and fixed
("Measured 2026-11-02: Frontier" below); these columns are to be re-measured.*

**Results, MoE layer alone** (job 23411075, gpub092+094, 2026-11-01, bf16,
8 GPUs, ms/step, steady state steps 23-43; "ShedLB" is the balancer as of
the morning, "ShedLB, critical path" the rule described below; checksums
identical between the example's arms in every pair):

| Mixtral-8x7B routing, layer 12 (bands: mean +- half-range of 3 runs on gpub001+002) | training | serving |
|-----------------------------------------------|---------:|--------:|
| example plain                                 | 162.0 +- 5.6 | 120.1 +- 1.6 |
| example ShedLB, morning rule (1 copy, 1 run)  |    185.9 |   119.8 (102.5 settled) |
| example ShedLB, critical-path rule (no copy)  | 164.8 +- 1.5 | 125.6 +- 9.5 |
| PyTorch reference, static                     |    334.5 |   197.6 |
| PyTorch reference, EPLB packing every 10 steps|    333.9 |   203.8 |
| PyTorch reference, one static copy            |    673.1 |   227.2 |
| Megatron-Core, grouped GEMM                   | 351.6 +- 16.9 | 182.3 +- 3.4 |
| Megatron-Core, capacity factor 1.25           |    337.3 |         |
| FastMoE as shipped                            |    386.3 |   193.2 |
| FasterMoE (schedule + shadows: none chosen)   | 376.3 +- 8.3 | 254.0 +- 2.2 |

| SmartMoE training routing, layer 2 (mean +- half-range of 3 runs) | training | serving |
|-----------------------------------------------|---------:|--------:|
| example plain                                 | 79.9 +- 0.1 | 33.4 +- 0.4 |
| example ShedLB, critical-path rule (1 copy)   | 50.9 +- 1.5 (47-49 settled) | 27.8 +- 1.2 (23-27 settled) |
| example ShedLB, morning rule (1 copy, 1 run)  |     51.6 |    25.8 |
| PyTorch reference, static                     | 178.4 +- 1.2 | 100.8 +- 1.5 |
| PyTorch reference, EPLB packing every 10 steps| 201.6 +- 2.2 | 127.9 +- 0.5 |
| PyTorch reference, one static copy            | 232.5 +- 1.2 | 73.3 +- 0.9 |
| Megatron-Core, grouped GEMM                   | 164.6 +- 0.5 | 95.4 +- 0.6 |
| Megatron-Core, capacity factor 1.25           | 165.1 +- 0.6 |         |
| FastMoE as shipped                            | 239.8 +- 1.8 | 106.6 +- 0.3 |
| FasterMoE (1.3 shadows per step, loss NaN from step 7-8) | 113.1 +- 6.8 | 130.7 +- 1.1 |

What the two routings say. On Mixtral's real routing every balancer is
idle or harmful: the hottest expert is 1.27x the mean with one expert per
GPU, EPLB's packing cannot change a one-per-GPU placement, a static copy in
PyTorch pays an fp32 partial all-reduce across the node link every step
(2x slower), FasterMoE's model chooses no shadow, and ShedLB's copy lowered
its own GPU but not the fullest one (+15% in training). The example's plain
arm is 2.2x faster than Megatron and 2.4x faster than FastMoE on the same
tokens, which is the runtime's token exchange and pipelining, not
balancing (bands of 3-5% on the example and FasterMoE, 5% on Megatron,
across two node pairs). On SmartMoE's training routing (the hottest expert 3.8x a GPU's
share) the balancer is the story: one copy and the critical-path rule take
the example from 80.1 to 52.2 in training and 33.4 to 28.0 in serving,
against FasterMoE's 113.1 (its shadows, bf16 numerics broken), Megatron
164.6, and the PyTorch reference's EPLB-style packing 201.6 and static copy
232.5; the example with ShedLB is 2.2x faster than FasterMoE and 3.2x faster
than Megatron in training, 3.4x faster than Megatron in serving, with bands
of 1-3% on every arm but FasterMoE's (6%). The copies
cap of two is what bounds it: the hottest expert's half is still 1.9x the
mean.

| OLMoE-1B-7B routing, layer 12 (critical-path rule) | training | serving |
|-----------------------------------------------|---------:|--------:|
| example plain                                 |    237.7 |   243.6 |
| example ShedLB (nothing to decide)            |    262.3 |   248.8 |
| PyTorch reference, static                     |    333.2 |   314.5 |
| PyTorch reference, EPLB packing every 10 steps|    308.5 |   284.0 |
| Megatron-Core, grouped GEMM                   |    455.6 |   280.2 |
| FastMoE as shipped                            |    697.4 |   332.5 |
| FasterMoE (2.8 shadows per step, loss NaN from step 3) | 641.1 | 403.0 |

OLMoE's routing (64 experts of 2048 x 1024, top-8, the hottest 0.46 of a
GPU's share) leaves the balancer nothing to do: per GPU the load is within
the tolerance, and the two balancing rounds in the window cost the example
the lookahead for a few steps each (+10% in training, +2% in serving) for no
decision. With eight slots per token the step is the token exchange: serving
costs what training does. EPLB's packing buys the PyTorch reference 7-10%
there; the example's plain arm is 1.3-1.4x faster than it and 1.2-1.9x
faster than Megatron.

**Results, transformer block** (the same arms with the attention sublayer in
every system, `-w` / `--attn`: q/k/v/o projections of 2048 x 2048, 32 heads,
causal, sequences of 1024, trained on its own loss; SmartMoE routing, job
23411075, critical-path rule):

| SmartMoE training routing, block              | training | serving |
|-----------------------------------------------|---------:|--------:|
| example plain                                 |     98.0 |    35.6 |
| example ShedLB (1 copy)                       |     63.8 |    30.0 (28.2 settled) |
| PyTorch reference, static                     |    185.8 |   103.2 |
| PyTorch reference, EPLB packing every 10 steps|    212.1 |   131.9 |
| PyTorch reference, one static copy            |    241.2 |    73.7 |
| Megatron-Core, grouped GEMM                   |    175.6 |    97.8 |
| Megatron-Core, capacity factor 1.25           |    174.7 |         |
| FastMoE as shipped                            |    249.6 |   109.9 |
| FasterMoE (schedule + shadows, bf16 numerics broken) | 119.2 | 132.5 |

| Mixtral-8x7B routing, block (4096 x 4096 projections) | training | serving |
|-----------------------------------------------|---------:|--------:|
| example plain                                 |    178.9 |   137.9 |
| example ShedLB, critical path (no copy)       |    188.4 |   134.2 |
| PyTorch reference, static                     |    358.3 |   221.6 |
| PyTorch reference, EPLB packing every 10 steps|    362.3 |   205.8 |
| PyTorch reference, one static copy            |    695.9 |   242.8 |
| Megatron-Core, grouped GEMM                   |    368.3 |   202.6 |
| Megatron-Core, capacity factor 1.25           |    369.2 |         |
| FastMoE as shipped                            |    421.0 |   201.4 |
| FasterMoE (schedule + shadows: none chosen)   |    417.9 |   267.4 |

The attention sublayer adds the same local work to every system (about 18
ms of training step to the example at both shapes), so the ratios shrink a
little and the ordering does not change: on SmartMoE's routing the example
with ShedLB is 1.9x faster than FasterMoE and 2.8x faster than Megatron in
training, 3.3x faster than Megatron in serving; on Mixtral's the plain
example is 2.1x faster than Megatron in training and the balancer, with
nothing to decide, costs its two rounds' lookahead pauses (+5%).

**Scale: four GPUs** (one node, the SmartMoE routing, MoE layer alone, eight
experts per GPU, critical-path rule; the 8-GPU numbers are the table above):

| SmartMoE training routing, 4 GPUs             | training | serving |
|-----------------------------------------------|---------:|--------:|
| example plain                                 |     55.4 |    32.9 |
| example ShedLB (1 copy)                       |     50.4 (44.1 settled) | 27.3 (17.2 settled) |
| PyTorch reference, static                     |    105.9 |    52.8 |
| PyTorch reference, EPLB packing every 10 steps|    108.6 |    68.8 |
| PyTorch reference, one static copy            |    102.7 |    42.2 |
| Megatron-Core, grouped GEMM                   |     90.2 |    52.1 |
| Megatron-Core, capacity factor 1.25           |     87.8 |         |
| FastMoE as shipped                            |    121.4 |    52.1 |
| FasterMoE (schedule + shadows, bf16 numerics broken) | 62.8 | 82.3 |

| Mixtral-8x7B routing, 4 GPUs (two experts each) | training | serving |
|-----------------------------------------------|---------:|--------:|
| example plain                                 |    137.8 |   108.7 |
| example ShedLB (no copy; moves)               |    137.3 |   100.7 |
| Megatron-Core, grouped GEMM                   |    270.8 |   138.1 |
| FasterMoE (no shadow chosen)                  |    318.0 |   188.2 |

From four to eight GPUs the example's plain step on the SmartMoE routing
goes 55.4 to 80.1 (the hottest expert is a larger multiple of a GPU's
share, and the exchange crosses the node link) and the balanced step 44.1
to 49.8 once the copy is in place; Megatron goes 90.2 to 165.0 and FasterMoE
62.8 to 107.0, so the gap widens with scale: 1.2x over FasterMoE at four
GPUs, 2.0x at eight. On Mixtral's routing the example is 2.0x faster than
Megatron at four GPUs and 2.2x at eight. Two points are what two nodes
allow.

**Baseline tuning** (job 23414031, SmartMoE routing unless said, MoE layer
alone): FasterMoE's group-wise exchange at group sizes 2 / 4 (default) / 8
trains at 104.7 / 113.1 / 107.2 and serves at 128.2 / 130.7 / 130.5, so its
best setting is 2.1x the example with ShedLB; Megatron's allgather token
dispatcher is slower than the all-to-all one used in the tables (234.2
against 164.6 training, 135.8 against 95.4 serving; on Mixtral 461.4 against
351.6), and its capacity factor 1.25 changes nothing at these skews (the
hottest expert is hot on every GPU's tokens alike). The PyTorch reference
supports one copy, so the redundancy count was not swept.

**The critical-path rule** (ShedLB, 2026-11-01 afternoon): a copy is made
only if it lowers the fullest bin anywhere by more than `eps * mean`, not
only the fullest of the bins it touches. On Mixtral it declines every copy
(the gain would be 0.14 of a 2.23 mean against the 0.22 tolerance) and
training returns to the plain time; on SmartMoE it makes the same copy as
before.

**Copies up to four, validated** (2026-11-01 evening, job 23418061, one A40
node, four GPUs; `-C 4` checksums at every fourth step, the last ones
compared): on the SmartMoE routing the plain arm, `-X` (two copies from the
start) and ShedLB with `-R 2`, `-R 3` and `-R 4` give one set of checksums
(2.576076019366897e+06, 1.091965310301654e+05); the Mixtral routing plain
against `-R 4` in training and serving likewise. A drifting hashed skew
(`-z 1.5 -p 10 -R 2`) made three copies and retired three as the hot set
moved, declined one by the cost rule, and matched its plain arm; a dominant
expert (`-z 3.0`, 32 experts of 2048 x 1024, `-R 4`) went to two, three
and four copies over successive rounds, every copy receiving the balancer's
notice and switching in the same step, and matched its plain arm through
the three-copy and four-copy stretches (steps 20-43 and 44-73). On this
node the default-shape experts' third copy was refused by the memory ledger
("no bin can hold one"), which is a question about the ledger's view of a
16 GB pool, not about the protocol. With experts this small the copies'
per-step cost is dominated by the exchange messages themselves (the `-R 4`
step 4x the plain step at 8 MB weights), which the declared cost, bytes
over bandwidth, does not see; at real shapes the bytes are what counts.

**A kernel bound, found and removed** (2026-11-02): the same ladder on
eight GPUs (job 23419187) gave the SmartMoE routing with three copies
(`-R 3`, `-R 4`) w2 checksums that differed from the plain arm's from step
24 on, the first step with three copies, in copy 0 alone, while every
operand copy 0 summed matched the other copies' bit for bit. `nblocks` in
`moe.cu` caps a launch at 65535 blocks of 256 threads, and four kernels
had no loop over the grid: the tree-node add of the copies' partials
(`addIntoKernel`), the two attention reshapes of `-w` and its weight update
(`toHeadsKernel`, `mergeHeadsKernel`, `sgdPlainKernel`). Past 16,776,960
elements they stopped. A 2048 x 8192 expert's gradient is 16,777,216
elements, so the add left its last 256 unadded, and only copy 0 of three
had the operands in the order that put the missing tail on the landed
side. At the Mixtral shape (4096 x 14336) the add missed 71% of every
block sum in every arm, the plain arm included (the four-partition tree
adds partition sums whether or not an expert is copied), so the arms
agreed with each other while all were wrong; `-w`'s reshapes and its
update were partial at that shape too. The four kernels now loop over the
grid. Measured before the fix, so to be re-measured: every training row
of the four-partition binary (the paragraphs above from "Copies up to
four" on, whose checksums came from that binary; on four GPUs the experts
were below the bound, so those arms agreed among themselves), and every
`-w` row at both shapes. The timing is expected to move little, since the
missing work was the tail of one element-wise pass. The re-measurement
(the checksum ladder on eight GPUs, then the touched rows) was queued on
Delta and cancelled unrun when development moved to Frontier; it is the
first job there (`FRONTIER_START.md`).

**The copy's cost, declared** (2026-11-01 evening): the balancer now prices
every copy as its share plus its exchanges with the other copies, each
exchange at the cost the application declares for a peer on the same node
or on another (`setLBReplica`'s two last arguments). The example measures
the cost at start-up: before the experts initialise, PE 0 sends one partial
(the fp32 gradient pair, 470 MB at the Mixtral shape) to a PE on its node
and to the first PE of the other node, twice each, and the second transfer's
time is printed (`Exchange probe:`) and declared by every expert scaled to
the balancing interval, zero when serving (`-G`), where the copies exchange
nothing. A retire now takes the highest-ranked copy, when the group fits
fewer copies or when the exchanges cost the critical path more than the
sharing saves. The same change lets a round that decides nothing cost
nothing: every PE learns the round's job-wide decision count
(`LBManager::LastRoundDecisions`) and the dispatchers resume the lookahead
inside the round's window when it is zero. Rule and plumbing:
`doc/internals/lb-multiplicity.md`.

## Measured 2026-11-02: Frontier, one node, eight MI250X GCDs

The validation ladder of `FRONTIER_START.md`, run on one Frontier node (job
5833510): eight processes of one PE, one GCD each, ROCm 7.14.1, the
reconverse/LCI build over libfabric, same-host channels through shared memory,
the vmm device pool (`+gpupoolalloc vmm +gpupoolsize 16384 +gpupoolchunk 2046
+gpushm`), the same traces and arguments as the A40 rows (`-i 40 -u 3 -B -C 4`,
ShedLB `-R 2 -f 20 -b 20 -a`, steady state steps 23-43). Part 0 (`replica_test`)
passes; in Part 1 the last step's `out2 w2` are one set per shape across plain,
`-X`, `-R 3` and `-R 4`, and in Part 2 the checksums are identical within every
pair below, training and serving.

Three bugs stood in the way, all fixed in this commit: on HIP a `hipMemcpyAsync`
with a peer's imported vmm heap on either side crashes inside ROCr's pointer
lookup, so the runtime's copies with an imported side go through
`hipMemcpyBatchAsync` (the two batch flushes for every size, and the persistent
get/put and parked-payload pull through one helper); a new copy received the
other copies' first block sums before its state had landed (the state comes from
another PE and lost the race here where it did not on the A40s), so the copy now
makes its landing buffers on first contact; and the serving branch of the
expert's step still ran partitions 0 and 1 of the four, a copy running partition
`rank()` instead of its block, so plain serving skipped half the dispatchers
and a serving copy served nothing (the `-X` copy issued 0 chunks all run). The
third is why the A40 serving columns of the 2026-11-01 section were re-measured
(the "Measured 2026-11-02: the same matrix with NCCL on Slingshot" section
below).

ms/step, 8 GCDs, SmartMoE L2 / Mixtral-8x7B L12 / OLMoE-1B-7B L12 routing:

| MoE layer alone                      | training | serving |
|--------------------------------------|---------:|--------:|
| SmartMoE, example plain              |     79.4 |    28.7 |
| SmartMoE, ShedLB `-R 2` (1 copy)     |     44.4 |    15.5 |
| SmartMoE, ShedLB `-R 4`              |     44.5 |    17.4 |
| SmartMoE, `-X` (1 copy from the start) |   72.0 |    26.4 |
| Mixtral, example plain               |    139.7 |    55.9 |
| Mixtral, ShedLB `-R 2` (no copy)     |    138.0 |    55.9 |
| Mixtral, `-X`                        |    133.7 |         |
| OLMoE, example plain                 |     33.4 |    21.0 |
| OLMoE, ShedLB `-R 2` (nothing to decide) | 35.1 |    21.8 |

| Transformer block (`-w`), unfused attention (fused: the 2026-11-04 section) | training | serving |
|--------------------------------------|---------:|--------:|
| SmartMoE, example plain              |     92.1 |    35.8 |
| SmartMoE, ShedLB `-R 2` (1 copy)     |     63.9 |    22.3 |
| Mixtral, example plain               |    176.1 |    71.8 |
| Mixtral, ShedLB `-R 2` (no copy)     |    175.6 |    72.0 |

**The baselines on the same node** (the PyTorch reference and Megatron-Core
0.20.0, same arguments and traces; the ROCm 7.1 PyTorch 2.13 wheel in a venv,
one rank per GCD over RCCL, the launcher `ref_run.sh` beside the example's
Frontier scripts). Two ROCm accommodations in `ref/moe_ref.py`: the
rendezvous master is given as an IPv4 literal on `hsn0` (by name, c10d first
tries an address family the node lacks and the ranks never meet), and
hipBLASLt refuses a bf16 GEMM with a transposed operand and an fp32 output
(`torch.mm(..., out_dtype=float32)`, then a 30x slower fallback with a warning
per call), so on ROCm each chunk's gradient leaves the GEMM in bf16 and is
widened before the fp32 accumulation: the same FLOPs, one more bf16 rounding
per chunk, checksums within 4e-7 of the example's. Megatron's experts here are
its `SequentialMLP` (the header line says so): the grouped path is Transformer
Engine's, which has no MI250X build (AMD's wheel carries gfx942 and gfx950 code
only, and its grouped-GEMM backends are for those parts), and the A40 rows
labelled "grouped GEMM" before 2026-11-03 were sequential too (the 2026-11-03
section). Its training rows carry fp32 master weights since 2026-11-04 (job
5847934; the 2026-11-02 values, the bf16 parameters updated in place, in
parentheses: masters cost 0 to 5 ms a step). FastMoE/FasterMoE need a HIP port
of their extension and are not run.

| MoE layer alone, ms/step                 | training | serving |
|------------------------------------------|---------:|--------:|
| SmartMoE: example plain                  |     79.4 |    28.7 |
| SmartMoE: example ShedLB `-R 2`          |     44.4 |    15.5 |
| SmartMoE: PyTorch reference, static      |     93.0 |    33.4 |
| SmartMoE: PyTorch reference, EPLB packing every 10 steps |  102.7 | |
| SmartMoE: PyTorch reference, one static copy |  80.8 |         |
| SmartMoE: Megatron-Core, sequential experts, masters | 76.9 (73.8) | 26.8 |
| SmartMoE: Megatron-Core, capacity factor 1.25 | 75.0 (74.2) |         |
| Mixtral: example plain                   |    139.7 |    55.9 |
| Mixtral: PyTorch reference, static       |    170.7 |    58.5 |
| Mixtral: Megatron-Core, masters          | 145.4 (144.7) | 50.0 |
| OLMoE: example plain                     |     33.4 |    21.0 |
| OLMoE: PyTorch reference, static         |     47.1 |    27.3 |
| OLMoE: Megatron-Core, masters            | 44.8 (45.2) |  22.3 |

What one node changes. The reference's dispatch all-to-all moves 169 MB in
and out of the busiest rank in 1.7 ms here (92 GB/s over Infinity Fabric)
against 18 ms on one A40 node (10.5 GB/s over PCIe) and worse across the
Delta node link, so the baselines lose the exchange cost that put them 2x
behind the example's plain arm on the A40 pair: on SmartMoE the reference's
step is now the hot rank's 84 ms of GEMMs plus 3 ms of combine, and
Megatron's lock-step structure is 3% under the example's plain arm. The
example did not gain, because its exchange overlaps the step's compute on
either machine. What remains is the balancer: on SmartMoE the example with
ShedLB is 1.73x faster than Megatron and 2.1x faster than the reference in
training, 1.7x and 2.2x in serving, with a copy that halves the hot GPU's
work, which no re-placement can do with one expert per hot GPU (EPLB's packing
costs the reference 10%; its static copy of the hottest expert, 2x slower on
the A40 pair for its cross-link partial all-reduce, gains 13% here and still
trails the example's copy by 1.8x). On Mixtral and OLMoE, where there is nothing to
balance, the three are within 1.2x of each other, the example's plain arm
fastest in training. The cross-node regime of the Delta tables is the next
section.

Against the two-node A40 rows (the corrected ones, "Measured 2026-11-02: the
same matrix with NCCL on Slingshot" below): SmartMoE training is the same step
(79.4 against 81.1) and the balancer's gain is larger (1.79x against 1.56x, the
copy's exchange crossing a GCD pair instead of a node link); Mixtral training is
1.16x faster and the balancer is correctly idle; OLMoE, whose step is the token
exchange, is 7x faster on one node than across the Delta node link. In serving,
SmartMoE is 28.7/15.5 here against 35.3/27.4 there (plain/ShedLB: the copy's
serving gain is 1.85x against 1.3x), Mixtral 55.9 against 112 to 130, OLMoE 21.0
against 249.9.

## Measured 2026-11-02, later: Frontier, two nodes, sixteen MI250X GCDs

The three-way comparison again on two Frontier nodes (job 5837505, frontier02930
and frontier03188, one hour): 16 processes of one PE, one GCD each, eight per
node. Every dispatcher still sends 8192 tokens, so a step carries twice the
tokens of the one-node rows and the skew doubles with it (SmartMoE's hottest
expert takes 48% of twice the tokens: 7.7x a GCD's share with two experts per
GCD). The example runs the buddy pool (`+gpupoolalloc buddy +gpupoolsize 2048
+gpupoolarenas 8`: across nodes ROCr cannot dma-buf-export vmm memory; on one
node of this pair the buddy pool measures 78.5 plain and 47.4 ShedLB against
the vmm pool's 79.4 and 44.4 above, so the pool is not a variable). The
baselines run RCCL over Slingshot through OLCF's aws-ofi-nccl 1.19.2 plugin
(the `rccl-net-plugin` module's ROCm 7.1.1 build and its CXI variables, set by
`ref_wrap.sh` when the step spans nodes); without it RCCL falls back to TCP
sockets on `hsn0`, which put the reference's SmartMoE step at 254 ms, its static
copy at 276 and Megatron at 291 (kept under `_socket` tags in the logs, not
rows). Megatron-Core does not run Mixtral on 16 ranks (eight experts must divide
the expert-parallel size). The Mixtral and OLMoE traces hold 21 steps of 16 x
8192 tokens, so every implementation wraps them twice by the same rule.
Checksums: identical within every example pair; the reference within bf16
noise of the example at every shape; Megatron prints none.

| MoE layer alone, ms/step, 16 GCDs              | training | serving |
|------------------------------------------------|---------:|--------:|
| SmartMoE: example plain                        |    132.6 |    58.8 |
| SmartMoE: example ShedLB `-R 2` (1 copy)       |     90.8 |    32.7 |
| SmartMoE: example ShedLB `-R 4` (2 copies from step 40) | 92.4 | 32.2 |
| SmartMoE: example ShedLB `-R 4 -f 10 -b 10` (3 copies by step 30) | 88.1 | 30.9 |
| SmartMoE: example `-X` (1 copy from the start) |     96.6 |    36.9 |
| SmartMoE: PyTorch reference, static            |    162.5 |    60.0 |
| SmartMoE: PyTorch reference, EPLB packing every 10 steps | hung (below) | |
| SmartMoE: PyTorch reference, one static copy   |    106.7 |         |
| SmartMoE: Megatron-Core, sequential experts, masters | 142.0 (137.4) | 53.8 |
| SmartMoE: Megatron-Core, capacity factor 1.25  | 141.7 (139.2) |     |
| Mixtral: example plain                         |    305.1 |   164.6 |
| Mixtral: example ShedLB `-R 2` (no copy; exit crash below) | 306.9 | 160.0 |
| Mixtral: PyTorch reference, static             |    328.9 |   115.6 |
| OLMoE: example plain                           |    159.2 |   163.1 |
| OLMoE: example ShedLB `-R 2` (nothing to decide) | 163.4 |   161.9 |
| OLMoE: PyTorch reference, static               |     74.6 |    50.5 |
| OLMoE: Megatron-Core, masters                  | 82.6 (83.6) | 50.5 |

| Transformer block (`-w`), 16 GCDs, unfused attention (fused: the 2026-11-04 section) | training | serving |
|------------------------------------------------|---------:|--------:|
| SmartMoE: example plain                        |    145.1 |    56.2 |
| SmartMoE: example ShedLB `-R 2` (1 copy)       |     96.1 |    38.6 |
| Mixtral: example plain                         |    307.0 |   126.7 |
| Mixtral: example ShedLB `-R 2` (no copy)       |    312.2 |   134.9 |

What two nodes change, shape by shape.

**SmartMoE.** The balancer's gain holds across the node link: ShedLB is 1.46x
faster than plain in training (1.79x on one node) and 1.8-1.9x in serving.
Per 8 x 8192 tokens the balanced step is flat (44.4 ms on one node, 45.4 on two)
while the reference's static step only falls from 93 to 81 and Megatron's from
77 to 71 (with masters, re-measured 2026-11-04; the 2026-11-02 values in
parentheses in the table), so against the baselines the balanced example is
1.6x faster than Megatron and 1.8x faster than the reference in training, 1.7x
and 1.9x in serving (30.9 against 53.8 and 60.0). The reference's static copy is now its
best training arm (106.7, 1.5x over its static; 13% over on one node): the
copy halves the hot rank where the one-node step was already compute-bound.
The balancer's step stops near 88-91 ms whatever the copy count: with a round
every 10 steps it makes three copies by step 30 (the hot GCD's tokens fall to
1.95x the mean, from 7.7x), but the hot GCD's time rises from 88 to 92-107 ms
because every copy adds a partial exchange that now crosses the node link, so
one copy buys most of the gain and the rest is paid back in exchange. The
EPLB arm of the reference hung twice before its fourth step, inside RCCL's
channel setup (38 proxy connections against the 48 of the arms that passed),
once over sockets and once over the plugin, while the five other baseline arms
set up in seconds; it hangs before any balancing code runs and is not
root-caused.

**Mixtral, eight experts on sixteen GCDs.** The example's block map lands
expert e on PE e: all eight on node 0, each with twice the tokens, node 1
dispatching only, hence 2.2x the one-node step. ShedLB makes no copy: every
expert's load (4.3-5.3) exceeds the target (2.4), but the rule copies only
when the fullest bin falls, and with eight bins loaded alike one copy leaves
another at the top ("would leave the fullest bin at 5.285 of 5.285: no copy");
using the empty node needs a round that copies every expert, which the
one-copy-per-round rule cannot express. The reference's block map (`e * P / E`)
spreads the experts over the even ranks of both nodes: its static training step
is level with the example's (328.9 against 305.1) and its serving step 1.4x
faster (115.6 against 164.6), a placement effect, not a compute one.

**OLMoE, where the step is the top-8 exchange.** The example's step grows from
33 to 159 ms (4.8x for twice the tokens) while the reference's grows from 47 to
75 and Megatron's from 45 to 83, and in serving the reference's 50.5 is 3.2x
under the example's 163.1. A step moves 4.3 GB of tokens and outputs across
the link in each direction (16 x 8192 x 8 slots x 4 KB, half of it crossing,
dispatch and combine): the reference's serving step moves it at 85 GB/s per
direction, close to the node's four NICs, the example's at 26 GB/s, one NIC's
worth. On the Delta pair the corrected rows ("Measured 2026-11-04" below) put
the baselines ahead of the example there too (reference 194.7 and Megatron
248.1 against 243.1 in training, 172.8 and 169.6 against 249.9 in serving);
only the socket-transport rows of 2026-11-01 (example 238, reference 333,
Megatron 456) had it the other way. Whether the example's sixteen LCI endpoints spread over the
four NICs of a node, as the baselines' plugin does ("found 4 nics",
`NCCL_CROSS_NIC=1`), is the open performance item of this section; nothing in
the runtime chooses a NIC today.

**Exit crash.** Three of the four Mixtral ShedLB arms (no copy made) ended in a
segfault on PE 14 after their last step, with the checksums already printed:
`ConverseExit` -> `CommBackendLCI::exit` -> LCI `free_device` ->
`wait_drained` -> `progress` delivers a device-channel receive completion ->
`CkRdmaDeviceRecvHandler` -> `CkCallback::send` -> `CkSendMsgBranch` ->
`issueAm` on the backend being torn down. A posted device receive (a dispatcher
PE's landing buffer for a step that never comes) completes during the exit
drain and its handler sends; the fix belongs in the exit sequence (drain
without delivering, or cancel the posted receives before the backend exits).
Open; the rows above are unaffected.

## Measured 2026-11-04: Frontier, the block rows with the fused attention

The HIP build's attention sublayer was unfused (the score GEMM, a softmax
kernel, the PV GEMM and their backward) while the references' attention on
MI250X is PyTorch's fused flash kernel, the mismatch the Delta correction of
2026-11-03 found on the A40s and cuDNN removed there. On ROCm the same kernels
are aotriton's, the library PyTorch's `scaled_dot_product_attention` dispatches
to, and the torch 2.13+rocm7.1 wheel ships it with its gfx90a kernel images and
headers; `moe.cu` calls its v3 API as PyTorch 2.13 does (`MOE_AOTRITON_ATTN`,
the Makefile's HIP branch; the library is staged beside the runtime's HIP rather
than the wheel's, `FRONTIER_START.md`). Q, K, V and dQ, dK, dV are the same
strided views of the packed arrays as on the cuDNN path, the causal forward
keeps the softmax statistics for the backward, and nothing is copied per head.
The startup self-check (one training step on the same weights, fused against
unfused) measured 3.6e-3 on the output and 2.0e-3 and 1.1e-3 on the weight
gradients at head size 64 (SmartMoE), 3.5e-3, 2.1e-3 and 1.1e-3 at 128
(Mixtral), the A40 values, in every run. Job 5847934, frontier04265 and
04266: the twelve block arms on both builds (the fused `moe`, the unfused
`moe.attn_unfused` from the same tree), the unfused build's plain arms on the
same nodes, one node and two. Within each build the plain and ShedLB arms agree
to the last digit, training and serving, both routings, both node counts; the
two builds differ in the fifth digit of `out2` after forty training steps, the
rounding of the self-check carried through the run. Steady state steps 23-43.

| transformer block, 8 GCDs, ms/step               | SmartMoE train | SmartMoE serve | Mixtral train | Mixtral serve |
|--------------------------------------------------|---------------:|---------------:|--------------:|--------------:|
| example plain, fused attention                   |           85.8 |           32.3 |         169.2 |          68.6 |
| example ShedLB, fused attention                  |           53.6 |           18.7 |         169.3 |          68.6 |
| example plain, unfused attention, same node      |           92.7 |           36.1 |         179.8 |          75.4 |
| MoE layer alone, plain (2026-11-02)              |           79.4 |           28.7 |         139.7 |          55.9 |

| transformer block, 16 GCDs, ms/step              | SmartMoE train | SmartMoE serve | Mixtral train | Mixtral serve |
|--------------------------------------------------|---------------:|---------------:|--------------:|--------------:|
| example plain, fused attention                   |          141.4 |           57.2 |         299.8 |         127.8 |
| example ShedLB, fused attention                  |           90.8 |           34.2 |   303.3 (a)   |   134.7 (a)   |
| example plain, unfused attention, same nodes     |          148.4 |           57.7 |         311.9 |         128.3 |
| MoE layer alone, plain (2026-11-02, later)       |          132.6 |           58.8 |         305.1 |         164.6 |

(a) all forty steps and the checksums complete, then the exit segfault of the
2026-11-02 two-node section (the LCI backend's exit drain delivering a device
receive, here on PE 10); the row stands, the crash is the open item.

What the fused attention changes. On one node the attention sublayer's
increment over the MoE-only training step falls from 13.3 to 6.4 ms on
SmartMoE's routing and from 40.1 to 29.5 on Mixtral's (head size 128, where
the unfused scores were 4 MB per head and sequence); in serving from 7.4 to
3.6 and from 19.5 to 12.7. The balancer's gain in the block grows with it,
because the attention runs on the dispatcher's lane in both arms and shrinking
it leaves more of the step to the experts: 1.60x in training (85.8 against
53.6; 1.44x unfused) and 1.73x in serving (32.3 against 18.7; 1.61x unfused).
On two nodes the training step gains the same 7 to 12 ms (141.4 against
148.4, 299.8 against 311.9) and the serving step nothing: at 57 and 128 ms it
is the cross-node exchange, which the attention overlaps either way (on
SmartMoE's routing the block serves in the MoE-only step's time, 57.2 against
58.8). The Mixtral serving step with the block is 37 ms under the MoE-only one
on two nodes (127.8 and 128.3 against 164.6), as it was on 2026-11-02 (126.7):
the attention's compute spaces the dispatchers' sends out, and the exchange
that bounds the MoE-only step runs with less contention. The mechanism is not
measured; it is an open item of this section. Against the A40 pair's fused
block rows (the 2026-11-04 Delta section, eight A40s over two nodes against
eight GCDs in one): SmartMoE training is the same step (85.8 against 89.3
plain, 53.6 against 57.8 with ShedLB) and serving gains more from the copy here
(18.7 against 27.0, the exchange crossing a GCD pair instead of a node link);
Mixtral training is the same (169.2 against 170.1) and its serving 1.7x faster
(68.6 against 118.3), the one-node exchange again.

**The baselines' block rows** (the same allocation: the PyTorch reference and
Megatron-Core with `--attn`, 32 heads and sequences of 1024; the harness's
attention is PyTorch's `scaled_dot_product_attention`, on MI250X the aotriton
flash kernel the example now runs, so the fixture is the same on both sides;
Megatron with fp32 masters; every two-node log names the Libfabric transport;
Megatron does not run Mixtral on 16 ranks). The example's rows are those above.

| transformer block, 8 GCDs, ms/step               | SmartMoE train | SmartMoE serve | Mixtral train | Mixtral serve |
|--------------------------------------------------|---------------:|---------------:|--------------:|--------------:|
| example plain, fused attention                   |           85.8 |           32.3 |         169.2 |          68.6 |
| example ShedLB, fused attention                  |           53.6 |           18.7 |         169.3 |          68.6 |
| PyTorch reference, static                        |          102.3 |           33.5 |         198.5 |          65.2 |
| Megatron-Core, sequential experts, masters       |           87.3 |           30.4 |         177.7 |          61.3 |

| transformer block, 16 GCDs, ms/step              | SmartMoE train | SmartMoE serve | Mixtral train | Mixtral serve |
|--------------------------------------------------|---------------:|---------------:|--------------:|--------------:|
| example plain, fused attention                   |          141.4 |           57.2 |         299.8 |         127.8 |
| example ShedLB, fused attention                  |           90.8 |           34.2 |   303.3 (a)   |   134.7 (a)   |
| PyTorch reference, static                        |          174.5 |           63.9 |         360.8 |         125.6 |
| Megatron-Core, sequential experts, masters       |          150.6 |           58.1 |               |               |

The attention costs the baselines what it costs the example now. Over their
MoE-only steps on one node the reference pays 9.3 ms in SmartMoE training
(102.3 against 93.0) and 28 in Mixtral's (198.5 against 170.7), Megatron 10.4
(87.3 against 76.9) and 32 (177.7 against 145.4), the example 6.4 and 29.5; in
serving the baselines pay 0 to 11 ms and the example 3.6 and 12.7. The block
ratios, the same fixture on both sides: on SmartMoE's routing the example
with ShedLB trains 1.6x faster than Megatron-Core and 1.9x faster than the
reference on one node (1.7x and 1.9x on two) and serves 1.6x and 1.8x faster
(1.7x and 1.9x on two); the plain example is level with Megatron in training
and serving at either node count (85.8 against 87.3 and 32.3 against 30.4;
141.4 against 150.6 and 57.2 against 58.1) and 1.2x ahead of the reference in
training. On Mixtral's routing, where there is nothing to balance, the example
trains 1.05x faster than Megatron and 1.17x faster than the reference on one
node (1.2x over the reference on two), and in serving both baselines are ahead
of it on one node (61.3 and 65.2 against 68.6: one all-to-all against one
message per expert on an exchange-bound step, as in the MoE-only rows) and the
reference level with it on two (125.6 against 127.8). The ratios sit a little
under the MoE-only ones (1.73x over Megatron and 2.1x over the reference in
one-node training), the attention being a fixed cost on every side. On the
A40 pair the balanced example's block margin over Megatron is 2.2x in
training; on the Frontier pair 1.7x, the balanced step there being the copy
exchange's floor of the 2026-11-02 two-node section.

## Measured 2026-11-02: the same matrix with NCCL on Slingshot, and a serving fix

> Superseded in part (2026-11-03): the Megatron-Core rows below ran
> sequential experts, and the FasterMoE serving rows paid a per-step
> garbage collection; "Measured 2026-11-03" has the cause and "Measured
> 2026-11-04" the eight-GPU baseline rows re-measured and the corrected
> ratios.

**Correction.** Every NCCL baseline of the 2026-11-01 section (the PyTorch
reference, Megatron-Core, FastMoE, FasterMoE) ran without the `aws-ofi-nccl`
plugin, so across the two nodes NCCL did not use Slingshot through libfabric:
its logs at `NCCL_DEBUG=INFO` show `NET/IB : No device found` and no plugin,
which leaves its socket transport. The example's own exchange goes through
LCI over the cxi provider. The baselines' two-node numbers were therefore
too slow and the ratios quoted there too large; the one-node (four-GPU)
rows, where NCCL stays on P2P and shared memory, and the example's own rows
stand. `ref/refwrap.sh` now sets what `module load aws-ofi-nccl/1.19.2` sets
(`NCCL_NET_PLUGIN=ofi`, `FI_PROVIDER=cxi`, `FI_CXI_DISABLE_HOST_REGISTER=1`,
`FI_MR_CACHE_MONITOR=userfaultfd`, the plugin and libfabric 2.3.1 on the
library path) and logs NCCL's INIT and NET lines, so every rank log carries
`Using network Libfabric`. Without the userfaultfd monitor one arm (Mixtral,
EPLB packing, 704 MB experts) hung inside NCCL's tree connection on all
eight ranks, the libfabric memhooks deadlock with CUDA (job 23452305); with
it, the arm ran. The FasterMoE calibration on the plugin: the 470 MB
broadcast to 8 ranks 154 -> 87.8 ms, so its shadow policy ran with
NETBW 5.35 GB/s instead of 3.05.

Jobs 23452305 (gpub002+012, MoE layer alone, SmartMoE and Mixtral) and
23454982 (gpub030+089, OLMoE, the block variants, the re-run arm), bf16, 8
GPUs, steady state steps 23-43 of 40 + 3, one run per arm; the example's
arms in the same jobs. The first column of each pair is the 2026-11-01 value
for reference, the second the re-measurement.

**MoE layer alone** (ms/step; "old | new"):

| SmartMoE training routing, layer 2        | training        | serving         |
|-------------------------------------------|----------------:|----------------:|
| example plain                             | 79.9 \| 81.1    | 33.4 \| 35.3 |
| example ShedLB, critical-path rule (1 copy)| 50.9 \| 52.1   | 27.8 \| 27.4 (23.4 settled) |
| PyTorch reference, static                 | 178.4 \| 158.5  | 100.8 \| 78.4   |
| PyTorch reference, EPLB packing every 10  | 201.6 \| 171.4  | 127.9 \| 97.2   |
| PyTorch reference, one static copy        | 232.5 \| 128.9  | 73.3 \| 53.5    |
| Megatron-Core, grouped GEMM               | 164.6 \| 126.6  | 95.4 \| 74.0    |
| Megatron-Core, capacity factor 1.25       | 165.1 \| 127.0  |                 |
| FastMoE as shipped                        | 239.8 \| 190.2  | 106.6 \| 84.1   |
| FasterMoE (schedule + shadows)            | 113.1 \| 76.4   | 130.7 \| 97.5   |

| Mixtral-8x7B routing, layer 12            | training        | serving         |
|-------------------------------------------|----------------:|----------------:|
| example plain                             | 162.0 \| 161.8  | 120.1 \| 117 (112-130) |
| example ShedLB, critical-path (no copy)   | 164.8 \| 164.2  | 125.6 \| 124.8 |
| PyTorch reference, static                 | 334.5 \| 275.5  | 197.6 \| 140.6  |
| PyTorch reference, EPLB packing every 10  | 333.9 \| 289.3  | 203.8 \| 153.9  |
| PyTorch reference, one static copy        | 673.1 \| 351.9  | 227.2 \| 165.4  |
| Megatron-Core, grouped GEMM               | 351.6 \| 263.0  | 182.3 \| 131.2  |
| Megatron-Core, capacity factor 1.25       | 337.3 \| 262.9  |                 |
| FastMoE as shipped                        | 386.3 \| 300.6  | 193.2 \| 129.9  |
| FasterMoE (no shadow chosen)              | 376.3 \| 259.7  | 254.0 \| 161.9  |

| OLMoE-1B-7B routing, layer 12             | training        | serving         |
|-------------------------------------------|----------------:|----------------:|
| example plain                             | 237.7 \| 243.1  | 243.6 \| 249.9 |
| example ShedLB (nothing to decide)        | 262.3 \| 240.8  | 248.8 \| 269.7 |
| PyTorch reference, static                 | 333.2 \| 194.7  | 314.5 \| 172.8  |
| PyTorch reference, EPLB packing every 10  | 308.5 \| 200.9  | 284.0 \| 178.1  |
| PyTorch reference, one static copy        |   -   \| 215.7  |   -   \| 189.1  |
| Megatron-Core, grouped GEMM               | 455.6 \| 261.0  | 280.2 \| 173.3  |
| Megatron-Core, capacity factor 1.25       |   -   \| 261.4  |                 |
| FastMoE as shipped                        | 697.4 \| 556.5  | 332.5 \| 268.2  |
| FasterMoE                                 | 641.1 \| 350.2  | 403.0 \| 230.9  |

**Transformer block** (`-w` / `--attn`; the example's rows are the first
with the attention kernels of 2026-11-02, which loop over the grid):

| SmartMoE routing, block                   | training        | serving         |
|-------------------------------------------|----------------:|----------------:|
| example plain                             | 98.0 \| 102.7   | 35.6 \| 36.4 |
| example ShedLB (1 copy)                   | 63.8 \| 68.9    | 30.0 \| 28.8 |
| PyTorch reference, static                 | 185.8 \| 166.2  | 103.2 \| 80.4   |
| PyTorch reference, EPLB packing every 10  | 212.1 \| 181.6  | 131.9 \| 100.4  |
| PyTorch reference, one static copy        | 241.2 \| 138.1  | 73.7 \| 55.5    |
| Megatron-Core, grouped GEMM               | 175.6 \| 137.0  | 97.8 \| 76.9    |
| Megatron-Core, capacity factor 1.25       | 174.7 \| 137.3  |                 |
| FastMoE as shipped                        | 249.6 \| 201.4  | 109.9 \| 87.3   |
| FasterMoE                                 | 119.2 \| 86.5   | 132.5 \| 100.0  |

| Mixtral routing, block                    | training        | serving         |
|-------------------------------------------|----------------:|----------------:|
| example plain                             | 178.9 \| 183.7  | 137.9 \| 148.1 |
| example ShedLB, critical path (no copy)   | 188.4 \| 184.0  | 134.2 \| 118.6 |
| PyTorch reference, static                 | 358.3 \| 308.3  | 221.6 \| 149.2  |
| PyTorch reference, EPLB packing every 10  | 362.3 \| 319.0  | 205.8 \| 161.1  |
| PyTorch reference, one static copy        | 695.9 \| 384.9  | 242.8 \| 174.2  |
| Megatron-Core, grouped GEMM               | 368.3 \| 296.1  | 202.6 \| 141.2  |
| Megatron-Core, capacity factor 1.25       | 369.2 \| 294.7  |                 |
| FastMoE as shipped                        | 421.0 \| 333.9  | 201.4 \| 140.0  |
| FasterMoE                                 | 417.9 \| 287.1  | 267.4 \| 172.3  |

What the corrected numbers say. On SmartMoE's training routing the
example with ShedLB is 1.5x faster than FasterMoE and 2.4x faster than
Megatron-Core in training (2.2x and 3.2x were quoted), and 3.6x faster than FasterMoE and 2.7x faster than Megatron-Core in serving (3.4x over Megatron-Core was quoted)
in serving. On Mixtral's routing the plain example is 1.6x faster than
Megatron-Core and FasterMoE in training (2.2x and 2.3x were quoted), and
in serving the plain example (112-130 over six runs) is level with Megatron-Core (131.2) and FastMoE (129.9) and 1.4x faster than FasterMoE (1.5x and 1.6x were quoted). On OLMoE's routing, where the step is the
token exchange (top-8), the PyTorch reference's one all-to-all on the
plugin is now faster than the example's one message per expert, 194.7
against 243.1 in training and 172.8 against 249.9 in serving; the 2026-11-01
text had the example 1.3-1.4x ahead there, which was the socket transport,
not the exchange. The baselines gained 11-48% from the plugin, the most
where fp32 partials crossed the node link every step (the reference's static
copy) and in FasterMoE's shadow broadcasts.

**A serving bug, found and removed** (2026-11-02). In the same jobs the
example's serving arms came out 16-30% slower than on 2026-11-01 while its
training arms matched (SmartMoE 43.3 against 33.4, Mixtral 139.3 against
120.1), and in serving the arm with a copy gave a different out2 than the
plain arm at every checksum step from the copy's first step on, where the
training pairs agreed to the last digit. The same node pair ran this
morning's binary against the 2026-11-01 17:13 one: 42.6 and 43.0 against
33.3 and 33.3 ms/step, with the GPU time per step halved (24 against 40 ms),
the same message and kernel counts, and two to three times the LCI send
retries. The serving branch of `Expert::runStep` still read `for (p = 0;
p < 2; p++)` from before the four-partition tree, and a copy ran partition
`rank()` instead of its block: in serving, the dispatchers of partitions 2
and 3 had their output segments never written and sent on never-recorded
events, ran free and flooded the sends. The forward now runs every partition
(a copy's other partitions hold no tokens). Validated in job 23456024 on
gpub001+002 (and 23465756 on gpub091+099): SmartMoE plain serving 35.3 and 35.3 ms/step (43.3 with the bug; this
morning's binary on the same nodes 42.1), with a copy 27.4 (23.4 settled);
OLMoE 249.9 and 269.7; the block variants 36.4 and 28.8 (SmartMoE), 148.1
and 118.6 (Mixtral); every plain-versus-copy serving pair equal to the last
digit, and the training checksum unchanged by the fix. Mixtral's serving
step spreads 112-130 ms over six runs on two node pairs (one run on
gpub001+002 at 155.0), with or without lookahead (114.2 against 116.8) and
with two or four partitions: its step is the 128 MB slab exchange and the
spread is the exchange's, not the binary's; the 2026-11-01 band (120.1 +-
1.6, three runs on one pair) understated it. The serving checksums of the "Copies up to
four, validated" paragraph came from that binary and covered half the
dispatchers' outputs; the training checksums there were right.


**Two partitions against four** (job 23465756, same source built with
`-DMOE_K_PART=2`, `moe.k2`): SmartMoE plain serving 33.9 and 33.7 ms/step
against 35.3 and 35.3 with four, with a copy 28.6 against 27.4; Mixtral
serving 130.2 and 115.8 against 116.8 and 112.4, inside its spread; the
block variant 150.3 against 148.1. The four-partition build is the default
because copies up to four need it; a serving deployment with at most two
copies per expert gets the 2026-11-01 step back from the two-partition
build. The two builds' checksums differ in the ninth digit (the partition
boundaries set the chunk sequence) and each build's plain and copy arms
agree with each other.

## Measured 2026-11-03: the baselines as their authors run them (one node)

**Correction.** An audit of the baseline harnesses against the way their
authors run them (prompted by the question of what a fair comparison must
match) found two faults, both of which slowed the baselines, and one
mismatch. Measured in job 23496712, one A40 node, four GPUs, the same
traces and arguments as the 2026-11-01 four-GPU rows.

1. **Megatron-Core's experts were sequential.** In Megatron-Core 0.20
   `get_gpt_layer_local_spec` builds `SequentialMLP` whatever
   `moe_grouped_gemm` says: there is no `GroupedMLP` in 0.20 and the
   `grouped_gemm` package is never used. The grouped path is the Transformer
   Engine spec, `TEGroupedMLP` on TE's `GroupedLinear`, which is Megatron-LM's
   default `--transformer-impl`. Every "Megatron-Core, grouped GEMM" row of
   the 2026-10-31, 2026-11-01 and 2026-11-02 sections ran one GEMM per
   expert; the harness printed "grouped GEMM" from its flag (the two fp32
   rows of `ref/README.md`'s Megatron table, sequential and "grouped", are
   identical for that reason). `ref/megatron_moe.py` now takes `--te`
   (and `--permute-fusion` for TE's fused token permutation) and prints the
   module class that ran.
2. **FasterMoE's serving step paid a full garbage collection per step.**
   FastMoE's fused forward records autograd graphs whatever the grad mode;
   the harness collected them every step to bound memory, and that
   collection scanned every long-lived object: 95 ms of forward in serving
   against 38 ms for the same forward in training (job 23452305, SmartMoE
   routing, eight GPUs). `gc.freeze()` before the loop limits each
   collection to the step's own garbage. Every FasterMoE serving row above
   carries the old cost.
3. **Master weights.** The Megatron and FastMoE harnesses updated their bf16
   parameters in place. Megatron-LM's bf16 optimizer keeps fp32 master
   weights and casts into the bf16 parameters each step, as the example does
   (an update below 2^-9 of a weight is rounded away by an in-place bf16
   update). Both harnesses now take `--master`; its cost is measured in the
   eight-GPU re-run, not here.

| four GPUs, ms/step                                   | SmartMoE train | SmartMoE serve | Mixtral train | Mixtral serve | OLMoE train | OLMoE serve |
|------------------------------------------------------|---------------:|---------------:|--------------:|--------------:|------------:|------------:|
| Megatron-Core, sequential experts (as published)     |      90.2 (a)  |           52.0 |         272.6 |         138.1 |       252.0 |       167.2 |
| Megatron-Core, Transformer Engine grouped experts    |           81.4 |           48.6 |         250.8 |         139.5 |       253.0 |       167.3 |
| Megatron-Core, TE grouped experts, fused permute     |           81.9 |           48.2 |         246.7 |         136.5 |       255.2 |       166.2 |
| FasterMoE, serving as published                      |                |       82.3 (a) |               |     188.2 (a) |             |             |
| FasterMoE, serving with the collection fixed         |                |           27.0 |               |         132.4 |             |       184.3 |
| FastMoE plain, serving                               |                |           52.0 |               |         134.6 |             |       216.4 |
| example plain (2026-11-01 rows)                      |           55.4 |           32.9 |         137.8 |         108.7 |             |             |
| example ShedLB (2026-11-01 rows)                     |           50.4 |           27.3 |         137.3 |         100.7 |             |             |

(a) the 2026-11-01 four-GPU row, another node; this job's sequential
controls reproduced the other 2026-11-01 one-node rows within 1% (52.0
against 52.1, 272.6 against 270.8, 138.1 against 138.1), and its SmartMoE
training control was lost to a harness error.

What the numbers say. The Transformer Engine experts are worth 10% on
SmartMoE training, 7% on SmartMoE serving and 8% on Mixtral training, and
nothing on Mixtral serving (two experts per GPU, nothing to group) or on
OLMoE (the step is the top-8 exchange); fused permutation is inside the
run-to-run band everywhere, so the eight-GPU re-run uses the TE experts
without it. FasterMoE's serving step on SmartMoE's routing is 27.0, equal
to the example with ShedLB (27.3) and ahead of the plain example (32.9):
its shadow of the hottest expert does in serving what ShedLB's copy does,
and the 2026-11-01 harness hid that behind the collection. On Mixtral's
flat routing it is level with FastMoE plain and Megatron-Core. The
eight-GPU serving claim over FasterMoE in the 2026-11-02 section (3.6x on
SmartMoE's routing, 1.4x on Mixtral's) is therefore void until the
eight-GPU matrix is re-measured with these harnesses, and the eight-GPU
Megatron-Core ratios there are too large by the margins above. Megatron's
DeepEP ("flex") dispatcher and its NVLink-multicast inference dispatcher
cannot run on Delta's A40 nodes (no InfiniBand, no NVLink), so the
all-to-all dispatcher stays the Megatron configuration.

## Measured 2026-11-04: the eight-GPU matrix with the baselines as their authors run them

Job 23497275, two A40 nodes (gpub026 and gpub032), the baseline arms of the
2026-11-02 matrix re-run with the corrections of the 2026-11-03 section:
Megatron-Core with the Transformer Engine grouped experts, fp32 master
weights on every training arm of Megatron and FastMoE, and FasterMoE's
serving step without the per-step collection. The example's rows are those
of the 2026-11-02 section (its serving rows from the fixed binary); its
block rows are re-measured with the fused attention in the next section.
"Was" is the 2026-11-02 value.

| MoE layer alone, 8 GPUs, ms/step               | SmartMoE train | SmartMoE serve | Mixtral train | Mixtral serve | OLMoE train | OLMoE serve |
|------------------------------------------------|---------------:|---------------:|--------------:|--------------:|------------:|------------:|
| example plain                                  |           81.1 |           35.3 |         161.8 |    112 to 130 |       243.1 |       249.9 |
| example ShedLB                                 |           52.1 |           27.4 |         164.2 |         124.8 |       240.8 |             |
| PyTorch reference, static (2026-11-02)         |                |                |         275.5 |         140.6 |       194.7 |       172.8 |
| Megatron-Core, TE experts, masters             | 119.8 (was 126.6) | 74.4 (74.0) | 243.5 (263.0) | 131.9 (131.2) | 248.1 (261.0) | 169.6 (173.3) |
| Megatron-Core, same, capacity factor 1.25      | 119.9 (127.0)  |                | 244.9 (262.9) |               | 248.1 (261.4) |           |
| Megatron-Core, TE experts, no masters          |          117.3 |                |               |               |             |             |
| FastMoE plain, masters                         | 192.5 (190.2)  |    84.1 (same) | 301.3 (300.6) |  129.9 (same) | 555.9 (556.5) | 268.2 (same) |
| FasterMoE, masters; serving fixed              |   78.6 (76.4)  |    40.3 (97.5) | 258.2 (259.7) | 105.1 (161.9) | 350.7 (350.2) | 171.9 (230.9) |

| transformer block, 8 GPUs, ms/step             | SmartMoE train | SmartMoE serve | Mixtral train | Mixtral serve |
|------------------------------------------------|---------------:|---------------:|--------------:|--------------:|
| example plain, unfused attention (2026-11-02)  |          102.7 |           36.4 |         183.7 |         148.1 |
| example ShedLB, unfused attention (2026-11-02) |           68.9 |           28.8 |         184.0 |         118.6 |
| Megatron-Core, TE experts, masters             | 129.3 (137.0)  |    77.3 (76.9) | 276.5 (296.1) | 141.8 (141.2) |
| Megatron-Core, same, capacity factor 1.25      | 128.9 (137.3)  |                | 276.2 (294.7) |               |
| FastMoE plain, masters                         | 202.5 (201.4)  |    87.3 (same) | 335.8 (333.9) |  140.0 (same) |
| FasterMoE, masters; serving fixed              |   89.3 (86.5)  |   43.7 (100.0) | 288.9 (287.1) | 117.1 (172.3) |

What the numbers say, eight GPUs, MoE layer alone. Master weights cost the
baselines 1 to 3 ms a step (Megatron TE on SmartMoE: 117.3 without, 119.8
with), and the Transformer Engine experts save 7 to 20 ms in training on
every routing and nothing in serving, where the step is the exchange. The
corrected ratios: on SmartMoE's routing the example with ShedLB trains 1.5x
faster than FasterMoE and 2.3x faster than Megatron-Core, and serves 1.5x
faster than FasterMoE (3.6x was published; FasterMoE's shadow of the hottest
expert does in serving what ShedLB's copy does, and the harness had hidden it)
and 2.7x faster than Megatron-Core; the plain example serves 1.1x faster than
FasterMoE. On Mixtral's routing the plain example trains 1.5x faster than
Megatron-Core and 1.6x faster than FasterMoE, and in serving FasterMoE is now
the fastest system (105.1 against the example's 112 to 130 and 124.8 with a
copy, Megatron-Core's 131.9 and FastMoE's 129.9): its group-wise exchange
overlaps the 128 MB slabs that bound every other step there. On OLMoE's
routing the PyTorch reference (194.7 / 172.8), Megatron-Core (248.1 / 169.6)
and FasterMoE (serving 171.9) are ahead of the example (243.1 / 249.9): one
all-to-all beats one message per expert when the step is the top-8 exchange.
Capacity factor 1.25 changes nothing on any routing.

**The block rows with the fused attention** (job 23499744, gpub002 and
gpub026, the same twelve arms with the fused build `moe` and the unfused
build `moe.attn_unfused`). The example's attention sublayer under `-B` now
runs on cuDNN's fused attention graphs (the Makefile's `MOE_CUDNN_ATTN`,
default on where cuDNN 9 and its frontend are found; the fp32 path and the
HIP build stay unfused): the scores never touch memory, as in the flash
kernel PyTorch's attention dispatches to for the references. Every run
checks the fused path at startup against the unfused kernels, one training
step on the same weights and a random input, and aborts past a relative L2
error of 5e-2; measured here it is 3.6e-3 on the output and 1.1e-3 to 2.1e-3
on the two weight gradients, bf16 rounding, at both head sizes (64 and 128).
The "Block:" header line names the kernels. The fused and unfused builds'
checksums differ by that rounding carried through the run (the fifth digit
of `out2` after forty training steps), and within each build the plain and
ShedLB arms agree to the last digit, in training and in serving, on both
routings.

| transformer block, 8 GPUs, ms/step               | SmartMoE train | SmartMoE serve | Mixtral train | Mixtral serve |
|--------------------------------------------------|---------------:|---------------:|--------------:|--------------:|
| example plain, fused attention                   |           89.3 |           32.6 |         170.1 |         118.3 |
| example ShedLB, fused attention                  |           57.8 |           27.0 |         172.6 |         136.7 |
| example plain, unfused attention, same nodes     |          102.9 |           36.2 |         183.7 |         153.6 |
| Megatron-Core, TE experts, masters (above)       |          129.3 |           77.3 |         276.5 |         141.8 |
| FastMoE plain, masters (above)                   |          202.5 |           87.3 |         335.8 |         140.0 |
| FasterMoE, masters, serving fixed (above)        |           89.3 |           43.7 |         288.9 |         117.1 |

The attention sublayer's increment over the MoE-only step in training is
now 8.2 ms on SmartMoE's routing (21.6 unfused; Megatron-Core pays 9.5 and
FasterMoE 10.7) and 8.3 on Mixtral's (21.9 unfused; the references pay 31
to 33: the example's attention runs on a lane while the dispatcher waits on
the exchange, so on an exchange-bound step it fills idle GPU time rather
than adding to the step). In Mixtral serving the unfused attention had cost
35 ms on top of the exchange (153.6 against 118.3 on the same nodes): its
score traffic competed with the slab exchange for memory bandwidth. The block
ratios: on SmartMoE's routing the example with ShedLB trains 1.5x faster than
FasterMoE and 2.2x faster than Megatron-Core (the plain example is level
with FasterMoE), and serves 1.6x faster than FasterMoE and 2.9x faster than
Megatron-Core; on Mixtral's routing the plain example trains 1.6x faster
than Megatron-Core and 1.7x faster than FasterMoE, and serves level with
FasterMoE (118.3 against 117.1), 1.2x faster than Megatron-Core and FastMoE.

## Measured 2026-11-05: SmartMoE's online placement, one node

**Why this section.** Two systems decide expert placement online from the
routing, the way the balancer does, and neither had a row: FlexMoE (Nie et
al., SIGMOD'23) and SmartMoE (Zhai et al., ATC'23). FlexMoE has no public
code (the paper puts it inside the Hetu framework and gives no URL); its
scheduler expands a replica of the hottest expert and shrinks one of the
coldest when a cost model (profiled tokens per second, all-to-all bytes over
link bandwidth, gradient-sync bytes over bandwidth) predicts a shorter step,
replicas split tokens evenly and all-reduce their gradients, and the paper
reports 1.30x over FasterMoE on 32 to 64 A100s; that is the replication
decision ShedLB makes, taken by a training framework's own scheduler, and the
example's `-X` and the reference's `--replicate` are its static analogue.
SmartMoE's artifact (thu-pacman/SmartMoE-AE, commit e19b3a9) had been used
only as the source of the `smartmoe_L02` routing trace. Its online part is
small: `fmoe/smartmoe/mapping.py` re-places the experts every
`--dynamic-freq` (10) iterations from the gate counts it samples at the last
two iterations of each window, largest-first onto the least loaded GPU with
an equal expert count per GPU (a permutation, never a copy), an expert left
where it is when its GPU is among the least loaded, and the new mapping
taken only if it lowers the maximum GPU load; `update_expert_mapping` then
swaps the parameters with one blocking point-to-point send per expert (the
optimizer state and the master weights stay where they were). The artifact
runs on a 2022 FastMoE fork over torch 1.10 and CUDA 11.3 with a Megatron-LM
fork and Apex, scripted for sixteen V100s; running it as shipped would put
three years of FastMoE and PyTorch kernel differences into the comparison.
The policy is therefore ported rank for rank into the reference as `--place
smartmoe` (`ref/smartmoe_check.py` compares the port with the artifact's
function on 2800 random instances, ties included: no mismatch), on the same
stack as the FasterMoE arm, with the masters and moments moved for
correctness in one batched exchange, as under every `--lb sync` placement.
It differs from the reference's EPLB-style packing (`--place lpt`) in three
things: the load it reads (the mean of the step before the LB step and of the
previous LB step, against the LB step's own counts, which the reference knows
a step ahead), the tie rule (stay if among the least loaded, against the
relabelling that keeps the most experts in place) and the acceptance (only if
the maximum falls, against always).

Job 23567906, gpub071, one A40 node, four GPUs, the three arms in one
allocation, `-i 40 -u 3 --dtype bf16 --fuse`, LB every 10 steps from step 10,
steady state steps 23 to 43 (the placement arms' LB steps 30 and 40 included,
as in every earlier row). The static controls reproduce the 2026-11-01
four-GPU rows within 1% (105.9 / 52.8 and 108.6 / 68.8 there). The other
systems' rows are the one-node rows of the 2026-11-01 and 2026-11-03
sections, other nodes.

| four GPUs, ms/step                                   | SmartMoE train | SmartMoE serve | Mixtral train | Mixtral serve | OLMoE train | OLMoE serve |
|------------------------------------------------------|---------------:|---------------:|--------------:|--------------:|------------:|------------:|
| PyTorch reference, static                            |          105.8 |           53.3 |         292.3 |         145.9 |       193.1 |       170.5 |
| PyTorch reference, EPLB packing every 10 steps       |          107.8 |           69.6 |         264.3 |         130.7 |       197.2 |       177.4 |
| PyTorch reference, SmartMoE online placement         |          110.4 |           71.1 |         263.4 |         128.1 |       197.5 |       178.8 |
| example plain (2026-11-01 rows)                      |           55.4 |           32.9 |         137.8 |         108.7 |             |             |
| example ShedLB (2026-11-01 rows)                     |           50.4 |           27.3 |         137.3 |         100.7 |             |             |
| Megatron-Core, TE experts (2026-11-03 rows)          |           81.4 |           48.6 |         250.8 |         139.5 |       253.0 |       167.3 |
| FasterMoE, serving fixed (2026-11-03 rows)           |                |           27.0 |               |         132.4 |             |       184.3 |

| experts moved at LB steps 10 / 20 / 30 / 40          | SmartMoE routing | Mixtral routing | OLMoE routing |
|------------------------------------------------------|-----------------:|----------------:|--------------:|
| EPLB packing (of 32 / 8 / 64 experts)                | 18 / 16 / 16 / 12 |   4 / 2 / 3 / 4 | 43 / 41 / 41 / 42 |
| SmartMoE placement                                   | 20 / 19 / 18 / 16 |   6 / 3 / 0 / 2 | 48 / 46 / 48 / 42 |
| tokens per GPU, max over mean, after the placement   |     1.94 to 2.00 |    1.06 to 1.13 |  1.00 to 1.07 |

What the numbers say. SmartMoE's placement and EPLB's packing are the same
decision within 2.5% on every routing, training and serving: the same
objective reached from a one-step-older load, and the acceptance rule changed
one decision in twelve (Mixtral, step 30, kept). On SmartMoE's own routing
neither helps, because the hottest expert takes 48% of the tokens, 1.9x a
GPU's share with four GPUs, and no permutation of whole experts can lower a
GPU's load below its hottest expert; what the arms pay is the exchange, 16 to
20 experts of 201 MB each at every LB step (2.3 GB at 12 GB/s, about 200 ms,
21 ms per step averaged over the run, the `migration` row of the phase table),
which in serving is 40% of a 53 ms step. The example with ShedLB, whose
balancer copies the hot expert instead of moving it, trains 2.2x faster and
serves 2.6x faster than the reference under SmartMoE's policy (50.4 against
110.4, 27.3 against 71.1); FasterMoE's shadow, the other copying policy,
serves it in 27.0. On Mixtral's routing (eight experts, two per GPU, a 1.06
to 1.13 token imbalance) both placements gain 10 to 12% over static, the
7 to 10% the 2026-11-01 section measured for EPLB's packing, the exchange of
2 to 6 experts of 470 MB paid back by a better-packed pair. On OLMoE's routing
(sixty-four experts, top-8, a 1.00 imbalance at the block map) there is
nothing to gain and the exchange of 40 to 48 experts per LB step costs 2 to
5%. The 2023 paper's 1.88x over FasterMoE is an end-to-end figure on its own
workloads with its offline parallelization search included; the online
placement alone, which is what this arm isolates, cannot move a one-expert
hot spot, and that is the case the balancer's copy exists for.

## Measured 2026-11-05: GPU timelines of the example and FasterMoE, eight GPUs

Job 23581139 (gpub009 and gpub099, two A40 nodes), the SmartMoE routing,
MoE layer alone, the configuration of "Measured 2026-11-04": the example
with ShedLB (`-R 2 -f 20 -b 20 -a`) and FasterMoE (smart schedule and
shadowing, fp32 masters in training, the serving step without the per-step
collection), each in training and serving, each as an untraced control and
as a traced run in the same job. The example's timeline is the runtime's
`+gputrace` (CUPTI kernels, entry methods and idle time per PE), recorded
over the whole run, once kernels only (`+gputracenocopy`) and once with the
copy records; FasterMoE's is torch.profiler (`ref/fastmoe_moe.py --profile`)
over steps 30 to 41: the host ops, CUPTI kernels and copies. Steps 30 to 40
are compared, step 41 holding FasterMoE's trace export. The breakdown is
`ranklogs/kernelclass.py --steps 30:40` on each trace; the Perfetto files
and the raw traces are in `/scratch/mzu/bhosale/moe_runs/prof/tr_23581139/`
(`perfetto_example_shedlb_{train,serve}_steps30-40.json`,
`perfetto_fastermoe_{train,serve}_steps30-40.json`).

**What the tracing costs** (ms/step, steady steps 23-43 and steps 30-41;
the example's three runs of each mode have identical out2 and w2):

| arm                   | untraced control    | traced, kernels only | traced with copies |
|-----------------------|--------------------:|---------------------:|-------------------:|
| example ShedLB, train | 52.5 / 51.2         | 52.1 / 50.5          | 53.8 / 51.5        |
| example ShedLB, serve | 28.0 / 24.5         | 26.9 / 24.8          | 25.6 / 23.0        |

FasterMoE's controls in the job crashed on a harness import error (the
`contextlib` import sat inside the `--profile` branch; fixed) and ran again
in job 23581553; in the traced runs the profiler is inactive before step 28,
so the same run holds its own control:

| FasterMoE          | control (job 23581553) | traced run, untraced steps 23-27 | traced steps 30-40 | README 2026-11-04 |
|--------------------|-----------------------:|---------------------------------:|-------------------:|------------------:|
| train, masters     | 78.5                   | 81.5                             | 80.4               | 78.6              |
| serve, fixed       | 40.5                   | 44.4                             | 38.7               | 40.3              |

The runtime's trace changes nothing measurable; torch.profiler's traced
steps run as the untraced ones do (its export then costs 7 s on step 41 and
another 7 s on step 42 in training, 0.4 and 0.3 s in serving, outside the
window).

**The step by kernel class** (steps 30 to 40, ms per step, mean over the
eight ranks with the busiest rank in parentheses; concurrent kernels add,
so a class can exceed the step; GPU busy is the share of the step with at
least one kernel of the class running):

| per step                            | example ShedLB, train | FasterMoE, train | example ShedLB, serve | FasterMoE, serve |
|-------------------------------------|----------------------:|-----------------:|----------------------:|-----------------:|
| step, from the GPU timeline         | 49.0                  | 79.3             | 24.6                  | 38.6             |
| GEMM kernels                        | 20.9 (33.0)           | 19.5 (26.3)      | 7.3 (11.6)            | 7.4 (9.4)        |
| other kernels                       | 10.8 (25.2)           | 8.4 (9.7)        | 1.7 (2.4)             | 1.1 (1.4)        |
| NCCL kernels                        | 0                     | 61.3 (65.3)      | 0                     | 32.7 (33.0)      |
| of which the token exchange (SendRecv) |                    | 41.8             |                       | 18.1             |
| of which the shadow broadcast       |                       | 14.9             |                       | 14.6             |
| of which the shadow's gradient reduce |                     | 4.4              |                       | 0                |
| copies (the example: its copies run) | 1.0 (1.6)            | 2.9 (3.5)        | 0.9 (2.1)             | 1.1 (1.5)        |
| GPU busy, kernels other than NCCL   | 55% (79%)             | 35% (45%)        | 29% (50%)             | 22% (27%)        |
| GPU busy, all kernels               | 55%                   | 96%              | 29%                   | 95%              |

Per rank, training (ms per step; the example's busy share counts all its
kernels, FasterMoE's the kernels other than NCCL's):

| rank | example GEMM | example other | example busy % | FasterMoE GEMM | FasterMoE other | FasterMoE NCCL | FasterMoE busy % |
|-----:|-------------:|--------------:|---------------:|---------------:|----------------:|---------------:|-----------------:|
| 0    | 13.4 | 9.1  | 37 | 19.5 | 8.2 | 64.8 | 35 |
| 1    | 33.0 | 7.3  | 76 | 13.1 | 8.8 | 63.6 | 28 |
| 2    | 14.2 | 8.0  | 40 | 21.7 | 8.1 | 63.3 | 38 |
| 3    | 29.4 | 4.5  | 69 | 11.6 | 6.8 | 65.3 | 23 |
| 4    | 24.8 | 25.2 | 79 | 18.2 | 8.6 | 60.2 | 34 |
| 5    | 22.1 | 12.1 | 56 | 26.3 | 9.7 | 56.6 | 45 |
| 6    | 17.4 | 10.7 | 45 | 23.9 | 8.9 | 56.1 | 41 |
| 7    | 13.1 | 9.2  | 36 | 22.0 | 8.0 | 60.5 | 38 |

Per rank, serving:

| rank | example GEMM | example other | example busy % | FasterMoE GEMM | FasterMoE other | FasterMoE NCCL | FasterMoE busy % |
|-----:|-------------:|--------------:|---------------:|---------------:|----------------:|---------------:|-----------------:|
| 0    | 4.8  | 1.4 | 19 | 7.3 | 1.0 | 32.7 | 22 |
| 1    | 10.4 | 1.5 | 47 | 5.0 | 1.3 | 32.2 | 16 |
| 2    | 4.4  | 1.2 | 21 | 8.9 | 1.0 | 32.7 | 26 |
| 3    | 11.6 | 2.0 | 50 | 4.2 | 0.7 | 32.9 | 13 |
| 4    | 7.9  | 2.4 | 28 | 7.1 | 0.9 | 32.8 | 21 |
| 5    | 8.7  | 2.1 | 33 | 9.1 | 1.4 | 32.7 | 27 |
| 6    | 6.2  | 1.4 | 21 | 9.4 | 1.1 | 32.5 | 27 |
| 7    | 4.4  | 1.3 | 17 | 8.0 | 1.1 | 33.0 | 24 |

What the timelines say:

- **The expert compute is the same in both systems.** Per GPU and step the
  GEMM kernels take 20.9 ms in the example and 19.5 in FasterMoE in
  training, 7.3 and 7.4 in serving: the same tokens through the same
  cuBLAS bf16 tensor-core kernels. The 1.6x and 1.5x between the steps is
  entirely in what surrounds the compute.
- **FasterMoE pays the shadow every step.** Its NCCL kernels are resident
  61 ms of the 79 ms training step: 42 ms the group-wise token exchange
  (`ncclDevKernel_SendRecv`, overlapped with the experts as the smart
  schedule intends), 15 ms the broadcast of the shadowed experts' weights
  (1.3 experts of 67 MB each step at the calibrated 5.35 GB/s) and 4 ms the
  reduce of their gradients. In serving, 18 ms of exchange and the same 15
  ms of broadcast fill 33 of the 39 ms. The shadow is transient, so the hot
  expert's weights cross the network again every step; the example's copy
  is made once, at the balancer's step, and the step carries only the
  tokens.
- **NCCL's kernels hold the SMs while they wait.** FasterMoE's GPUs read
  95 to 96% busy over all kernels and 22 to 35% over the kernels that
  compute; the example's exchange is RDMA and copy-engine traffic that
  occupies no SM, so its 55% (training) and 29% (serving) busy shares are
  compute. Of the example's training step, 27 ms of kernels on the busiest
  GPU sit in a 49 ms step: the remaining 22 ms are the exchange and the
  host-side per-message cost described under "Tracing" below.
- **Where the imbalance sits.** The example's GEMM time per rank runs from
  13 to 33 ms in training (ShedLB copied expert 5 from PE 1 to PE 3 at
  step 20, each copy serving half the dispatchers, and packed ten light
  experts onto PE 4; rank 4's 25 ms of other kernels is their update
  passes, `addIntoKernel` and `applySgdKernel` at 2.1 ms per expert and
  step whatever its tokens, the price of the bit-identical update scheme),
  and the step is the busiest rank's compute plus the exchange. FasterMoE's shadow spreads the hot expert over all
  ranks (12 to 26 ms of GEMM per rank), but the step is set by the
  exchange and the broadcast, not by any rank's compute.


## Measured 2026-11-05, later: every copy in one round, and what more copies cost

The timelines above showed the training step set by the two copy holders
(33.0 and 29.4 ms of GEMM against a 20.9 ms mean) with the light ranks
idle for half the step. The balancer had reached its floor: one copy per
balancing round, so a run with `-f 20` stops at two copies of the hot
expert, each with half of its 47.9% of the tokens. ShedLB now plans every
copy a group needs in one round (`src/ck-ldb/ShedLB.C`, the group pass):
each count from the present one to the need is planned in turn and the
count that leaves the fullest bin lowest is taken, if it passes the
critical-path test; a count that gains nothing on the way is passed over.
`CentralLB` keeps only the contiguous prefix of a group's new ranks when
the memory verifier refuses one, and an expert here takes several clones in
one round (`decided_new`, `n_clone`), announcing the final count. Validated
in job 23581818 (gpub035 and gpub091): `replica_test` passes, and the
`-R 4` runs' out2 and w2 equal the two-copy controls of job 23581139 bit
for bit, through three and four copies.

| SmartMoE routing, 8 GPUs, ms/step   | copies from step 21 | steady 23-43 | steps 30-41 | tokens/PE max/avg, step 30 | out2 / w2 |
|-------------------------------------|--------------------:|-------------:|------------:|---------------------------:|-----------|
| ShedLB `-R 2`, train (job 23581139) | 2                   | 52.5         | 51.2        | 2.00                       | 5.148446218722643e+06 / 1.091916901199783e+05 |
| ShedLB `-R 4`, train                | 3 (the 4th declined; 4 from step 41) | 105.5 | 107.3 | 1.95                 | identical |
| ShedLB `-R 2`, serve (job 23581139) | 2                   | 28.0         | 24.5        | 1.96                       | 7.433357930553174e+06 / 1.092300638431098e+05 |
| ShedLB `-R 4`, serve                | 4                   | 31.8         | 25.4        | 1.88                       | identical |

More copies made the training step twice as long and the serving step no
shorter, for two reasons the traces (`tr_23581818/ex_r4_{train,serve}_k`)
make plain:

- **Three copies share a four-partition expert as 1/4, 1/4 and 1/2.** The
  dispatcher partitions are fixed at four (`K_PART`) and a copy serves a
  contiguous block of them, so the third copy's holder, PE 2, still ran 31.5
  ms of GEMM against 15.0 and 16.4 on the other two. The balancer had priced
  every copy at a third of the load and believed three copies left the
  fullest bin at 1.026 of 1.266; it therefore also declined the fourth copy,
  which under its model would have gained nothing more. The hot expert's
  tokens themselves are even across the eight dispatchers in this trace
  (12.5% each, every step), so with four copies each holder carries a quarter.
- **The copies' exchange grows with their number and does not overlap.**
  Each copy sends its fp32 block sum, two buffers of 67 MB, to every other
  copy after its compute and waits for theirs before its update; the three
  copies sat on one node, and their PEs were idle 95 of every 100 ms. The
  start-up probe prices one partial alone at 5.5 ms; two partials per copy
  with the receives serialised on the comm stream and the fabric shared cost
  about 50 ms a step. With four copies in serving, where nothing is
  exchanged, the copy holders fell to 5 to 10 ms of GEMM, the fullest GPU
  was a non-holder (PE 5, 11.4 ms), and the step stayed at 25 ms: the
  serving step is bound by the token exchange and its per-message host
  cost, not by the compute (the busiest GPU is 43% busy).

Two corrections followed. The declaration carries the partition count
(`setLBReplica(..., parts)`, `LDObjData::replicaParts`, LB format 10), and
the balancer prices copy `r` of `k` with its block's share
(`LDObjData::replicaShare`), so three copies of a four-partition expert are
seen for what they are and the count search can pass over them. And a copy
that has exchanged declares what its exchange cost here, its wait per step
and peer, in place of the probe's figure (`Expert::xchg_wait_s`); until it
has, the probe stands. Job 23582210 (gpub009 and gpub099, the controls'
node pair) measured the balancer's choice under both; `replica_test` passes
on the new declaration, and every checksum is the controls' again:

| SmartMoE routing, 8 GPUs, ms/step        | copies from step 21                          | steady 23-43 | steps 30-41 | tokens/PE max/avg, step 30 |
|------------------------------------------|----------------------------------------------|-------------:|------------:|---------------------------:|
| ShedLB `-R 2`, train, reference          | 2                                            | 51.6         | 47.9        | 1.96                       |
| ShedLB `-R 4`, train                     | 4 at step 20 (three passed over); one retired at step 40 on the measured cost | 356.7 | 405.2 | 1.49 |
| ShedLB `-R 4`, serve                     | 4                                            | 31.5         | 26.9        | 1.88                       |

At step 20 the balancer planned four copies in one round, priced with the
probe's 5.5 ms a partial. With four copies each one sends three block sums
and receives three, twelve transfers of 134 MB a step through one node's
PCIe fabric: the copies measured 292 to 338 ms of exchange wait a step
(97 to 112 ms per peer declared), and at step 40 the balancer, now pricing
with that, retired a copy ("the exchange costs more than the sharing
saves"). In serving, where the copies exchange nothing, four copies cost
nothing and gained nothing: the step is the token exchange's.

What this settles. On PCIe-attached A40s the fp32 block-sum exchange, 134
MB per peer and step at this expert shape, bounds replication in training
at two copies: one peer costs about 10 ms a step and overlaps in part, two
cost 50, three cost 300. The balancer now learns that cost from the copies
after their first interval and retires what does not pay, one copy per
round; in the first round it prices with the probe's one-partial figure,
which is optimistic whenever the application allows more than two copies
(`-R 4`), so a run that allows them pays for one interval of them. The
standard configuration (`-R 2`) is unchanged by all of this: the same
copy, the same step (51.6 against 52.5). The imbalance the timelines show,
two copy holders at 1.6 times the mean GEMM with the light ranks idle, is
the price of not exchanging more on this fabric. What would move it:
inter-GPU bandwidth (NVLink, or the MI250X fabric on Frontier, where the
measured cost would let the balancer keep more copies); bf16 block sums,
halving the bytes but rounding where the fp32 plain run does not; or a
reduce-scatter and all-gather among the copies in row slices pipelined
under the compute, which cuts a copy's traffic from `k - 1` to `2 (k - 1) / k`
block sums and hides part of it, but still puts 800 MB a step on the node
at four copies, near the compute it would hide under. None of these is in.

## Measured 2026-11-05, evening: the balancer chooses the copy count, priced from the calibration

`-R` is gone. The example had named the copies the balancer might keep and
priced their exchange with a start-up probe of its own, one partial sent
alone; neither is the application's to decide. Now each expert declares
what it can keep consistent (four copies, the four-partition tree's
limit), what a copy sends each peer per balancing interval (two fp32
block sums a step: `replicaExchangeBytes`, `replicaExchangeMsgs`), what
each further peer costs it in device memory (`replicaExtraBytesPerPeer`,
the landing pairs, allocated as copies appear rather than for the most
there could be), and, once a copy has exchanged, what that exchange waited
per step and peer. ShedLB prices the exchange of a plan with `k` copies
from the runtime's calibrated transfer-cost table at that fan: `lbcalib`
gained a fan sweep (`-F 4 -X 128`, job 23583472, gpub057 and gpub100:
`k` PEs, one per device, each sending 4 to 128 MB buffers to every other
at the same time, and two PEs across the hosts), fitted per fan as what
one peer costs a copy, built in as the `xchg_fan<k>` entries of the
default table (`+LBCostConfig` overrides it). A copy's own measurement
sets the level at its present fan and the table's ratio carries it to
another.

| one peer, buffers of 4 to 128 MB            | alpha, per message | beta, per byte | rate       |
|---------------------------------------------|-------------------:|---------------:|-----------:|
| fan 2, one host                             | 51 us              | 5.46e-11 s     | 18.3 GB/s  |
| fan 3, one host                             | 1.9 ms             | 6.82e-10 s     | 1.5 GB/s   |
| fan 4, one host                             | 1.9 ms             | 1.08e-9 s      | 0.9 GB/s   |
| fan 2, across hosts                         | 51 us              | 7.02e-11 s     | 14.2 GB/s  |
| one transfer alone (the tier table)         | 29 us              | 3.98e-11 s     | 25 GB/s    |

Two copies exchange at the bus rate; three or four at a tenth of it,
which is not the PCIe fabric's limit but the runtime's concurrent IPC
receive path, open as a runtime item. The table says what the copies had
found: a 134 MB block sum costs a peer 7 ms at two copies and 150 at four.

Job 23583534 (gpub065 and gpub100), the matrix's ShedLB arms on the three
routings with the balancer choosing, `-f 20 -b 20 -a` and nothing else;
`replica_test` passes, and every SmartMoE checksum is the plain arm's:

| 8 GPUs, ms/step                | the balancer's decision                                   | steady 23-43 | steps 30-41 | before (`-R 2`, 2026-11-04) |
|--------------------------------|-----------------------------------------------------------|-------------:|------------:|----------------------------:|
| SmartMoE plain, train / serve  | (no balancing step)                                       | 81.6 / 35.1  | 81.3 / 35.8 | 81.1 / 35.3                 |
| SmartMoE ShedLB, train         | 2 copies at step 20; 4 at step 40 (see below)             | 54.0         | 45.8        | 52.1                        |
| SmartMoE ShedLB, serve         | 4 copies at step 20, exchanging nothing                   | 31.3         | 26.1        | 27.4                        |
| Mixtral ShedLB, train / serve  | no copy: "no count up to 2 lowers the fullest bin enough" | 162.4 / 120.9 | 165.9 / 112.1 | 164.2 / 124.8            |
| OLMoE ShedLB, train / serve    | no copy                                                   | 246.2 / 265.6 | 233.2 / 245.8 | 240.8 / (plain 249.9)    |

The first round priced four copies of the SmartMoE expert at 150 ms a peer
and kept two, as it should. At step 40 the two copies had measured their
exchange (6.8 and 11.8 ms per step and peer, the table's fan-2 figure) and
the balancer, which then took a measured cost as the cost at every fan,
planned four copies for the last three steps. The fix: a measurement
applies at the fan it was taken at and is carried to another by the
table's ratio. Job 23583704 (gpub065 and gpub099) re-ran the arm with it:
two copies at step 20, and at step 40 "no count up to 4 lowers the fullest
bin enough (3 copies: 1.696 of 1.322): no copy", so two copies throughout;
52.9 ms/step over steps 23-43 and 51.9 over 30-41, against the forced
two-copy 52.5 and 51.2 of the morning's control, checksums the same. The
copies' measured exchange that round: 15.4 and 3.9 ms per step and peer.

What this settles. The application no longer names a copy count or prices
its exchange; it declares its capability and its bytes, and measures what
it pays. The balancer finds, from the calibration, that two copies of the
SmartMoE expert are what this fabric affords in training and four what
serving affords, and that Mixtral and OLMoE afford none, without being
told; the rows are those of the forced `-R 2` runs where those were right
and better where they were not (Mixtral serving: 120.9 against 124.8 with
the forced copy). What remains is the runtime's own: the concurrent IPC
receive path that makes a third copy's exchange ten times slower than a
second's, which the calibration now exposes as a number.

## Measured 2026-11-06: two A100 nodes, NVLink inside the node

The question behind this section: how much of what the A40 rows show is the
PCIe fabric's doing. Two `gpuA100x4-interactive` nodes (gpua048 and gpua053:
four A100-SXM4-40GB each, every pair on four NVLink lanes, Slingshot between
the nodes, the same four 16-core NUMA nodes as the A40 nodes with GPU L on
node 3-L), the SmartMoE routing, the MoE layer alone, eight GPUs, the shapes
and options of "Measured 2026-11-04". Job 23601306 ran the whole matrix in
7 min 22 s of node time; job 23608560 validated the built-in table and
recorded the copy records (below).

What it took to run there: device code for sm_80 as well as sm_86 (the
Makefile's `CUDA_ARCHS ?= 80 86`; one binary serves both node types, the
A40 rows are reproducible from the same file), FastMoE's extension rebuilt
for `8.0;8.6` by `ref/build_fastmoe.sh` (it was sm_86 only; Transformer
Engine's wheel already ships sm_80 code), and the launch wrappers reading
each GPU's NUMA node and core range from sysfs (`ranklogs/gpunuma.sh`; the
first line of every rank log now says which GPU, node and cores the rank
got) instead of assuming the A40 layout. The machine was calibrated first:
`lbcalib -i 200 -w 20 -F 4 -X 128 -N 10` into `lbcalib/lbcost.a100.2n.fan.conf`,
and FasterMoE's own parameters (`ranklogs/fm_calib.a100.txt`: 15.4 GB/s
broadcast, 274 bf16 TFLOP/s, against 5.35 GB/s and 121 on the A40 nodes; the
A40 file is kept as `fm_calib.a40.txt`). In job 23601306 the ShedLB arms
priced from the new table through `+LBCostConfig`; the runtime now carries
it as a second built-in table, chosen by the name of the device the PE drives
(`DiffusionCostConfig::loadBuiltin`: an A100 gets the A100 table, anything
else the A40 one, the chosen table and device named in the start-up line),
so no option is needed, as on the A40 nodes.

### The calibration: the fan-in collapse is the PCIe fabric's

| one peer, buffers of 4 to 128 MB   | A40 nodes (PCIe) | A100 nodes (NVLink) |
|------------------------------------|-----------------:|--------------------:|
| 2 copies, one host                 | 18.3 GB/s        | 89.2 GB/s           |
| 3 copies, one host                 | 1.5 GB/s         | 75.2 GB/s           |
| 4 copies, one host                 | 0.9 GB/s         | 90.1 GB/s           |
| 2 copies, across hosts (Slingshot) | 14.2 GB/s        | 11.4 GB/s           |
| one transfer alone, cross-GPU tier | 25 GB/s          | 106 GB/s            |
| migration, device bytes            | 23 GB/s          | 84 GB/s             |

Three and four copies exchanging at once keep their rate on NVLink. The
collapse of "Measured 2026-11-05, evening" is therefore the PCIe fabric's
concurrent peer reads, not the runtime's receive path as such; the open
item is narrowed to that fabric. On this table a 134 MB block sum costs a
copy 1.5 ms a peer at any fan, so the balancer made four copies at once.

### The matrix

ms/step, steps 30 to 41 (steady 23 to 43 in parentheses), eight GPUs:

| MoE layer alone, SmartMoE routing          | A100 train   | A100 serve   | A40 train | A40 serve |
|--------------------------------------------|-------------:|-------------:|----------:|----------:|
| example plain                              | 44.3 (43.9)  | 22.8 (22.1)  | 81.1      | 35.3      |
| example ShedLB (4 copies at step 20)       | 45.5 (50.3)  | 16.4 (19.4)  | 52.1      | 27.4      |
| FasterMoE (schedule + shadows, masters)    | 31.7 (31.7)  | 16.2 (16.0)  | 78.6      | 40.3      |
| Megatron-Core, TE grouped experts, masters | 43.8 (44.6)  | 21.7 (21.9)  | 119.8     | 74.4      |

The balancer's decisions (both arms, the first round at step 20, priced from
the A100 table): "replicate group 5: load 0.937 exceeds the per-bin target
0.268; copy 1 from PE 1 to PE 3, copy 2 to PE 2, copy 3 to PE 0, 4 copies in
all; the fullest bin 0.961 -> 0.305" in training, the same shape in serving
(0.295 against 0.089, the fullest bin 0.305 -> 0.098). Every ShedLB checksum
equals plain's (5.148460962447461e+06 / 1.091916901522711e+05 training,
7.433384418063989e+06 / 1.092300638431098e+05 serving).

What the A100 rows say about the claims: on these nodes the example does not
beat FasterMoE in training (FasterMoE is 1.4x faster: 31.7 against 44 to 45),
ties it in serving (16.2 against 16.4), ties Megatron-Core in training (43.8
against 44.3 plain, 45.5 ShedLB) and is 1.3x faster than it in serving (16.4
against 21.7). The 1.5x over FasterMoE and 2.3x over Megatron-Core of
"Measured 2026-11-04" are measured on PCIe-attached A40s, where the baselines'
NCCL exchange and FasterMoE's per-step shadow broadcast are bound by the
fabric; on NVLink those costs fall by 3 to 4x while the example's step is
bound by something the fabric does not change (next two subsections). The
copy decision itself paid in serving (22.8 -> 16.4, as the 35.3 -> 27.4 of the
A40 nodes) and not in training.

### Training: four copies that cost nothing by the table and nothing gained by the step

The traced training arm (same job, `+gputrace +gputracenocopy`; read with
`ranklogs/stepchain.py --group 5`) over steps 31 to 40, the four copies of
expert 5 on PEs 0 to 3:

| the hot copies' step, A100, training       | ms        |
|--------------------------------------------|----------:|
| compute (GEMM on the copy's lane)          | 6.6 to 7.3 |
| the four copies' step starts, spread       | 13 to 20 |
| last partial in, after the latest copy's computeDone | 20 to 23 |
| applyDone after the last partial (copy 5.3, step 35) | 20  |
| a hot copy's period                        | 38 to 44 |
| a dispatcher's period, waiting for those outputs | 42 to 46 |
| exchange wait the copies measured, per peer and step | 8.5 to 11.4 |
| the table's price per peer                 | 1.5      |

The copies serve different dispatcher partitions, and the dispatchers' step
starts are spread by up to 32 ms, so the copies start their steps 13 to 20
ms apart; each copy's apply waits for the latest copy's block sum, and that
last partial lands 20 to 23 ms after the latest copy finished its compute
(copy 5.0's partial reached copy 5.2 within 4.4 ms and copies 5.1 and 5.3
some 20 ms later); the apply then queues on a stream behind other experts'
chunks (copy 5.3 at step 35: every partial in by 34.6 ms, applyDone at 54.7).
Seven ms of compute thus become a 40 ms period, and the step is plain's 44
with a 29 to 63 ms scatter where plain holds 44. The exchange wait the copies
report (8.5 to 11.4 ms a peer) is mostly waiting for each other and for their
streams, which the table's transfer price (1.5 ms a peer) does not contain:
on the A40 nodes at two copies the 7 ms a peer matched the table because the
transfer was the larger term. This is the same gap as the serving "free
copies" of 2026-11-05, now in training: a decision right by its model and
wrong by the step, which only measuring the step after the round can catch.

### Serving: the step is the put's completion time on either fabric

The traced serving arm, steps 31 to 40 (four copies of expert 5):

| the serving chain, A100 against A40        | A100     | A40 (2026-11-05) |
|--------------------------------------------|---------:|-----------------:|
| hot copy's compute                         | 2.4 to 3.3 | 12.0           |
| its output puts complete, after computeDone | 13 to 15 | 12.7            |
| a dispatcher's 64 token puts complete, after its step start | 7 to 12 | 4 to 15 |
| the step                                   | 16       | 25               |
| the busiest GPU, busy                      | 5 to 7   | 12               |

The compute fell four to five times with NVLink and the faster GEMMs; the
time a copy's two 16 MB output puts take to complete after its compute did
not move, and neither did the dispatchers' token puts. A 16 MB device put
completing 13 to 15 ms after it is issued, on NVLink as on PCIe, is the
runtime's put path (issue, the landing on the receiver's comm stream, the
completion notice), not the fabric, and it is what bounds the serving step on
both machines. That is the open runtime item of these sections, sharpened.
The same-node put is an IPC write: the channel maps the receiver's buffer
once per device and writes into it from the producing stream, the puts of
one entry method gathered into one `cudaMemcpyBatchAsync`
(`ckrdmadevice.C`, `deviceChannelBatchAdd` / `deviceChannelBatchFlush`);
across hosts it is an RDMA write with the note as immediate data. CUPTI's
memcpy activity does not record batched copies, which is why the A40 trace
with copy records (job 23581139, `stepchain.py --copies 4`) shows on the hot
copy's PE only the local segment copies (ten of 16 MB a step at 260 to 290
GB/s) and nothing landing on a dispatcher's PE: the bytes move, unrecorded.
Everything after the copy is local to the host: the batch's completion is a
callback on the sender's own stream, whose handler marks every put's source
done for the sender and writes the receiver PE's note into its shared-memory
ring (`deviceChannelShmNoteFn`), which the receiver's progress loop polls
and turns into the landed entry. So the 13 to 15 ms between a copy's
compute and its puts' completion sit in: the flush (at the end of the entry
method that made the puts, or the scheduler loop's poll), the submitted
copy's launch and its run behind whatever the producing stream holds, the
sender's host noticing the stream callback, and the receiver's ring poll
and delivery. Job 23608560 runs the traced arms with the example's flight
log (`-F`, or `MOE_FLIGHT_LOG=1` through `ranklogs/env.extra`: per output
message the send, envelope and consumed times) and the runtime's zero-copy
statistics (`CHARM_ZC_STATS=1`: puts by tier, the host time of each put,
and the new `[chan-done]` line: each same-host batch's time from its
submission to its completion handler, mean and maximum), which place the
time among those four.

## Measured 2026-11-06, afternoon: the outputs go back through channels

The flight log of the morning's runs (`-F`, or `MOE_FLIGHT_LOG=1`; read with
`ranklogs/flights.py`) put the serving floor on the receiving side: on the A40
nodes a same-host 16 MB output reached the dispatcher's host 0.04 ms after it
was sent and was consumed 16 ms later (24 at the 90th percentile); the small
outputs waited 24 ms; the runtime's own counter gave an IPC receive 1.8 ms
from post to completion, 64 of them a dispatcher a step, one landing copy and
one post entry and one acknowledgement each. The token slabs, which go the
other way through device channels, completed their batches in 0.6 ms. Two
other candidates were measured out first, in job 23609803 (two A40 nodes,
2 min 40 s): a second build of the same source (meant as the pinned-flag
completion path against host-function callbacks; the example's Makefile
carried a dead `-DHAPI_CUDA_CALLBACK`, which the hybridAPI library reads
when it is compiled, not an application, and the library is built without
it, so both builds polled pinned flags, as every row in this file did) and
the process layout (one process per node driving four GPUs, a dispatcher
per GPU):

| two A40 nodes, ms/step, steps 30 to 41 | plain serve | ShedLB serve | ShedLB train |
|----------------------------------------|------------:|-------------:|-------------:|
| the build of every row so far          | 35.5        | 23.2         | 50.1         |
| the same source built again            | 35.6        | 26.8         | 52.9         |
| one process per node                   | 34.3        | 25.6         | 48.5         |

The repeat shows the run-to-run band (ShedLB serving 23 to 27), the layout
moves nothing. So the outputs now return the way the tokens go
(the default; `MOE_OUT_ENTRY=1` restores the entry-method sends): the
dispatcher owns a device channel per (expert, copy rank, step parity) over
the expert's fixed segment of its receive slab (`cap_src` rows per expert,
so the slab is `n_experts * cap_src` rows instead of the tokens'), hands
them to the copy that serves it (`takeOutChannels`, at start-up and when the
serving copy changes), takes each put's note (`outputArrived`: the tag
carries the step and the copies announced for the next one) and releases the
segments once the step is combined. The copy puts its output into the
channel as each dispatcher's chunk completes, each put its own batch: the
runtime gathers an entry method's puts into one copy flushed when the method
ends, which held every dispatcher's output until the run's last chunk and
took ShedLB training from 50 to 63 ms a step (job 23612036); with
`CkRdmaDeviceChannelBatchFlush()` after each put the streaming is back. A
retiring copy destroys its holder copies and tells the dispatchers, which
close theirs; a migrating copy carries its holder copies and re-resolves
them. The checksums are the entry path's bit for bit.

| two A40 nodes, ms/step, steps 30 to 41 | plain serve | plain train | ShedLB serve | ShedLB train |
|----------------------------------------|------------:|------------:|-------------:|-------------:|
| entry-method sends                     | 35.5        | 81.4        | 23.2         | 50.1         |
| output channels, batched per entry     | 31.1        | 81.7        | 21.6         | 63.3         |
| output channels, one batch per put     | 31.3        | 81.4        | 21.7         | 52.1         |

The chain of plain serving afterwards (job 23612304, `stepchain.py`): the
hot expert's 27.5 ms of GEMM on one GPU is the step; the dispatchers' host
time is 0.3 ms a step (it was 4 to 9). With four copies in serving the step
is 21 ms where the copies compute 5 to 6 ms each: the copies' wall spans are
13 to 15 ms on PEs holding up to nine experts on four lanes with 8 ms of
launch host time a step, and one copy's slabs arrive 20 ms after the
others' (its dispatchers' token puts complete 13 ms into the step on the PEs
whose copy engines also carry a copy's outputs).

The block-sum exchange of training went the same way (job 23612625, the
same nodes, 1 min 50 s): a copy owns a channel per (peer slot, step parity)
over its landing pair, which is now one contiguous allocation of the two
blocks, hands it to that peer (`takeGradChannels`, when the group forms or
grows), and puts its own pair into the peers' channels as one put each,
its own batch; the peer's figures (tokens, PE, device time, retiring) go as
the `peerStats` message serving already used; the landed pairs are released
after the apply. A retiring copy drops its holder copies of the survivors'
channels and the survivors drop theirs of its, each telling the other
(`gradChannelsDropped`), before the channels close and the copy goes; a
copy left alone frees its pairs once no peer holds a channel over them.

| two A40 nodes, ms/step, steps 30 to 41          | ShedLB train | exchange wait, copies 0 and 1 |
|-------------------------------------------------|-------------:|------------------------------:|
| entry-method sends (morning)                    | 50.1         | 12.9 / 6.9 ms a peer          |
| outputs by channel                              | 52.1         | 17.5 / 5.3                    |
| outputs and block sums by channel               | 46.1         | 11.3 / 3.2                    |

Plain training stays at 81.1 and every checksum equals the entry path's.
The drifting synthetic router (`-z 1.3 -p 10`, rounds every ten steps) made
five copies and retired three in one run through both channel kinds, and
its checksums equal its own plain run's. The traced chain of training with
two copies: each copy carries 33 ms of GEMM and the exchange adds 8; the
step is the copies' compute, and more copies would pay here, which the
balancer does not plan because the table prices three and four copies from
the morning's fan sweep, measured on entry-method sends; the channel path's
fan-in is a calibration to take (below).

### The A100 nodes with the channels (job 23612682, 4 min 40 s)

| two A100 nodes, ms/step, steps 30 to 41 | plain train | plain serve | ShedLB train | ShedLB serve |
|-----------------------------------------|------------:|------------:|-------------:|-------------:|
| this morning (entry-method sends)       | 44.3        | 22.8        | 45.5         | 16.4         |
| outputs and block sums by channel       | 40.8        | 25.6        | 27.2         | 23.0         |
| FasterMoE (this morning)                | 31.7        | 16.0        |              |              |

Training with four copies went from 45.5 to 27.2 ms a step, 1.17x faster
than FasterMoE here (30.8 against 31.7 over steps 23 to 43, the transition
included); the copies' exchange wait fell from 8.5 to 11.9 ms a peer to 1.2
to 5.1. The traced chain: 7 ms of compute per copy, then the all-pairs
exchange of 268 MB pairs (each copy sends three and receives three) and
the apply, 15 ms together. Serving went the other way, 16.4 to 23.0, and
the trace says why: the balancer, binning by device time, put ten experts
on PE 3 (four to six elsewhere), whose host then spent 10.4 ms a step in
their entry methods (0.5 to 2.4 ms per expert and step for the chunk
launches and the output puts) and was idle 4.5 ms of 22. On the entry path
that PE's host was busy the same way, but every dispatcher's receive path
was slower still, so it did not show. The balancer now treats the host as a
second resource (ShedLB: no move or copy may raise a PE's host time above
the device target, or above the hosts' mean with the same slack). With it
(job 23613467) no PE's host passes 6 ms a step, and the rows are plain
40.8 / 25.2, ShedLB 27.3 / 22.2: training reproduced, serving one ms
better, so the host was a part of it and not the whole. What the serving
chain shows next is the dispatchers' token puts: after the gather at 0.2
ms, the first of a dispatcher's 32 puts completes 4 to 6 ms later in most
steps and the last 8 to 18 ms in, on NVLink, for 16 same-host puts of 20
MB in one batch and 16 writes across the hosts; the per-put flight records
(`[FLIGHT] T`, `flights.py`) and a one-batch-per-put switch
(`MOE_TOK_FLUSH=1`) are the next measurement.

### The per-put records on the A40 nodes (job 23615034, 7 November, 00:12)

Plain serving four ways, the same build, with every token put's issue and
completion recorded (`[FLIGHT] T`), steps 30 to 41:

| two A40 nodes, plain serve                  | ms/step | puts >= 8 MB, done - issue, median / p90 / max | 1 to 8 MB | < 1 MB | last put of a dispatcher's step, median / max |
|---------------------------------------------|--------:|-----------------------------------------------:|----------:|-------:|----------------------------------------------:|
| batched per entry (the default)             | 31.2    | 6.5 / 23.0 / 23.9 ms                           | 0.6 / 4.8 | 0.4 / 4.0 | 6.6 / 23.9 |
| one batch per put (`MOE_TOK_FLUSH=1`)       | 31.5    | same band                                      |           |        | 6.9 / 24.0 |
| the same source built again (`moe.poll`)    | 31.4    | same band                                      |           |        | 6.8 / 24.2 |
| completion by event query (`+gpueventquery`) | 31.1   | same band                                      |           |        | 6.6 / 24.0 |

The four tie, so neither the batching nor the completion signal (the
runtime's pinned flag, written by a memset in stream order, against an
event record and query) is the latency. The records say what is: the hot
expert's 15.7 MB slabs, one from each of the eight dispatchers in the same
step. From the four dispatchers on the expert's own host they complete one
after another:

| expert 5's token put, same host, by source PE | 0    | 1    | 2    | 3   | other host: 4 | 5   | 6   | 7   |
|-----------------------------------------------|-----:|-----:|-----:|----:|--------------:|----:|----:|----:|
| done - issue, ms (step 30; every step alike)  | 23.6 | 16.1 | 9.6  | 3.6 | 3.5           | 3.0 | 5.3 | 6.9 |

On one host's clock the four complete at the same instant (23.4 to 23.6
ms after the host's first put of the step) from issue times 0, 5.5, 13.9
and 19.8 ms, while every other put of the same dispatchers completes
within 0.3 ms of its issue. They wait for the channel's credit: the expert
returns a slab's credit when the step's kernels that read it are complete
(`sendOutputs`), and expert 5 holds 55% of the tokens, about 20 ms of GEMM
a step on an A40. Plain serving is bound by the hot expert's compute, the
floor the copies remove, and the dispatchers' steps drift apart by up to
20 ms behind it. The fabric is not in it. The calibration by channel puts
(job 23626734, after a fix to lbcalib's own round counting, see below)
says the same: on the A40 nodes the same-host fan-in holds at 24.1 GB/s a
peer for two, 16.1 for three and 13.9 for four, against 18.3 / 1.5 / 1.0
by entry-method sends. The collapse of the afternoon's table was the
entry-method receive path (a landing copy, a post entry and an ack per
message), which the channels do not have:

| two A40 nodes, per peer, GB/s (alpha)      | fan 2, one host | fan 3 | fan 4 | fan 2, two hosts |
|--------------------------------------------|----------------:|------:|------:|-----------------:|
| entry-method device sends (`-e`)           | 18.3 (44 us)    | 1.5 (3.0 ms) | 1.0 (2.9 ms) | 14.2 (47 us) |
| device channel puts (`lbcost.a40.2n.chan.conf`) | 24.1 (34 us) | 16.1 (19 us) | 13.9 (12 us) | 4.9 (0) |

The two-host row by channels, 4.9 GB/s against the entry path's 14.2 on
the same NIC, is not the link's rate: the next job (23627792, below) found
the cause in the communication library's registration of the regions. The
balancer's exchange is by channels now, so the channel rows are the table's.
lbcalib's sweep had hung on its first exchange (job 23615034): a member
reset its arrival count in its round entry, and a peer's put for the next
round can land before that entry runs (job 23625084, with the runtime's
channel trail: the arrival and its release two lines before the member's
own put). The count is cumulative now.

### The A100 nodes with every put and output recorded (job 23624945, 01:00, 3 min)

The same binary, outputs by channels against the entry-method sends
(`MOE_OUT_ENTRY=1`), serving only, steps 30 to 41:

| two A100 nodes, ms/step      | plain serve | ShedLB serve | the copies' output puts, computeDone to drained |
|------------------------------|------------:|-------------:|------------------------------------------------:|
| outputs by channels          | 24.8        | 21.8         | 11 to 17 ms                                     |
| outputs by entry-method sends| 22.1        | 17.1         | 7 ms                                            |
| FasterMoE                    |             | 16.0         |                                                 |

The output channels are the A100 regression, and the chains say where:
the four copies of expert 5 finish their compute at the same times on
both paths, but with channels their two 16 MB output puts take 11 to 17
ms to complete against 7, and the slow one is the put across the hosts.
The calibration by channels on the A40 nodes had measured that put: a
cross-host channel put runs at 4.9 GB/s a writer where an entry-method
transfer gets 14.2 on the same NIC. The balancer placed all four copies
on one host (the cheapest exchange, which serving does not have), so the
other host's four dispatchers send their 55% of expert 5 across the link
and get the outputs back the same way, about 100 MB each way a step; at
4.9 GB/s that is the 22 ms step, at the entry path's rate the 17.

A cross-host channel put is one one-sided write carrying the arrival
signal as immediate data (the whole 16 MB in one operation, the fabric's
maximum being 2 GB); the entry path is a read by the receiver. Job
23627792 (two A40 nodes, 04:49) took the signal off the write
(`CHARM_CHAN_NOSIGNAL=1`, the note going as a message after the write)
and split the write into 1, 4 and 16 concurrent operations
(`+gpuputchunks`), against the signalled write and the entry-method read,
on lbcalib's fan 2 inter row, 128 MB:

| two A40 nodes, one writer per host, 128 MB | ms per put | GB/s |
|--------------------------------------------|-----------:|-----:|
| signalled write (the default)              | 28.7       | 4.6  |
| plain write, one operation                 | 27.3       | 4.9  |
| plain write, 4 concurrent operations       | 26.6       | 5.0  |
| plain write, 16 concurrent operations      | 21.2       | 6.3  |
| entry-method read                          | 9.5        | 14.2 |

Neither the signal nor the single operation is the cause. The runtime
posts a put and a read the same way (one one-sided operation each, the
same registrations), so the difference is below it. The first suspect was
LCI's libfabric backend: every region was registered with `FI_RMA_EVENT`
(requested on 2026-10-19 for the put with immediate data), and the
Slingshot provider's manual (fi_cxi(7), "Optimized RMA") says a remote
write takes the low-overhead packet format only when the target region
carries no `FI_RMA_EVENT`. Job 23636832 (two A100 nodes, 05:20) measured
it with the flag made opt-in (`LCI_OFI_RMA_EVENT=1` is the old behaviour):
the NIC's restricted-packet counter went from zero to 2.2 million an arm
without the flag, so the format did change, and the rate did not (3.0 GB/s
with the flag, 3.1 without, 3.2 with the signal off as well; the
entry-method read 11.8 on the same nodes). The flag was dead weight (the
signal on that provider is the write followed by a zero-length message,
which needs no remote event) and stays off, but it was not the cost.

The cost is the verb against the memory it lands in. `rma_bw`
(`benchmarks/charm++/cuda/gpudirect/rma_bw/`, LCI over the provider, no
Charm++, one operation at a time, job 23637366, two A100 nodes, the NIC
and the device on the same NUMA node):

| 128 MB, GB/s                                   | one direction | both hosts sending at once |
|------------------------------------------------|--------------:|---------------------------:|
| host to host, put or get                       | 24.0          |                            |
| device source into host memory, put or get     | 16.1          |                            |
| host source into device memory, put            | 12.3          |                            |
| host source into device memory, get            | 24.3          |                            |
| device to device, put                          | 10.9          | 3.7                        |
| device to device, get                          | 16.1          | 12.5                       |

A write landing in device memory is the slow leg (12.3 alone, 3.7 a
direction when both hosts write at once, which is what an exchange does);
the same bytes fetched by a read land at the link's rate. So the runtime's
cross-host channel put is announced and read by the owner again: the path
the channels already use for a put made while the owner is closed for a
move (the announce follows the element, the owner reads in put order and
says DONE), and the one the cross-host tier had until the write with the
note as immediate data replaced it on 2026-10-19; the write stays behind
`CHARM_CHAN_WRITE=1` for measurement. The next jobs run the channel test
with migration, the calibration by channels on the read path (the tables
the balancer prices a copy's exchange from), and the matrices on both
machines. The second lever is the placement, built
while that job queued: the dispatcher partitions are contiguous blocks now
(`partitionOf`: with four dispatchers a host, partitions 0 and 1 are the
first host's and 2 and 3 the second's), each dispatcher's PE declares its
partition to the balancer (`LBManager::setPartition`), and ShedLB, for a
group that declares no exchange, tries the bins of the host that feeds a
copy's block before any other (`ProcStats::partition`, carried with the
load statistics). Serving's copies then sit with their tokens; training's
exchange keeps them on one host as before. Measured (job 23627965, 02:01,
4 min), the copies placed with their inputs, steps 30 to 41:

| two A100 nodes, ms/step                      | plain serve | ShedLB serve | plain train | ShedLB train |
|----------------------------------------------|------------:|-------------:|------------:|-------------:|
| copies with their inputs, entry-method outputs | 26.2      | **13.5**     |             |              |
| copies with their inputs, output channels    |             | 18.6         | 38.8        | 26.8         |
| all copies on one host (job 23624945)        |             | 17.1 / 21.8  |             |              |
| FasterMoE                                    |             | 16.0         | 31.7        |              |

Serving is 1.19x faster than FasterMoE on this fabric by entry-method
outputs, and training 1.18x. Every ShedLB checksum equals plain's under
the new partitions. The channel path still pays its cross-host put rate
(the next job's question). The round placed copies 1, 2 and 3 on the
second host: right for ranks 2 and 3, whose blocks are that host's
dispatchers, wrong for rank 1, whose block under four copies is the first
host's. The planner was incremental: it placed copy 1 while weighing two
copies, where rank 1's block is half the partitions and does sit on the
second host, and kept the placement as it went on to three and four. It
now rebuilds the plan from scratch for every count, each copy placed with
the block it has under that count. Measured (job 23631271, 02:29, 2 min),
steps 30 to 41:

| two A100 nodes, ms/step, the planner per count | ShedLB serve | ShedLB train | the copies' PEs |
|------------------------------------------------|-------------:|-------------:|-----------------|
| output channels                                | **10.6**     | 26.1         | serve 1, 2, 5, 6: one host's two blocks on its own PEs, the other's on its own |
| entry-method outputs                           | 18.4         |              | 1, 3, 2, 0: the second host's bins refused |
| FasterMoE                                      | 16.0         | 31.7         |                 |

By channels, with every copy on the host that feeds it, serving is 1.5x
faster than FasterMoE (steps 25 to 41 run 8.6 to 13.9 ms, the two at 4
and 6 ms are the trace's lighter steps) and training 1.2x; checksums equal
plain's. The entry-method arm placed copies 2 and 3 on the first host
again, and the balancer says why: 57 destination checks were refused
because the PE's host time would pass the cap (the hosts' mean 0.157
against a device target of 0.090: on the entry path the host, not the
device, is the busier resource, and the second host's PEs, which fetch the
hot expert's 16 MB outputs across the link by reads their own host issues,
sit above it). The channels' receive costs the host nothing of that, so
the cap never bites and the copies land where their tokens are. The
channel path is the design; the cross-host put rate (the write into
device memory, above) is what it still pays here in training's exchange
and the copies' remote partition, until the next section's read path.

The host cap on the A40 nodes, same job:

| two A40 nodes, ms/step, steps 30 to 41       | ShedLB serve | ShedLB train | exchange wait, copies 0 and 1 |
|----------------------------------------------|-------------:|-------------:|------------------------------:|
| outputs and block sums by channel            | 21.7         | 46.1         | 11.3 / 3.2 ms a peer          |
| with the host cap                            | 19.8         | 46.7         | 11.3 / 3.2                    |

Serving: four copies of expert 5 (PEs 1, 3, 2 and 0), checksums equal to
plain's; the busiest PE's device is busy 14 ms of a 20 ms period and the
idlest 6. Training is in its band (two copies, compute-bound at 33 ms of
GEMM per copy).

### The read path on the A100 nodes (jobs 23637513, 23637932 and 23637968, 06:08 to 06:25)

Every cross-host channel put is announced and read by its owner (the
previous section). The channel test's six arms, migration included, pass
on it, and lbcalib's sweep by channels (`lbcost.a100.2n.chan.conf`, the
table the balancer prices a copy's exchange from, now built in for the A100
nodes) reads:

| two A100 nodes, by channels, per peer | 2026-11-06 (write) | 2026-11-07 (read) |
|---------------------------------------|-------------------:|------------------:|
| fan 2, one host                       | 89.2 GB/s          | 93.3 GB/s         |
| fan 3, one host                       | 75.2               | 93.4              |
| fan 4, one host                       | 90.1               | 93.3              |
| fan 2, across hosts                   | 3.0                | 12.1              |

The matrix, every row from one binary in one job (23638138, gpua050 and
gpua082, 07:16: the read path with both fixes below and the built-in A100
table), ms/step over steps 30 to 41, FasterMoE on eight ranks with its own
A100 parameters; the first run of the read path (jobs 23637513 and
23637968, gpua010 with gpua029 and gpua007) beside it as the band:

| two A100 nodes, ms/step                | serve | train | copies of expert 5                   | first run: serve / train |
|----------------------------------------|------:|------:|--------------------------------------|-------------------------:|
| example plain                          | 14.5  | 37.9  |                                      | 14.7 / 37.3              |
| example ShedLB, outputs by channels    | 6.2   | 20.3  | PEs 3 / 5, 4 (serve); 3, 2, 0 (train) | 6.3 / 22.6               |
| FasterMoE (schedule + shadows, masters)| 19.8  | 32.2  |                                      | 15.1 / 32.1              |

Serving is 2.4 to 3.2x faster than FasterMoE (its serving step varies
with the node pair, 15.1 to 19.8; the example's 6.2 and 6.3 do not) and
training 1.4 to 1.6x; every ShedLB checksum equals its plain's. Plain
serving itself fell from 24.8 to 14.5: its dispatchers send the slabs to
the other host's experts by the same channels. The training copies'
exchange wait is 6.3 to 8.9 ms a step (12.7 to 19.1 on the write path);
its copies sit on one host because the exchange is what they pay for, and
the balancer's cost table says the exchange across hosts costs 7.7x the
one within.

The read path had been dormant since 2026-10-19 while the migration
protocol was built around the write, and the first training run found two
defects in that interplay, each root-caused from the channel's own trail
(the abort now prints the last 24 events of the channel):

- A sender's closing ack reported its last announced put as written into
  the owner's buffer; the owner then carried the buffer in its pup as bytes
  landed unseen and raised, at its new home, a slab that was never read.
  The ack and the credit accounting now count direct puts only.
- A zero-token slab took the issue path's zero-byte shortcut (the note goes
  straight to the application) while, on the read path, a sender issues
  without waiting for a credit: the note for the step after next overtook
  the current put's release at the owner, which released once for both and
  aborted on the second release. On the read path a zero-byte put joins the
  announce queue, which the owner takes as the note alone.

The job's balancer line says 25 destination checks were refused for host
time in serving (the hosts' mean 0.097 above the device target 0.089: with
the receive costing the host nothing, the dispatchers' routing is what
loads it), and every copy still landed with its inputs. The write path
stays behind `CHARM_CHAN_WRITE=1` for measurement, with
`CHARM_CHAN_NOSIGNAL=1` and `+gpuputchunks` only under it.

### The read path on the A40 nodes (job 23637512, 06:30, gpub011 and gpub095)

The same binary, in one job: the channel test (six arms, migration
included, all pass), the put path by lbcalib on the fan 2 inter row, the
full sweep by channels (`lbcost.a40.2n.chan2.conf`, now the built-in A40
table), and the matrix.

| two A40 nodes, one writer per host, 128 MB | GB/s |
|--------------------------------------------|-----:|
| channel put, read by the owner             | 14.2 |
| channel put, one-sided write (`CHARM_CHAN_WRITE=1`) | 4.9 |
| entry-method read                          | 14.2 |

| two A40 nodes, by channels, per peer | 2026-11-06 (write) | 2026-11-07 (read) |
|--------------------------------------|-------------------:|------------------:|
| fan 2, one host                      | 24.1 GB/s          | 24.1 GB/s         |
| fan 3, one host                      | 16.1               | 16.0              |
| fan 4, one host                      | 13.9               | 13.9              |
| fan 2, across hosts                  | 4.9                | 14.2              |

The matrix, ms/step over steps 30 to 41, the ShedLB arms priced from the
sweep's file (the built-in table carries the same numbers), the 2026-11-05
rows (the write path, the entry-path table) beside them:

| eight A40 GPUs, ms/step        | plain serve | ShedLB serve | ShedLB train | 2026-11-05: plain serve / ShedLB serve / ShedLB train |
|--------------------------------|------------:|-------------:|-------------:|------------------------------------------------------:|
| SmartMoE routing               | 29.4        | 11.1         | 45.5         | 35.8 / 26.1 / 45.8                                     |
| Mixtral routing                | 63.8        | 63.2         | 144.5        | 124.8 / 112.1 / 165.9                                  |
| OLMoE routing                  | 124.8       | 115.4        | 120.8        | 249.9 / 245.8 / 233.2                                  |

SmartMoE serving: four copies of expert 5 at step 20, on PEs 3, 5 and 6
(each with its inputs; 13 destination checks refused for host time, none
of them on the hosts that feed the copies), the step 2.6x plain's; its
plain training 81.2. Training: two copies (the exchange across hosts is
priced at 14.2 GB/s now and the planner keeps two), exchange wait 3.5 and
10.4 ms a step. Mixtral: no copy in serving (no count lowers the fullest
bin enough under the fan rows), the step itself halved by the read path.
OLMoE: no copy; 36 and 50 destination checks refused for host time (the
hosts' mean 0.63 against a device target of 0.26: with 64 experts and
top-8 routing the host, not the device, is the loaded resource there),
serving and training both half of 2026-11-05's. Every ShedLB serving
checksum equals its plain's; the SmartMoE training pair too; the Mixtral
and OLMoE training checksums have no plain training run beside them in
this job.

## Measured (2026-09-06, one A40 node, four PEs, `+gpupool`)

Correctness: `out2` and `w2` are bit-identical across no-LB, sync and async
at every checkpoint, at the small size and at the defaults, with SGD and
Adam, top-1 and top-2, CUPTI-measured and estimated loads, pool and non-pool
transport, and 1 or 4 compute lanes. Migrations happened in every LB run.

Step time at the defaults (64 experts of 2048 x 8192, 8192 tokens per PE),
no LB, as the runtime and the app were fixed during the performance work:

| state                                                        | ms/step |
|--------------------------------------------------------------|--------:|
| first version (one stream per PE)                            |     275 |
| runtime: cross-process receives no longer host-synchronized  |     335 |
| app: comm stream + 4 compute lanes                           |     239 |
| app: sends tagged with dedicated data-ready streams          |     206 |
| app: gather on the send stream                               |     198 |

The max-PE GPU time is 153 ms throughout, so the step is now 77% compute.
What was found, in order (runtime changes are in `src/`, see git):

- **Every cross-process device receive was host-synchronized.** The receive
  path's "same process, different device" case compared node-local device
  indices, which differ for every peer process too; with one device per
  process every direct IPC receive waited on the host for the sender's stream
  and then drained its own. Measured as ~3 ms per 1 MB receive, an exchange
  at 0.5 GB/s on 24 GB/s links, and no overlap of transfers with compute.
  Fixed in `ckrdmadevice.C`; pic2d's no-LB run went from 1.65 s to 0.72 s
  with the same checksum.
- **A send is tagged with its stream's tail at send time.** With the
  host waits gone, a landing copy carries a device-side wait on the sender's
  stream. Anything queued on the sender's stream after the data was produced
  (another expert's GEMMs on a shared lane, landings waiting on other PEs on
  a shared comm stream) delays the receiver's stream and everything behind
  it. Hence three stream roles per PE: `comm` for every landing and the
  combine, compute lanes for the experts, and one send stream per sender
  that waits only on the event marking its own data ready.
- **Runtime hygiene found on the way:** plain DEVICE-mode pups did not note
  pool sources as read (a pool free could race the pack copy); the unpack
  copied on the legacy default stream behind every other payload's landing;
  the polled event queue was one FIFO across streams, so a landing recorded
  after a 150 ms compute-done event was not noticed until the kernel ended.
  All three are fixed; none was the bottleneck here.

What is left, from the trace: token exchange 33 ms, output exchange 10 ms,
combine and drain 7 ms per step. Consumption of the 48 remote slabs a PE
receives is paced at 0.3 to 0.5 ms per slab whatever their size (26 MB and
96 MB take the same 30 ms), and the profile is flat across the scheduler,
network progress, pxshm and event queries: per-message host work in the
device zerocopy path, roughly 0.35 ms all-in on the receiver. Fewer, larger
messages (a slab per destination PE instead of per expert) would remove most
of it at the cost of location transparency; a cheaper per-message path in
the runtime would keep it.

Load balancing, final build (DiffusionLB, `-f 5 -b 5`, async `-l 3`), after
the source fusion and the load-model fix below:

| config                      | no LB | sync | async |
|-----------------------------|------:|-----:|------:|
| defaults, 30 steps, ms/step |   191 |  199 |   185 |
| 4096 x 14336, 20 steps      |   560 |  545 |   530 |

Before those two changes the same runs were 205 / 216 / 207 and 631 / 623 /
605. What changed:

- **One chunked pass per expert, not one per source** (the default; `-U`
  restores the old way). A chunk runs two weight-update GEMMs that read and
  rewrite both weight matrices whatever the chunk holds, so a step's cost is
  set by the number of chunks, and per source that is at least one per
  (source, expert) pair: a typical expert receives a few hundred tokens from
  each dispatcher and paid a full weight update for each. At the defaults that
  is 260 chunked passes per step against 78, and 30 ms per GPU of weight
  traffic. The expert's tokens are copied into one run first and the outputs
  copied back, about 0.8 ms per GPU, so the sends and both pup phases are
  unchanged. Concatenating in dispatcher order keeps the chunk boundaries a
  function of the routing alone: the checksums differ from the unfused ones in
  value and are still identical across placements. Compute span 154 -> 131 ms.
- **Seconds per token is pooled, not per expert** (per PE when this was
  written; job-wide since 2026-10-31, below). An expert timed
  itself with events around its own kernels, but up to `-n` experts share the
  device, so the span was inflated by however much company it had -- and a hot
  expert, outliving its lane-mates, had less of it. Per-expert figures
  compressed exactly the skew the balancer has to see. Every expert here does
  identical work per token, so one pooled figure is both the right model and an
  accurate one. `CHARM_MOE_FLATLOAD=1` declares an exact token-proportional
  load instead, which separates a balancer that cannot use a good signal from
  a bad signal.
- **The checksum's copy is waited for before a pack reads it.** `h_chk` is
  filled by an asynchronous device-to-host copy, and `pup_device_order` orders
  only DEVICE-mode copies, so a migration in that window packed whatever was
  there before. It cost one expert's slot in the `w2` reduction, seen once in
  eight async runs at Llama size as a deficit of exactly 1/64. The outputs
  were never affected: `out2` matched in the failing run.

GreedyRefineCentralGPULB, a centralized greedy re-placement of the kind the
reference uses, was tried on the same runs. It reaches the same balance as
DiffusionLB and loses anyway, because it moves four to five times as many
experts:

| balancer, defaults           | ms/step | experts moved | tokens/PE |
|------------------------------|--------:|--------------:|-----------|
| no LB                        |     191 |             0 | 1.67-1.84 |
| DiffusionLB, sync            | 199-201 |         19-23 | 1.07-1.25 |
| DiffusionLB, async           | 189-205 |         18-27 | 1.17-1.31 |
| GreedyRefineCentralGPULB sync| 281-298 |        95-112 | 1.07-1.25 |
| GreedyRefineCentralGPULB async|    212 |         53-55 | 1.13-1.35 |

`+LBPercentMovesAllowed 25` brings its async case to 209 ms, still behind both.
At 134 MB of weights per expert the migration traffic costs more than the
balance buys, which is the same conclusion the move-cap experiment reached from
the other direction. Two runtime bugs had to be fixed before this could be
measured at all; see the git log for `+gpupool`'s null comm buffer and the
memory contract's staging figure.

What the balancer still leaves on the table: between reshuffles DiffusionLB
holds `tokens/PE max/avg` at 1.20 to 1.33, where a greedy re-placement reaches
1.01 (see `ref/`). The per-step move cap is a count applied to objects of very
unequal load, so it can stop a shed part-way -- a node asked to shed 0.187 of
its load stopped after four objects with 0.102 unshed and the job sat at 1.9x
for five steps. Lifting the cap (`+LBDiffusionMaxMoveFrac 1`) does recover in
one step and holds 1.2 to 1.33, but moves 35 to 54 objects where the capped
runs move 6 to 9, and the step time does not improve. At this object size the
migration traffic pays for the balance it buys; a cap expressed in the load
dimension rather than by object count is the open item.

Balancing cuts tokens per PE from 1.9x to 1.3x and the GPU-time imbalance
from 1.5x to 1.15x. In the Llama-sized run the balanced steps take 476 to 497
ms against 575 to 601 ms without balancing, until the hot set reshuffles and
the next LB step catches up; the averages above include the reshuffle steps.
Async now overlaps everything: its first LB step costs nothing visible (228
ms against 529 ms in sync at the defaults).

Tracing: `CHARM_MOE_TRACE=1` prints timestamped LB, migration and step
events per PE, `=2` adds each expert's compute and send events, `=3` each
slab's post and consume. `ranklogs/phases.py <dir>` decomposes a level-2
trace into routing, token exchange, compute, output exchange and drain;
`ranklogs/tracesum.py` summarizes migration costs; `ranklogs/pairs.py`
per-pair latencies from a level-3 trace. `CHARM_MOE_HOSTSYNC=1` restores the
original host-synchronized pack and free for A/B runs. With more than one
lane the per-expert device time is a wall-clock span on a shared device, so
`gpu/PE max` overcounts; its ratio across PEs is still meaningful.
The runtime's own GPU timeline, `+gputrace <dir>` (`+gputracenocopy` for
kernels only), records every kernel and copy with the entry methods and
idle time per PE; `../sph3d/gputrace_merge.py` builds the Perfetto trace
and `ranklogs/kernelclass.py --steps A:B` the per-step time by kernel class
(GEMM, NCCL, copies, other) per rank, with the step windows from the
`Dispatcher::runStep` entries. The same tool reads the FasterMoE and
Megatron harnesses' torch.profiler traces (`ref/fastmoe_moe.py --profile`),
and `ranklogs/kineto_merge.py` joins their ranks into one Perfetto trace;
"Measured 2026-11-05: GPU timelines" compares the two systems that way.
`ranklogs/stepchain.py <dir> --step S` reads such a trace as one step's
chain: per PE the dispatcher's gather, puts, outputs, combine and drain, per
expert copy its slabs, compute, output puts, exchange and apply, on one clock
(`--group E` lines up an expert's copies; `--steps A:B` averages). The
tables of "Measured 2026-11-06" come from it.

Running a single PE: a one-task `srun` step owns only eight of the
allocation's cores, so the NUMA wrapper's `+pemap` aborts the run
(`CmiSetCPUAffinity failed`) with an empty log. Launch single-PE tests
without the wrapper.

## Reference implementation

`ref/moe_ref.py` is the same layer in PyTorch with NCCL, written the way an
MoE training framework writes it: one `all_to_all_single` per exchange instead
of one message per expert, one chunked pass per expert instead of one per
source, and expert re-placement between steps instead of object migration. Its
routing, initialization and update are ports of this example's, checked bit for
bit by `ref/selftest.py`, so the two produce the same checksums and the same
imbalance and can be compared step for step on the same node.

On four A40s at the defaults the example takes 204 ms per step without
balancing against the reference's 189, and 195 ms balanced against the
reference's 153. The gap is not the migration mechanism: the example's async
balancing beats its own sync, while the reference's does not, because
overlapping NCCL weight transfers costs a step of stale placement and shares
the links with the dispatch. The gap is the placement decision, where greedy
re-placement reaches a token imbalance of 1.01 against DiffusionLB's 1.28, and
the GEMM structure, where one pass per expert is worth 30% of the compute
time. `ref/README.md` has the tables and the derivations.

`ref/megatron_moe.py` runs the same layer through Megatron-Core's MoE layer
(expert parallelism, all-to-all dispatcher, optional grouped GEMM and bf16)
with the example's routing in place of its top-k, as a throughput reference
for a production framework; `ref/run_mg.sh` launches it on one A40 node. Its
numbers are in the 2026-10-31 section above.

## Several PEs per process

`MOE_PPN=N` through `ranklogs/moewrap.sh` (or `+ppn N +pemap <cores>` by
hand) runs N PEs per process on one GPU. The program is the same at any N:
one dispatcher per GPU, on the first PE the runtime maps to it
(`hapiDeviceIndexOfPe`: `+ppn` PEs over `+gpudevs` devices, `+gpumap`), owns
that GPU's `-t` tokens, routes them and ships one slab per expert exactly as
with one PE, so the work per GPU, the slab count and the routing do not
change with N and the checksums match the one-PE run bit for bit. What N
changes is where the experts run: they spread over all the process's PEs,
each with its own compute context (the lane count defaults to four per GPU
divided among the PEs the runtime puts on it, `-n` overrides), so their
kernel launches, landing copies and output sends run on N PEs instead of
one, and the balancer moves them between the PEs of a process on host time
as well as across processes on device time. The other PEs' dispatcher
branches hold no tokens and run no steps; they exist for that context.

The same rule gives one process per node driving all its GPUs:
`MOE_GPUS_PER_PROC=4` in the wrapper runs `+ppn 4` with PE i on the first
core of GPU i's NUMA node, the runtime's block map puts PE i on device i,
and the example then has a dispatcher on every PE. The work per GPU is the
one-process-per-GPU work; what changes is the put between two GPUs of a node,
a copy in one address space instead of an IPC write into a mapped buffer
("Measured 2026-11-06" for why that matters).

Measured 12 Sep 2026, one A40 node, 8 cores per process, ms/step:

    PEs per process          1        2        4        6        7        8
    no LB, defaults      188-193    193   180-182    183      181    202-205
    async LB, defaults   215-248     -    216-230     -        -     216-230
    no LB, Llama size      557       -      530       -        -       576
    async LB, Llama        659       -      650       -        -       721

Four PEs per process is the sweet spot, 6% faster than one with no
balancer and 5% at the Llama size. The step's phase trace
(`CHARM_MOE_TRACE=2`, `ranklogs/phases.py`) shows where: route+gather
23.8 -> 10.9 ms, compute span 131 -> 128, combine 14 -> 18, output
exchange 17 -> 12.5. At eight PEs the output exchange -- the time from
the last expert's output sends to the dispatcher having every output
landed -- triples to 46-53 ms and eats the gains; the dispatcher PE consumes
landed outputs in bursts tens of milliseconds apart instead of as they
arrive. Same-process and remote outputs are delayed alike, so it is not the
intra-process transport, and `MOE_ONE_DEVICE=1` (which cuts the driver's
per-GPU threads from four to one) leaves it unchanged, so it is not thread
oversubscription of the eight cores either. Open. Use four until it is
understood.
