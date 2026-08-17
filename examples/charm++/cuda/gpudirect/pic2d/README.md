# pic2d — 2D electrostatic PIC miniApp for GPU load balancing

A Particle-in-Cell miniApp designed to exercise Charm++'s CUPTI-based GPU
load balancing with *natural, dynamic* load imbalance. Unlike
`jacobi2d-imbalance` (artificial, static imbalance via an extra-iterations
knob), the GPU load here is driven by the particles themselves: a Gaussian
bunch drifts across the domain, so the per-patch particle count — and with it
the cost of the deposit and push kernels — shifts between chares over time.

## Physics

Standard 2D electrostatic PIC on a fully periodic domain, in normalized units
(dx = dy = ε₀ = 1):

1. **Deposit**: electrons (q/m = −1) are deposited onto a cell-centered charge
   grid with CIC (bilinear) weighting. An immobile uniform ion background
   neutralizes the mean charge, so ρ̄ is known analytically and no global
   reduction is needed.
2. **Field solve**: ∇²φ = −(ρ − ρ̄) via a fixed number of Jacobi iterations
   per step (`-j`), warm-started from the previous step's φ.
3. **E field**: E = −∇φ by central differences.
4. **Push**: CIC gather of E at each particle, leapfrog update, periodic wrap.

The per-particle charge is scaled by a coupling constant (`-q`) so the bunch's
self-field stays gentle; the load movement comes primarily from the bunch
drift (`-v`), not from electrostatic blowup.

## Decomposition and communication

A 2D chare array of patches; each patch owns a `-w`×`-h` block of cells plus
the particles inside it. All inter-chare data movement is GPU zerocopy
(`CkDeviceBuffer` + post entry methods), no host staging:

- **Charge ghosts** (8-way, faces + corners): edge deposits land in the ghost
  ring and are shipped to neighbors, which accumulate them into their interior
  boundary cells.
- **Phi ghosts** (4-way, once per Jacobi iteration, plus a final
  exchange-only round): double-buffered by round parity since a neighbor may
  run one round ahead.
- **E ghosts** (8-way): the gather needs E in ghost cells, including corners.
- **Particles** (8-way): leavers are compacted on-device into per-direction
  send buffers by the push kernel; zero-count messages are still sent to keep
  the SDAG receive counts fixed.

Each patch uses two CUDA streams (compute + comm) with event-based
cross-stream ordering, following the `jacobi2d-imbalance` conventions.

## Load balancing integration

- `usesAtSync = true`; `AtSync()` is called at iteration `-f` and every `-b`
  iterations after that, with both streams drained first.
- All kernels are launched from Patch entry methods, so the CUPTI external
  correlation set up by the runtime attributes every kernel (deposit, jacobi,
  push, packing, ...) to the owning chare automatically — the app contains no
  explicit instrumentation calls.
- Migration PUPs only the live particles and φ (warm start) with
  `PUP::PUPMode::DEVICE`; ρ, E and all exchange buffers are reallocated and
  recomputed. Device buffers are allocated with `hapiMalloc`, so per-chare
  GPU memory attribution hooks see them.

## Building

Requires a Charm++ build with CUDA and device zerocopy support (the `mpi` or
`ucx` machine layers define `CMK_GPU_COMM`), e.g.:

```sh
./build charm++ ucx-linux-x86_64 cuda smp -j16 --with-production
cd examples/charm++/cuda/gpudirect/pic2d
make
```

## Running

```
./pic2d
  -W/-H  grid cells (default 1024 x 1024)
  -w/-h  patch cells (default 128 x 128  ->  8 x 8 chares)
  -p     particles per cell (default 32)
  -i     timed iterations (default 100)
  -u     warmup iterations (default 10)
  -j     Jacobi iterations per step (default 20; 0 = field-free drift)
  -d     initial distribution: 0 uniform, 1 gaussian bunch,
         2 two-stream (default 1)
  -F     fraction of particles in the bunch (default 0.5, -d 1 only)
  -s     bunch sigma as fraction of min(W,H) (default 0.1, -d 1 only)
  -v     drift velocity, cells/unit time (default 20); +x bunch drift
         for -d 1, counter-streaming +-x beam speed for -d 2
  -t     thermal velocity (default 1.0)
  -T     dt override (default: capped so nothing crosses a patch per step)
  -q     field coupling strength (default 1e-3)
  -f     first LB iteration (default 10)
  -b     LB frequency (default 9999)
  -r     particle capacity headroom factor (default 12)
  -x     exchange buffer fraction of capacity (default 0.05)
  -c     stats print frequency, 0 disables (default 1)
  -P     print final per-patch particle counts
```

The `-d/-F/-s` knobs control how concentrated the initial distribution is and
therefore how strong the imbalance is: `-d 0` is perfectly balanced, larger
`-F` / smaller `-s` concentrate more particles in fewer patches.

`-d 2` starts perfectly uniform but splits the particles into two
counter-streaming beams (±`-v` in x), driving the **two-stream instability**:
density perturbations grow exponentially from shot noise at a rate ~ω_p =
√coupling, bunching particles at a wavelength of roughly 2π·v/ω_p. Unlike the
drifting bunch, the resulting load imbalance is self-generated and emerges
where the instability happens to peak. With the default `-q 1e-3` the
e-folding time is ~300 steps; raise the coupling (e.g. `-q 0.1`, e-folding
~30 steps) or run longer to see the imbalance develop within a benchmark run.

### Sanity checks

Particle count conservation is verified by a reduction every step (the run
aborts if a particle is lost), and the initial particle count is verified
against the expected total.

```sh
# uniform, no drift: imbalance ratio should stay ~1.00
./pic2d -d 0 -v 0 -i 20 +p4

# bunch drifting through patches; max/avg np should move and stay > 1
./pic2d -d 1 -i 50 -c 5 +p4
```

### Load balancing demo

~50% of particles start in a few patches and the hot spot drifts in +x,
crossing roughly one patch every ~60 steps with the default dt:

```sh
# baseline: no LB
./pic2d -W 2048 -H 2048 -w 256 -h 256 -i 100 -u 10 -f 99999 -c 10 +ppn 8

# CUPTI-driven GPU LB, first at iter 10, then every 30 iterations
./pic2d -W 2048 -H 2048 -w 256 -h 256 -i 100 -u 10 -f 10 -b 30 -c 10 +ppn 8 \
        +balancer GreedyRefineCentralLB
```

Compare `Total time`, the per-window `ms/iter` lines, and the particle
imbalance ratio. Because the bunch keeps moving, periodic re-balancing
(`-b 30`) should beat a single LB invocation over long runs.

For self-generated imbalance from a uniform start, use the two-stream mode
with a stronger coupling and a longer run; the imbalance ratio should climb
from ~1.0 as the instability grows:

```sh
./pic2d -d 2 -q 0.1 -i 300 -u 10 -f 50 -b 50 -c 20 +ppn 8 \
        +balancer GreedyRefineCentralLB
```

### Memory sizing

Per patch, the dominant allocations are `2 * capacity * 16 B` for the particle
double buffer plus `16 * exch_capacity * 16 B` for exchange buffers, where
`capacity = min(ppc * w * h * headroom, total particles)`. With defaults
(`-p 32 -w 128 -h 128 -r 12`) that is ≈230 MB per patch. Reduce `-p` or `-r`
for small GPUs; increase `-r` (or use `-d 0`) if a hot patch aborts with a
capacity error.

## Known assumptions

- Homogeneous GPUs: particle initialization generates the global distribution
  redundantly on every patch and filters by ownership, which assumes all
  devices compute bit-identical positions (same binary, same architecture).
- A particle must not cross more than one patch per step; the push kernel
  flags violations and the run aborts with a pointer to `-T`/`-v`.
- Exchange buffers are fixed-capacity; overflow aborts with a pointer to
  `-x`/`-r` rather than resizing.
