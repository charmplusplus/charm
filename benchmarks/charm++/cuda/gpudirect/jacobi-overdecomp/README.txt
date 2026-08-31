jacobi-overdecomp -- portable stencil for GPU overdecomposition experiments
===========================================================================

A copy of examples/charm++/cuda/gpudirect/jacobi2d ported to the hapi_portable.h
macros so it builds for both backends, kept separate from the example so
experiments here disturb nobody. The algorithm, decomposition and SDAG are
unchanged; what is added is measurement.

Build
-----
  make GPU=cuda CHARM_DIR=<build> [CUDA_ARCH=sm_80]
  make GPU=hip  CHARM_DIR=<build> [GPU_ARCH=gfx90a]

GPU=hip implies a reconverse build: GPU-direct under HIP exists only in the
reconverse half of ckrdmadevice.h.

What it reports
---------------
Two machine-readable lines, in addition to the original's timings.

  JACOBI_OD_RESULT pes=.. nodes=.. phys_nodes=.. visible_devices=..
                   chares=.. chares_per_visible_device=.. iters=..
                   avg_iter_us=.. zerocopy=..

    One line per run, for sweeping overdecomposition. Raise chares per device
    until avg_iter_us stops improving; compare that point against the hardware
    queue ceiling (32 on NVIDIA with CUDA_DEVICE_MAX_CONNECTIONS=32, ~8 on AMD
    with GPU_MAX_HW_QUEUES=8) divided by streams per chare -- this program uses
    two, a compute stream and a higher-priority comm stream.

  JACOBI_OD_PLACE block=(x,y) pe=.. node=.. device=.. send_us_per_iter=..

    One line per chare. Two uses:

    - Confirm the run got the configuration intended. For multiple GPUs per
      process, distinct `device` values must appear within one `node`. If every
      chare in a process reports the same device, the mapping collapsed and the
      experiment is not testing what it looks like.

    - send_us_per_iter isolates the cost of issuing ghost sends. On the
      same-process (MEMCPY) path the runtime synchronises the stream before the
      metadata message goes out (issue #3957), so a blocking wait shows up here:
      it grows with chares per PE while kernel time stays flat. Comparing this
      against avg_iter_us across a chares-per-PE sweep is the measurement that
      settles how much that blocking costs.

Suggested first runs on a multi-GPU node
----------------------------------------
  # 2 processes per node, GPUs divided between them, >1 GPU per process
  srun -N1 -n2 -c<cores/2> --gpus-per-node=8 ./jacobi_od -W 8192 -H 8192 \
       -w 1024 -h 1024 +pe <pes>

  # then sweep the block size (hence chares per device) with everything else fixed
  for w in 2048 1024 512 256; do ... -w $w -h $w ... ; done

Set CUDA_DEVICE_MAX_CONNECTIONS=32 (or GPU_MAX_HW_QUEUES=8 on ROCm) before
running: at the default of 8, streams alias onto shared queues and the
measurement reports the driver's ceiling rather than the runtime's behaviour.
