# Device channels on Delta and Frontier: what to run

The `device-channel` branch replaces `CkDevicePersistent` with device channels
(`CkDeviceChannel`, manual section "GPU Channels"). It is built on `recharm` c40d81714.
Before it is proposed upstream, it needs the runs below on real GPUs and a real network:

- Delta: A40 GPUs, CUDA;
- Frontier: MI250X GPUs, HIP.

This file covers the builds, the runs, and how to read the logs. It exists only for these
runs and will be removed from the branch before a pull request.

## What has run, and what has not

All of these pass on one RTX 3050 (CUDA 12.0, LCI over libfabric's tcp provider):

- chantest, with and without migration, in one process and in two;
- CkDyn's `devpersist` phases;
- jacobi3d `-k`, whose checksum matches the host-staged run;
- `anytime_migration` and `migrate_then_atsync`.

That machine has one GPU, so the two processes shared it.

None of the following has ever run:

| Path | Shape that reaches it | Steps below that reach it |
|---|---|---|
| A put between hosts: the receiver reads the sender's device buffer with RDMA | 2 nodes | chantest, channel, jacobi3d `-k` |
| A CkDyn persistent transfer between hosts (message, RDMA get, ack) | 2 nodes | ckdyntest `devpersist/remote` |
| IPC between two different GPUs | 2 processes on 1 node | chantest, ckdyntest, jacobi3d `-k` |
| A put between two GPUs inside one process | 1 process, 2 PEs | chantest |
| The deferred move over a real network (`emigrate` waits for outstanding channel work) | 2 nodes | chantest `-l`, the site tier |
| A staged put (see "Staged puts" below) | any multi-process shape, by timing only | chantest `-l -p 1` |
| Everything on HIP, starting with the compile | Frontier | all of the above |

A staged put is one the sender makes after it has acked the receiver's closing and
before the receiver's new location reaches it.

## Getting the branch

Delta:

```
cd $HOME
git clone -b device-channel https://github.com/charmplusplus/charm.git charm-channel
cd $HOME/charm-channel
git submodule update --init contrib/reconverse
```

Frontier:

```
cd /lustre/orion/csc710/scratch/$USER/hip
git clone -b device-channel https://github.com/charmplusplus/charm.git charm-channel
cd /lustre/orion/csc710/scratch/$USER/hip/charm-channel
git submodule update --init contrib/reconverse
```

## Building

`buildcmake` fetches the LCI named by the reconverse pin (uiuc-hpc/lci 9757d6ab) and
builds it with the tree's GPU support. Use this LCI on both machines, for two reasons:

- it has the ROCr registration fix for libfabric 2.5;
- it does not request `FI_RMA_EVENT` from cxi.

If `LCI_ROOT` is set in the environment, the build uses that install instead. That is
why the commands below unset it.

Without `--with-production`, `buildcmake` builds Debug.

### Delta

First load Delta's cmake, gcc, CUDA and libfabric modules. recharm's
`tests/reconverse-site-run.sh` does not pin their names either, because they change with
the software stack.

```
cd $HOME/charm-channel
pkg-config --modversion libfabric
unset LCI_ROOT LCI_DIR
./buildcmake charm++ reconverse-linux-x86_64 cuda -j 16 --with-production \
  --with-cmake-args="-DLCI_NETWORK_BACKENDS=ofi" 2>&1 | tee build-cuda.log
grep 'GIT_TAG 9757d6ab' build-cuda.log
```

Check two things in the output:

- the `pkg-config` line prints a version, because LCI finds libfabric through
  `pkg-config`;
- the `grep` finds the fetch.

Then build the programs. These commands target the A40 (`sm_86`); on the A100 nodes, use
`CUDA_ARCH=sm_80`.

```
cd $HOME/charm-channel
T=$PWD/reconverse-linux-x86_64-cuda; CU=$(dirname $(dirname $(which nvcc)))
for d in examples/charm++/cuda/gpudirect/chantest examples/charm++/cuda/gpudirect/jacobi3d benchmarks/charm++/cuda/gpudirect/channel; do make -C $d CHARM_DIR=$T GPU=cuda HAPITOOLKIT_HOME=$CU CUDA_ARCH=sm_86 || break; done
make -C tests/charm++/ckdyn CHARMC=$T/bin/charmc ckdyntest
```

### Frontier

Nothing in this branch has ever been compiled with HIP. A compile error in any of these
files is the first finding:

- `src/ck-core/ckdevicechannel.C`;
- `src/ck-core/ckdyn.C`;
- `src/arch/cuda/hybridAPI/hapi_portable.h`.

If one occurs, bring back the error.

```
cd /lustre/orion/csc710/scratch/$USER/hip/charm-channel
source /lustre/orion/csc710/scratch/$USER/hip/env.sh
export CC=cc CXX=CC FC=ftn PATH=$ROCM_PATH/lib/llvm/bin:$PATH
pkg-config --modversion libfabric
unset LCI_ROOT LCI_DIR
./buildcmake charm++ reconverse-linux-x86_64 amd -j 16 --with-production \
  --with-cmake-args="-DLCI_NETWORK_BACKENDS=ofi -DCMAKE_HIP_ARCHITECTURES=gfx90a" 2>&1 | tee build-amd.log
grep 'GIT_TAG 9757d6ab' build-amd.log
T=$PWD/reconverse-linux-x86_64-amd
for d in examples/charm++/cuda/gpudirect/chantest examples/charm++/cuda/gpudirect/jacobi3d benchmarks/charm++/cuda/gpudirect/channel; do make -C $d CHARM_DIR=$T GPU=hip HIP_ARCH=gfx90a || break; done
make -C tests/charm++/ckdyn CHARMC=$T/bin/charmc ckdyntest
```

What these settings are for:

- `env.sh` loads the module set from the Frontier work of 2026-10-22. It leaves out
  `craype-accel-amd-gfx90a`, whose compiler wrappers link a ROCm 6 library.
- `CC=cc` keeps the build off the system GCC 7.5.
- `charmc` calls a bare `clang`, hence the `PATH` change.

## The channel runs

One batch job runs every step, each as its own `srun`, and then ends.

The job script is the block at the end of this section. To extract it to
`chan_site.sh`, run this from the checkout's root:

```
sed -n '/^#!\/bin\/bash$/,/^echo "done/p' DEVICE_CHANNEL_SITE_TESTS.md > chan_site.sh
```

Submit the job from the shell you built in, so that it inherits that environment.

Delta:

```
cd $HOME/charm-channel
sbatch -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=2 --gpus-per-node=4 -c 8 -t 01:00:00 -o chan-site-%j.out chan_site.sh delta $PWD
```

Frontier:

```
cd /lustre/orion/csc710/scratch/$USER/hip/charm-channel
sbatch -A csc710 -p batch -N 2 --ntasks-per-node=2 --gpus-per-node=8 -c 8 -t 01:00:00 -o chan-site-%j.out chan_site.sh frontier $PWD
```

Each step is limited to 2 minutes, and each ckdyntest step to 5. The job therefore ends
within the hour even if every step hangs. A typical step takes seconds.

The launcher settings are recharm's, from `tests/reconverse-site-run.sh`:

- Delta needs `--mpi=pmix`, and Frontier plain `srun`. With `--mpi=pmi2`, every task
  starts as its own one-process job, and every multi-process test passes without testing
  anything.
- Every Frontier step needs `--network=single_node_vni`; without it, cxi finds no VNI on
  a one-node step.

GPUs: recharm gives each process the GPU whose number is the process's rank on the node.
So:

- the two processes of a one-node step use GPUs 0 and 1;
- the two PEs of the one-process step use GPUs 0 and 1 within that process.

Each log's `HAPI> Config:` line shows what was used.

The steps, by shape:

- `1p` is one process; `2p1n` is two processes on one node; `2p2n` is two processes, one
  on each node.
- `4p2n` is two processes on each node. jacobi3d runs in this shape, so its neighbours
  are reached by a copy within a PE, by IPC on the node, and by RDMA across nodes.

```
#!/bin/bash
# chan_site.sh <delta|frontier> <checkout>: the device-channel site runs of
# DEVICE_CHANNEL_SITE_TESTS.md. One log per step in <checkout>/site-runs/<job id>/,
# and summary.txt there.
SITE=$1; SRC=$2
case $SITE in
  delta)    T=$SRC/reconverse-linux-x86_64-cuda; GPN=4; SRUN="srun --mpi=pmix"
            export FI_MR_CACHE_MONITOR=uffd ;;
  frontier) T=$SRC/reconverse-linux-x86_64-amd; GPN=8; SRUN="srun --network=single_node_vni"
            export FI_MR_CACHE_MONITOR=kdreg2 PMI_MAX_KVS_ENTRIES=1000 ;;
  *) echo "usage: chan_site.sh <delta|frontier> <checkout>"; exit 2 ;;
esac
export LD_LIBRARY_PATH=$T/lib:$LD_LIBRARY_PATH FI_PROVIDER=cxi LCI_NETWORK_BACKENDS=ofi
ulimit -c 0
OUT=$SRC/site-runs/$SLURM_JOB_ID; mkdir -p $OUT; S=$OUT/summary.txt
CH=$SRC/examples/charm++/cuda/gpudirect/chantest
J3=$SRC/examples/charm++/cuda/gpudirect/jacobi3d
BM=$SRC/benchmarks/charm++/cuda/gpudirect/channel
KD=$SRC/tests/charm++/ckdyn
KEY='Starting Reconverse|HAPI> Config|\[chantest\] (block|PASS)|PASS devpersist|mode (copy|IPC|RDMA)|ALL DONE|Checksum|CHAN-WATCH|wrong at|second sender|Reason:|FAIL'

# run <name> <nodes> <processes> <directory> <program and arguments>; TMO= seconds (120)
run() {
  local name=$1 n=$2 np=$3 dir=$4; shift 4
  (cd $dir && timeout ${TMO:-120} $SRUN -N $n -n $np --ntasks-per-node=$((np / n)) --gpus-per-node=$GPN \
     -c 8 --cpu-bind=none --exact --kill-on-bad-exit=1 stdbuf -oL -eL "$@") > $OUT/$name.log 2>&1
  local rc=$?
  echo "$name rc=$rc | $(grep -E "$KEY" $OUT/$name.log | tr '\n' ' ' | cut -c1-600)" >> $S
}
# count <name>: the protocol lines of a CHARM_DEBUG_CHANNEL step that name the path taken
count() {
  local f=$OUT/$1.log
  echo "    shm credits $(grep -c 'CHAN-CREDIT .*via shm' $f), RDMA reads $(grep -c 'CHAN-READ.*RDMA' $f)," \
       "staged puts $(grep -c 'CHAN-STAGED' $f), IPC failures $(grep -c 'IPC open of PE' $f)," \
       "closings acked $(grep -c 'CHAN-CLOSE.*ack from' $f)" >> $S
}

echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, $(date), $(git -C $SRC rev-parse --short HEAD)" > $S
for shape in "1p 1 1" "2p1n 1 2" "2p2n 2 2"; do
  read s n np <<< "$shape"
  run chantest_${s}_plain $n $np $CH ./chantest +pe 2 -i 40
  run chantest_${s}_lb    $n $np $CH ./chantest +pe 2 -i 40 -l -p 3
  run chantest_${s}_lb1   $n $np $CH ./chantest +pe 2 -i 60 -l -p 1
  run chantest_${s}_4mb   $n $np $CH ./chantest +pe 2 -i 20 -s 524288 -l -p 2
  run chantest_${s}_dbg   $n $np $CH env CHARM_DEBUG_CHANNEL=1 ./chantest +pe 2 -i 12 -l -p 1
  count chantest_${s}_dbg
  run chantest_${s}_qd    $n $np $CH ./chantest +pe 2 -i 100 -q
done
for shape in "1p 1 1" "2p1n 1 2"; do
  read s n np <<< "$shape"
  TMO=120 run chantest_${s}_hold $n $np $CH ./chantest +pe 2 -i 5 -w 40
  echo "    probe lines $(grep -c 'CHAN-PROBE' $OUT/chantest_${s}_hold.log)" >> $S
  run chantest_${s}_twosend $n $np $CH ./chantest +pe 2 -i 5 -x
done
TMO=300 run ckdyn_2p1n      1 2 $KD ./ckdyntest +pe 4 -skip device
TMO=300 run ckdyn_2p2n      2 2 $KD ./ckdyntest +pe 4 -skip device
TMO=300 run ckdyn_2p2n_full 2 2 $KD ./ckdyntest +pe 4
for m in host: zerocopy:-d channels:-k; do
  run jacobi3d_4p2n_${m%%:*} 2 4 $J3 ./jacobi3d +pe 4 -c 32 -i 100 -w 10 ${m#*:}
done
run channel_2p1n 1 2 $BM ./channel +pe 2 -x 1048576
run channel_2p2n 2 2 $BM ./channel +pe 2 -x 1048576
echo "done $(date)" >> $S
```

## Reading the results

`site-runs/<job id>/summary.txt` has one line per step: its name, its exit code, and the
lines from its log that matter.

### Every step: the process count

Find the `Reconverse> Starting Reconverse with` line. It must appear once and name the
step's process count: 1, 2 or 4.

If it appears once per task and each copy says 1 process, the launcher started separate
jobs, and nothing that step reports counts.

### Every step: watchdog lines

A `CHAN-WATCH` line in any step is a finding, even if the step passed. It means the
runtime's watchdog either:

- saw a channel make no progress for 10 s, or
- counted refused or misdirected protocol messages.

Bring back that step's log.

### chantest

- **Passing:** exit 0 and one `[chantest] PASS` line, ending `quiescence reached`: every run
  ends by quiescence detection, so a protocol message or a shared-memory note or credit left
  uncounted shows as a timeout here.
- **Banner:** names the step's processes and hosts, for example `2 process(es), 2 host(s)`
  in `2p2n`.
- **With `-l`:** the PASS line counts the moves. chantest aborts if there were none.
- **A wrong value:** aborts with `wrong at index`.
- **`_qd` (`-q`):** every iteration ends by quiescence. `quiescence reached before` is a
  failure: an arrival was still on its way.
- **`_hold` (`-w 40`):** the sender waits 40 s for a credit, so the watchdog probes the owner
  (the indented line counts `CHAN-PROBE` lines, more than 0) and the run must still pass.
- **`_twosend` (`-x`):** passes by aborting with `has a second sender`, naming both elements.

### The `_dbg` steps

These run with `CHARM_DEBUG_CHANNEL=1`. The indented line below each in `summary.txt`
counts the protocol lines that name the path taken:

| Step | Expected |
|---|---|
| `chantest_2p1n_dbg` | shm credits > 0, RDMA reads 0, IPC failures 0 |
| `chantest_2p2n_dbg` | RDMA reads > 0 |
| every `_dbg` step | closings acked > 0 (the elements moved with their channels open) |

An IPC failure count above 0 means an IPC handle did not open between the two GPUs, and
the endpoint used RDMA inside the node instead. The step still passes; the log's `IPC open
of PE` line is the finding.

### ckdyntest

`ckdyn_2p1n` and `ckdyn_2p2n` must both show:

- `PASS devpersist/remote`;
- `PASS devpersist/samepe`;
- `ALL DONE`.

The remote mode must be `IPC` in `2p1n` and `RDMA` in `2p2n`.

`ckdyn_2p2n_full` also runs the `device` phases, which this branch does not touch.
Their result is about recharm, not this branch:

- On an unmodified recharm they fail over libfabric's tcp provider with
  `element 7 received 0 at 0, expected 84`.
- This step shows whether they pass on cxi.

### jacobi3d

The three `jacobi3d_4p2n_*` checksums must be equal.

- `channels` differing from `host` is a channel bug.
- `zerocopy` differing comes from recharm's `nocopydevice` path, not from this branch.

### channel

This step must complete. For each message size, it prints the round trip and rate of
channel puts against zero-copy entry methods. Keep both tables, `2p1n` and `2p2n`: they
are the measurement for the upstream discussion.

### Staged puts

chantest moves its elements between iterations. A put is therefore staged only when the
next iteration's put reaches the sender before the receiver's new location does, and no
test forces that.

The `_dbg` steps count `CHAN-STAGED` lines, and `-p 1` gives the most chances. If every
count is 0 after both sites, the staged path has still never run and needs a test that
puts while its receiver moves; report that with the results.

## The site tier, on Delta

recharm's pull request template asks for a run on an HPC site for any change to the
runtime core. This branch is one: it changes `cklocation.C`, `init.C` and `ck.C`. One site
is enough, so the tier runs on Delta, after the job above:

```
cd $HOME/charm-channel
sbatch -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=2 --gpus-per-node=4 -c 8 -t 01:00:00 -o tier-%j.out --wrap "SITE=delta NODES=2 CHARM_BUILD=$PWD/reconverse-linux-x86_64-cuda $PWD/tests/reconverse-site-run.sh"
```

Its last line must be `SUMMARY: ALL PASSED`. A directory that fails is the branch's
problem only if it passes on a recharm tree built the same way.

## What to bring back

From each site:

- `summary.txt` and `chan-site-<job id>.out`;
- the full log of any step that failed or printed `CHAN-WATCH`;
- the two channel tables.

From Delta, also the tier's `RESULT` lines and its `SUMMARY`.

These results, with the machine names and job IDs, also go in the pull request's
"HPC-site run" entry.
