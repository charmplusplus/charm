gpumigrate -- device-state migration test
=========================================

What it covers
--------------
A chare that keeps state in device memory and pups it with PUPMode::DEVICE
must find that state intact after it migrates. This test builds the smallest
thing that can check that: each element owns two device buffers filled with a
pattern derived from its array index, every element is moved one PE onward
several times, and after each move the buffers are read back and compared
element by element.

Two buffers rather than one, and the second is deliberately not a multiple of
DEVICE_PUP_ALIGN. The device side of a migration stream carries no description
of its own layout -- the receiver reconstructs it by making the same sequence
of pup calls the sender made -- so a packer and an unpacker that disagree about
where the second buffer begins would go undetected with a single buffer.

On arrival the buffers are poisoned before the device pup runs. Without that
the test can pass while doing nothing: on the same-process path the source has
just freed buffers of exactly the right size, and cudaMalloc is free to return
that same device memory with the old contents still in it.

Migration is driven with ckMigrate(), not a load balancer, so the device
migration path is exercised on its own rather than through a strategy's
decision about whether to move anything.

Build prerequisite
------------------
The Charm++ build must set -DCMK_GLOBAL_LOCATION_UPDATE=1, passed through
EXTRA_OPTS at configure time:

  cmake ... -DEXTRA_OPTS="-DCMK_GLOBAL_LOCATION_UPDATE=1"

Device zerocopy sends are addressed to the PE the sender believes hosts the
target. Without the global update a send issued around a migration arrives at
a PE that no longer hosts it, and CkRdmaDeviceIssueRgets aborts rather than
read the wrong buffer. This test does not itself send device zerocopy messages,
so it passes either way -- but a real GPU application being balanced needs the
option, so build with it.

Running it
----------
  make CHARM_DIR=<build>
  make test                     # 8 blocks, 3 rounds, 2 PEs in one process

or directly:

  ./gpumigrate [blocks] [rounds] +pe <n>

It needs at least 2 PEs; with one there is nowhere to migrate to.

Transport coverage
------------------
The device payload travels on the ordinary device zerocopy path, so which
transport carries it is decided by findTransferModeDevice, not by this test:

  same process                  device-to-device copy   `+pe 2`
  same physical node, 2+ procs  CUDA IPC                two processes on one host
  different physical nodes      device RDMA             a 2-node job

`make test` covers only the first. The other two need a job layout a bare make
cannot arrange, and are what a reviewer with a multi-GPU machine or a two-node
allocation should run:

  # same node, two processes, one GPU each
  ./gpumigrate 16 5 +p2 ++ppn 1

  # two nodes
  <launcher> -n 2 -N 1 ./gpumigrate 16 5
