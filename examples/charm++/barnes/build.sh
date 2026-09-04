#!/bin/bash
# Build wrapper. Two things the Makefile does not know:
#  - CHARM_PATH/STRUCTURES_PATH defaults are the in-tree layout this checkout
#    does not use.
#  - liblci.so has an unresolved DT_NEEDED on liblct.so in the same directory;
#    ld only finds an indirect dependency through -rpath-link/LD_LIBRARY_PATH.
cd /u/bhosale/charm-reconverse/examples/charm++/barnes
export LD_LIBRARY_PATH="/u/bhosale/lci_install/lib64:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:$LD_LIBRARY_PATH"
exec make \
  CHARM_PATH=/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda \
  STRUCTURES_PATH=/u/bhosale/utility/structures \
  "$@"
