#!/bin/bash
exec gdb -batch -ex run -ex "bt 30" --args "$@" > gdb_r$SLURM_PROCID.log 2>&1
