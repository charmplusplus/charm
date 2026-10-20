#!/bin/bash
if [ "$SLURM_PROCID" = "0" ]; then exec gdb -batch -ex run -ex "bt 30" --args "$@"; else exec "$@"; fi
