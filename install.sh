#!/bin/bash

make clean -f Makefile.template
make CHARMDIR="$HOME/curcvs/charm/net-linux-x86_64-smp-opt" -f Makefile.template

make clean -f Makefile.template
make CHARMDIR="$HOME/curcvs/charm/net-linux-x86_64-smp-dbg" -f Makefile.template
