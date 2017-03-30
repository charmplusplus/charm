#!/bin/bash
# This script can be run on a compiled file (.o, .so, .a, linked ELF
# binary) to produce a list of symbols representing global variables
# potentially of concern for AMPI from a privatization stand-point

PROG=$1

nm -C $PROG | egrep ' [BbCDdGgSs] ' | grep -v __ioinit
