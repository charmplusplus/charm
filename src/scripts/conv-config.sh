#!/bin/sh
#
# Generic converse configuration script:
#   Reads various configuration scripts and sets defaults.

if [ -z "$CHARMINC" ]
then
	echo "conv-config.sh: CHARMINC must point to the charm include dir"
	exit 1
fi

if [ -r $CHARMINC/conv-mach-pre.sh ]
then
. $CHARMINC/conv-mach-pre.sh
fi

if [ ! -r $CHARMINC/conv-mach.sh ]
then
	echo "Can't find conv-mach.sh in $CHARMINC directory."
	exit 1
fi

. $CHARMINC/conv-mach.sh

[ -z "$CMK_C_OPTIMIZE" ] && CMK_C_OPTIMIZE="-O"
[ -z "$CMK_C_DEBUG" ] && CMK_C_DEBUG="-g"
[ -z "$CMK_CXX_OPTIMIZE" ] && CMK_CXX_OPTIMIZE="$CMK_C_OPTIMIZE"
[ -z "$CMK_CXX_DEBUG" ] && CMK_CXX_DEBUG="$CMK_C_DEBUG"
[ -z "$CMK_F90_OPTIMIZE" ] && CMK_F90_OPTIMIZE="-O"
[ -z "$CMK_F90_DEBUG" ] && CMK_F90_DEBUG="-O"

[ -z "$CMK_CC" ] && CMK_CC='cc '
[ -z "$CMK_CXX" ] && CMK_CXX='c++ '
[ -z "$CMK_SUF" ] && CMK_SUF='o'
[ -z "$CMK_AR" ] && CMK_AR='ar q'
[ -z "$CMK_QT" ] && CMK_QT='generic'
[ -z "$CMK_LD" ] && CMK_LD="$CMK_CC"
[ -z "$CMK_LDXX" ] && CMK_LDXX="$CMK_CXX"

[ -z "$CMK_CF90_FIXED" ] && CMK_CF90_FIXED="$CMK_CF90"
[ -z "$CMK_CC_RELIABLE" ] && CMK_CC_RELIABLE="$CMK_CC"
[ -z "$CMK_CC_FASTEST" ] && CMK_CC_FASTEST="$CMK_CC"
[ -z "$CMK_CC_RELIABLE" ] && CMK_CC_RELIABLE="$CMK_CC"
[ -z "$CMK_SEQ_CC" ] && CMK_SEQ_CC="$CMK_CC"
[ -z "$CMK_SEQ_CXX" ] && CMK_SEQ_CXX="$CMK_CXX"
[ -z "$CMK_SEQ_LD" ] && CMK_SEQ_LD="$CMK_LD"
[ -z "$CMK_SEQ_LDXX" ] && CMK_SEQ_LDXX="$CMK_LDXX"
[ -z "$CMK_SEQ_F90" ] && CMK_SEQ_F90="$CMK_CF90"
[ -z "$CMK_SEQ_AR" ] && CMK_SEQ_AR="$CMK_AR"
[ -z "$CMK_SEQ_RANLIB" ] && CMK_SEQ_RANLIB="$CMK_RANLIB"

if [ -r $CHARMINC/conv-mach-opt.sh ]
then
. $CHARMINC/conv-mach-opt.sh
fi

