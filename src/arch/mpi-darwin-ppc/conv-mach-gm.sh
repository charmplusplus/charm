#default gm dir
#guess where the gm.h is installed
if test -z "$CMK_INCDIR"
then
  if test -f /turing/software/gm-2.0.15/include/gm.h
  then
    CMK_INCDIR="-I/turing/software/gm-2.0.15/include "
    CMK_LIBDIR="-L/turing/software/gm-2.0.15/lib "
  fi
fi


CMK_SYSLIBS="$CMK_SYSLIBS -lgm"
