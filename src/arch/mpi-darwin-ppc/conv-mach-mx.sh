#default gm dir
#guess where the gm.h is installed
if test -z "$CMK_INCDIR"
then
  if test -f /turing/software/mx/include/myriexpress.h
  then
    CMK_INCDIR="-I/turing/software/mx/include"
    CMK_LIBDIR="-L/turing/software/mx/lib"
  fi
fi
#CMK_SYSLIBS="$CMK_SYSLIBS -lmyriexpress"
