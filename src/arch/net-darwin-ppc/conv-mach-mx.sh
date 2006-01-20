#default mx dir
#guess where the myriexpress.h is installed
if test -z "$CMK_INCDIR"
then
  if test -f /turing/software/mx/include/myriexpress.h
  then
    CMK_INCDIR="-I/turing/software/mx/include"
    CMK_LIBDIR="-L/turing/software/mx/lib"
  fi
fi


CMK_LIBS="$CMK_LIBS -lmyriexpress"
