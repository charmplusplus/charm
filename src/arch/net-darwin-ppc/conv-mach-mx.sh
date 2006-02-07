#default mx dir
#guess where the myriexpress.h is installed
if test -z "$CMK_INCDIR"
then
  if test -f /turing/software/TIGER/mx/include/myriexpress.h
  then
    CMK_INCDIR="-I/turing/software/TIGER/mx/include"
    CMK_LIBDIR="-L/turing/software/TIGER/mx/lib"
  elif test -f /turing/software/mx/include/myriexpress.h
  then
    CMK_INCDIR="-I/turing/software/mx/include"
    CMK_LIBDIR="-L/turing/software/mx/lib"
  else
    echo
    echo "Please specify path to mx installation in conv-mach-mx.sh!"
    echo
    exit 1
  fi
fi


CMK_LIBS="$CMK_LIBS -lmyriexpress"
