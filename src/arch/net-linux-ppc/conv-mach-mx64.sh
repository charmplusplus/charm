#default mx dir
#guess where the myriexpress.h is installed
if test -z "$CMK_INCDIR"
then
  if test -f /opt/mx/include/myriexpress.h
  then
    CMK_INCDIR="-I/opt/mx/include"
    CMK_LIBDIR="-L/opt/mx/lib64"
  else
    echo
    echo "Please specify path to mx installation in conv-mach-mx.sh!"
    echo
    exit 1
  fi
fi


CMK_LIBS="$CMK_LIBS -lmyriexpress"
