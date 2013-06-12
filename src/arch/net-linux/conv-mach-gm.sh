#default gm dir
#guess where the gm.h is installed
if test -z "$CMK_INCDIR"
then
  # gm ver 1.0
  if test -f /usr/gm/include/gm.h
  then
    CMK_INCDIR="-I /usr/gm/include"
    CMK_LIBDIR="-L /usr/gm/lib"
  # gm ver 2.0
  elif test -f /opt/gm/include/gm.h
  then
    CMK_INCDIR="-I /opt/gm/include"
    CMK_LIBDIR="-L /opt/gm/lib"
  elif test -f /usr/local/gm/include/gm.h
  then
    CMK_INCDIR="-I /usr/local/gm/include"
    CMK_LIBDIR="-L /usr/local/gm/lib"
  fi
fi


CMK_LIBS="$CMK_LIBS -lgm"
