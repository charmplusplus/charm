#default gm dir
#guess where the gm.h is installed
if test -z $CMK_INCDIR 
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
  fi
fi

CMK_CPP_C="$CMK_CPP_C -E  $CMK_INCDIR "
CMK_CC="$CMK_CC $CMK_INCDIR "
CMK_CC_RELIABLE="$CMK_CC_RELIABLE $CMK_INCDIR "
CMK_CC_FASTEST="$CMK_CC_FASTEST $CMK_INCDIR "
CMK_CXX="$CMK_CXX $CMK_INCDIR "
CMK_CXXPP="$CMK_CXXPP -E $CMK_INCDIR "
CMK_LD="$CMK_LD $CMK_LIBDIR "
CMK_LDXX="$CMK_LDXX $CMK_LIBDIR "

CMK_LIBS="$CMK_LIBS -lgm"
