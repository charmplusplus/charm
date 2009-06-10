CMK_USE_PAPI=true

#default papi dir
#guess where the papi.h is installed
if test -z "$CMK_INCDIR"
then
  # abe.ncsa
  if test -f /usr/apps/tools/papi/include/papi.h
  then
    CMK_INCDIR="-I /usr/apps/tools/papi/include"
    if test -n "$CMK_CC64"
    then
      CMK_LIBDIR="-L /usr/apps/tools/papi/lib64"
    else
      CMK_LIBDIR="-L /usr/apps/tools/papi/lib"
    fi
  fi
fi


CMK_LIBS="$CMK_LIBS -lpapi"
