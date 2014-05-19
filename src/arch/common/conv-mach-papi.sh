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
      PAPI_LIBDIR="/usr/apps/tools/papi/lib64"
    else
      PAPI_LIBDIR="/usr/apps/tools/papi/lib"
    fi
    CMK_LIBDIR="-L$PAPI_LIBDIR"
    CMK_LD="$CMK_LD -Wl,-rpath,$PAPI_LIBDIR"
    CMK_LDXX="$CMK_LDXX -Wl,-rpath,$PAPI_LIBDIR"
  fi
fi

if test -z "$PAPI_LIBDIR"
then
  #macros needs to be changed to point to the correct papi directory
  PAPI_LIBDIR="$HOME/papi/lib"
  PAPI_INCDIR="$HOME/papi/include"
  CMK_INCDIR="$CMK_INCDIR -I$PAPI_INCDIR"
  CMK_LIBDIR="-L$PAPI_LIBDIR"
  CMK_LD="$CMK_LD -Wl,-rpath,$PAPI_LIBDIR"
  CMK_LDXX="$CMK_LDXX -Wl,-rpath,$PAPI_LIBDIR" 
  CMK_LIBS="$CMK_LIBS -lpapi"
fi
