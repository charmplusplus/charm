if echo $LOADEDMODULES | grep -q papi; then
    true
else
    echo 'Must have a papi module loaded (e.g. module load papi)' >&2
    exit 1
fi
CMK_USE_PAPI=true
USE_SPP_PAPI=true
#you should run module load papi
PAPI_INCDIR=`pkg-config --cflags papi`
PAPI_LIBDIR=`pkg-config --libs papi`
CMK_INCDIR="$CMK_INCDIR $PAPI_INCDIR"
CMK_LIBDIR="$CMK_LIBDIR $PAPI_LIBDIR"
CMK_LD_FLAGS="$CMK_LD_FLAGS -Wl,-rpath,$PAPI_LIBDIR"
CMK_LDXX_FLAGS="$CMK_LDXX_FLAGS -Wl,-rpath,$PAPI_LIBDIR"
