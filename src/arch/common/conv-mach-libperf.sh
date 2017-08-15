CMK_USE_PERF=true

if test -f /usr/local/include/libperf.h
then
    CMK_INCDIR="-I /usr/local/include"
elif test -f /usr/include/libperf.h
then
    CMK_INCDIR="/usr/include"
elif test -f /usr/share/include/libperf.h
then
    CMK_INCDIR="/usr/share/include"
fi

if test -n CMK_INCDIR
then
    PERF_LIBDIR="/usr/local/lib"
    CMK_LIBDIR="-L$PERF_LIBDIR"
    CMK_LD_FLAGS="$CMK_LD_FLAGS -Wl,-rpath -Wl,$PERF_LIBDIR"
    CMK_LDXX_FLAGS="$CMK_LDXX_FLAGS -Wl,-rpath -Wl,$PERF_LIBDIR"
    CMK_LIBS="$CMK_LIBS -lperf"
fi
