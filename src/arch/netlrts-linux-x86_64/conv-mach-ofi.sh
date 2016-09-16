if test -z "$CMK_INCDIR"
then
    CMK_INCDIR="-I/usr/include/"
    CMK_LIBDIR="-L/usr/lib64/"
fi

CMK_LIBS="$CMK_LIBS -lfabric"
