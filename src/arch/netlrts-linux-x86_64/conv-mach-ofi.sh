#test libfabric path
if test -z "$CMK_INCDIR"
then
  if test -f /opt/cisco/libfabric/include/rdma/fabric.h
  then
    CMK_INCDIR="-I/opt/cisco/libfabric/include"
    OFI_LIBDIR="/opt/cisco/libfabric/lib"
    CMK_LIBDIR="-L$OFI_LIBDIR"
    CMK_LD="$CMK_LD -Wl,-rpath,$OFI_LIBDIR"
    CMK_LDXX="$CMK_LDXX -Wl,-rpath,$OFI_LIBDIR"
  else
    CMK_INCDIR="-I/usr/include/"
    CMK_LIBDIR="-L/usr/lib64/"
  fi
fi

CMK_LIBS="$CMK_LIBS -lfabric"
