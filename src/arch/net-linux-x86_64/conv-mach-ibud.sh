#default ibverbs path for openib
if test -z "$CMK_INCDIR"
then
  #openib-1.1
  if test -f /opt/ofed/include/infiniband/verbs.h
  then
    CMK_INCDIR="-I/opt/ofed/include/"
    CMK_LIBDIR="-L/opt/ofed/lib64"
	fi
  if test -f /usr/local/ofed/include/infiniband/verbs.h
  then
    CMK_INCDIR="-I/usr/local/ofed/include/"
    CMK_LIBDIR="-L/usr/local/ofed/lib64"
	fi
fi

CMK_LIBS="$CMK_LIBS -libverbs"

