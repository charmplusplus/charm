#!/bin/sh

export CHARMINC=../include
. ./conv-config.sh

if test -n "$BUILD_CUDA"
then
  echo "---------------------- special.sh for cuda executing ----------------"
  if [ "$CMK_GDIR" == "gni" ] ; then
    export CRAY_CUDA_PROXY=1
  fi

  ./gathertree $SRCBASE/arch/cuda .

# make links
  test ! -f "../include/hapi_src.h" && ./system_ln "../tmp/hybridAPI/hapi_src.h" ../include && test ! -f "../include/hapi.h" && ./system_ln "../tmp/hybridAPI/hapi.h" ../include && test ! -f "../include/wrqueue.h" && ./system_ln "../tmp/hybridAPI/wrqueue.h" ../include

#make library
  export CHARMINC=../include
  . ./conv-config.sh

fi
