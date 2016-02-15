#!/bin/sh

export CHARMINC=../include
. ./conv-config.sh

if test -n "$BUILD_CUDA"
then
  echo "---------------------- special.sh for cuda executing ----------------"
  if [ `cat .gdir` == "gni" ] ; then
    export CRAY_CUDA_PROXY=1
  fi

  ./gathertree $SRCBASE/arch/cuda .

# make links
  test ! -f "../include/cuda-hybrid-api.h" && ./system_ln "../tmp/hybridAPI/cuda-hybrid-api.h" ../include && test ! -f "../include/wr.h" && ./system_ln "../tmp/hybridAPI/wr.h" ../include && test ! -f "../include/wrqueue.h" && ./system_ln "../tmp/hybridAPI/wrqueue.h" ../include

#make library
  export CHARMINC=../include
  . ./conv-config.sh

fi
