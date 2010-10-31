#!/bin/sh

echo "---------------------- special.sh for cuda executing ----------------"

./gathertree ../../src/arch/cuda .

# make links
test ! -f "../include/cuda-hybrid-api.h" && ./system_ln "../tmp/hybridAPI/cuda-hybrid-api.h" ../include && test ! -f "../include/wr.h" && ./system_ln "../tmp/hybridAPI/wr.h" ../include && test ! -f "../include/wrqueue.h" && ./system_ln "../tmp/hybridAPI/wrqueue.h" ../include

#make library
export CHARMINC=../include
. ./conv-config.sh

#cat > Makefile.cuda << EOF
#PPU_CC = $CMK_CC \$(OPTS)
#PPU_CXX = $CMK_CXX \$(OPTS)
#SPU_CC = $CMK_SPE_CC \$(OPTS)
#SPU_CXX = $CMK_SPE_CXX \$(OPTS)
#SPU_LD = $CMK_SPE_LD
#SPU_LDXX = $CMK_SPE_LDXX
#SPU_AR = $CMK_SPE_AR
#PPU_EMBEDSPU = $CMK_PPU_EMBEDSPU
#SPERT_LIBS = $CMK_SPERT_LIBS
#EOF
