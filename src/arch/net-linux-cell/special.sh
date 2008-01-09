#!/bin/sh

./gathertree ../../src/arch/cell .

# make links
test ! -f "../include/cell-api.h" && ./system_ln "../tmp/cell-api.h" ../include
for f in cell_lib/*.h
do
  test ! -f "../include/$f" && ./system_ln "../tmp/$f" ../include
done

#make library
export CHARMINC=../include
. ./conv-config.sh
#%if test ! -f $CELL_SDK_DIR/sysroot/usr/include/libspe.h
#if test ! -f $CELL_SDK_DIR/sysroot/usr/include/libspe2.h
#then
#  echo "Please define CELL_SDK_DIR in charm/src/arch/net-linux-cell/conv-mach.sh!"
#  exit 1
#fi

cat > Makefile.cell << EOF
PPU_CC = $CMK_CC \$(OPTS)
PPU_CXX = $CMK_CXX \$(OPTS)
SPU_CC = $CMK_SPE_CC \$(OPTS)
SPU_CXX = $CMK_SPE_CXX \$(OPTS)
SPU_LD = $CMK_SPE_LD
SPU_LDXX = $CMK_SPE_LDXX
SPU_AR = $CMK_SPE_AR
PPU_EMBEDSPU = $CMK_PPU_EMBEDSPU
SPERT_LIBS = $CMK_SPERT_LIBS
EOF

cd cell_lib && make install
