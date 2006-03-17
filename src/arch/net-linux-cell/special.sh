#!/bin/sh

./gathertree ../../src/arch/cell .

# make links
test ! -f "../include/cell-api.h" && ./system_ln "../tmp/cell-api.h" ../include
for f in cell_lib/*.h
do
  test ! -f "../include/$f" && ./system_ln "../tmp/$f" ../include
done

#make library
. ./conv-mach.sh
if test ! -f $CELL_SDK_DIR/sysroot/usr/include/libspe.h
then
  echo "Please define CELL_SDK_DIR in charm/src/arch/net-linux-cell/conv-mach.sh!"
  exit 1
fi

cat > Makefile.cell << EOF
CELL_SDK_DIR=$CELL_SDK_DIR
SPU_CC = spu-gcc \$(OPTS)
SPU_CXX = spu-g++ \$(OPTS)
SPU_AR = spu-ar
PPU_EMBEDSPU = ppu32-embedspu
EOF

cd cell_lib && make install
