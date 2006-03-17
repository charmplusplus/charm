#!/bin/sh

./gathertree ../../src/arch/cell .

# make links
test ! -f "../include/cell-api.h" && ./system_ln "../tmp/cell-api.h" ../include
for f in cell_lib/*.h
do
  test ! -f "../include/$f" && ./system_ln "../tmp/$f" ../include
done

#make library
cd cell_lib && make install
