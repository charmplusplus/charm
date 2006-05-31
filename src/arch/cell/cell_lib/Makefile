
CHARMDIR=../..
CHARMC=$(CHARMDIR)/bin/charmc

include $(CHARMDIR)/tmp/Makefile.cell

RM = rm -f


libs: libcellppu.a libcellspu.a


# Common object file (shared)
spert_common.o : spert_common.h spert_common.cpp
	$(CHARMC) -I$(CELL_SDK_DIR)/sysroot/usr/spu/include -c -o spert_common.o spert_common.cpp

libcellppu.a: spert_ppu.o spert_common.o
	$(CHARMC) -o libcellppu.a spert_ppu.o spert_common.o

spert_ppu.o: spert_ppu.cpp spert_ppu.h
	$(CHARMC) -I$(CELL_SDK_DIR)/sysroot/usr/include -o spert_ppu.o spert_ppu.cpp

libcellspu.a: spert.o
	$(SPU_AR) r libcellspu.a spert.o

spert.o: spert.cpp spert.h
	cp spert.cpp spert.c
	spu-gcc $(OPTS) -c -I$(CELL_SDK_DIR)/sysroot/usr/spu/include -o spert.o spert.c

install: libcellppu.a libcellspu.a
	cp libcellppu.a libcellspu.a $(CHARMDIR)/lib


# Clean-up
clean :
	$(RM) *.o spert_ppu_test spert *.a
