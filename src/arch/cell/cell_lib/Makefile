TOOLCHAIN_DIR = /opt/sce/toolchain-2.3
CELL_SDK_DIR = /home/strife/cellsim

CHARMDIR=../..
CHARMC=$(CHARMDIR)/bin/charmc

PPU_CC = ppu32-gcc $(OPTS)
PPU_CXX = ppu32-g++ $(OPTS)
SPU_CC = spu-gcc $(OPTS)
SPU_CXX = spu-g++ $(OPTS)
SPU_AR = spu-ar
PPU_EMBEDSPU = ppu32-embedspu

RM = rm -f


libs: libcellppu.a libcellspu.a


# Common object file (shared)
spert_common.o : spert_common.h spert_common.cpp
	$(CHARMC) -I$(CELL_SDK_DIR)/cbe_linux/src/libspe -c -o spert_common.o spert_common.cpp

libcellppu.a: spert_ppu.o spert_common.o
	$(CHARMC) -o libcellppu.a spert_ppu.o spert_common.o

spert_ppu.o: spert_ppu.cpp spert_ppu.h
	$(CHARMC) -I$(CELL_SDK_DIR)/sysroot/usr/include -o spert_ppu.o spert_ppu.cpp

libcellspu.a: spert.o
	$(SPU_AR) r libcellspu.a spert.o

spert.o: spert.cpp spert.h
	$(SPU_CXX) -c -I$(CELL_SDK_DIR)/sysroot/usr/spu/include -o spert.o spert.cpp 
	
install: libcellppu.a libcellspu.a
	cp libcellppu.a libcellspu.a $(CHARMDIR)/lib


# Clean-up
clean :
	$(RM) *.o spert_ppu_test spert *.a
