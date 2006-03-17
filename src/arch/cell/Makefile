TOOLCHAIN_DIR = /opt/sce/toolchain-2.3
CELL_SDK_DIR = /home/strife/cellsim

CHARMC=$(HOME)/cellsim/charm/net-linux-cell/bin/charmc

PPU_CC = ppu32-gcc $(OPTS)
PPU_CXX = ppu32-g++ $(OPTS)
SPU_CC = spu-gcc $(OPTS)
SPU_CXX = spu-g++ $(OPTS)
SPU_AR = spu-ar
PPU_EMBEDSPU = ppu32-embedspu

RM = rm -f

GENERAL_DEPEND = general.h common.h Makefile

libs: libcellppu.a libcellspu.a


# Common object file (shared)
common.o : common.h common.cpp general.h
	@echo Compiling common.o...
	$(PPU_CXX) -I$(CELL_SDK_DIR)/cbe_linux/src/libspe -c -o common.o common.cpp
	@echo

libcellppu.a: spert_ppu.o common.o
	$(CHARMC) -o libcellppu.a spert_ppu.o common.o

spert_ppu.o: spert_ppu.cpp spert_ppu.h
	$(CHARMC) -I$(CELL_SDK_DIR)/sysroot/usr/include -o spert_ppu.o spert_ppu.cpp

libcellspu.a: spert.o
	$(SPU_AR) r libcellspu.a spert.o

spert.o: spert.cpp spert.h
	$(SPU_CXX) -c -I$(CELL_SDK_DIR)/sysroot/usr/spu/include -o spert.o spert.cpp 
	


# Clean-up
clean :
	$(RM) *.o spert_ppu_test spert *.a
