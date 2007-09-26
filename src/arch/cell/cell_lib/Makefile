CHARMDIR=../..
CHARMC=$(CHARMDIR)/bin/charmc

include $(CHARMDIR)/tmp/Makefile.cell

RM = rm -f

all: libs tools
tools: rawToProj
libs: libcellppu.a libcellspu.a


##### Rules for installing the files for use by Charm++ #####

install: libs tools
	cp libcellppu.a libcellspu.a $(CHARMDIR)/lib
	cp rawToProj $(CHARMDIR)/bin


##### Rules for making the libraries (Offload API / SPE Runtime) #####

spert_common.o : spert_common.h spert_common.cpp
	$(CHARMC) -c -o spert_common.o spert_common.cpp

libcellppu.a: spert_ppu.o spert_common.o
	$(CHARMC) -o libcellppu.a spert_ppu.o spert_common.o

spert_ppu.o: spert_ppu.cpp spert_ppu.h
	$(CHARMC) -o spert_ppu.o spert_ppu.cpp

libcellspu.a: spert.o
	$(SPU_AR) r libcellspu.a spert.o

spert.o: spert.cpp spert.h
	cp spert.cpp spert.c
	$(SPU_CC) --param max-inline-insns-single=1000000 --param inline-unit-growth=1000 --param large-function-growth=1000 -c -o spert.o spert.c


##### Rules for creating SPE static timing data for the SPE Runtime #####

spert.s: spert.cpp spert.h
	cp spert.cpp spert.c
	$(SPU_CC) --param max-inline-insns-single=1000000 --param inline-unit-growth=1000 --param large-function-growth=1000 -c -S -g spert.c

spert.s.timing: spert.s
	/opt/ibm/cell-sdk/prototype/bin/spu_timing spert.s


##### Rules for making tools #####

rawToProj: rawToProj.cpp
	$(PPU_CXX) -o rawToProj rawToProj.cpp


##### Clean-up #####

clean :
	$(RM) *.o spert_ppu_test spert *.a spert.s spert.s.timing rawToProj
