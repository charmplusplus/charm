CHARM_BASE_DIR = ../../../../..
CHARM_BIN_DIR = $(CHARM_BASE_DIR)/bin
CHARMC = $(CHARM_BIN_DIR)/charmc $(OPTS)

include $(CHARM_BASE_DIR)/tmp/Makefile.cell

all: jacobi

jacobi: jacobi.o jacobi_shared.o
	$(PPU_CXX) -L$(CHARM_BASE_DIR)/lib -o jacobi jacobi_shared.o jacobi.o -lspe2 -lcellppu

jacobi_shared.o: jacobi_shared.cpp jacobi_shared.h
	cp jacobi_shared.cpp jacobi_shared.c
	$(SPU_CC) -c -I$(CHARM_BASE_DIR)/include -o jacobi_shared_tmp.o jacobi_shared.c
	$(SPU_LD) -L$(CHARM_BASE_DIR)/lib -o jacobi_shared jacobi_shared_tmp.o $(SPERT_LIBS)
	$(PPU_EMBEDSPU) spert_main jacobi_shared jacobi_shared.o

jacobi.o: jacobi.cpp jacobi_shared.h
	$(PPU_CC) -I$(CHARM_BASE_DIR)/include -c jacobi.cpp -o jacobi.o

clean:
	rm -f *.decl.h *.def.h conv-host *.o jacobi charmrun jacobi_shared jacobi_shared.c
