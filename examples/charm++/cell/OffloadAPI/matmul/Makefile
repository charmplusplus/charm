CHARM_BASE_DIR = ../../../../..
CHARM_BIN_DIR = $(CHARM_BASE_DIR)/bin
CHARMC = $(CHARM_BIN_DIR)/charmc $(OPTS)

include $(CHARM_BASE_DIR)/tmp/Makefile.cell

all: matmul

matmul: matmul.o matmul_shared.o
	$(PPU_CXX) -L/opt/IBM/cell-sdk-1.1/sysroot/usr/lib -L$(CHARM_BASE_DIR)/lib -o matmul matmul_shared.o matmul.o -lspe -lcellppu

matmul_shared.o: matmul_shared.cpp matmul_shared.h
	cp matmul_shared.cpp matmul_shared.c
	$(SPU_CC) -c -I$(CHARM_BASE_DIR)/include -o matmul_shared_tmp.o matmul_shared.c
	$(SPU_LD) -L$(CHARM_BASE_DIR)/lib -o matmul_shared matmul_shared_tmp.o $(SPERT_LIBS)
	$(PPU_EMBEDSPU) spert_main matmul_shared matmul_shared.o

matmul.o: matmul.cpp matmul_shared.h
	$(PPU_CC) -I$(CHARM_BASE_DIR)/include -c matmul.cpp -o matmul.o

clean:
	rm -f *.decl.h *.def.h conv-host *.o matmul charmrun matmul_shared matmul_shared.c
