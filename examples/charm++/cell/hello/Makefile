CHARM_BASE_DIR = ../../../..
CHARM_BIN_DIR = $(CHARM_BASE_DIR)/bin
CHARMC = $(CHARM_BIN_DIR)/charmc $(OPTS)

include $(CHARM_BASE_DIR)/tmp/Makefile.cell

OBJS = hello.o

all: hello

hello: $(OBJS) hello_shared.o
	$(CHARMC) -language charm++ -o hello hello_shared.o $(OBJS)

hello_shared.o: hello_shared.cpp hello_shared.h
	cp hello_shared.cpp hello_shared.c
	$(SPU_CC) -c -I$(CHARM_BASE_DIR)/include -o hello_shared_tmp.o hello_shared.c
#	$(SPU_CC) -L$(CHARM_BASE_DIR)/lib -L$(CELL_SDK_DIR)/sysroot/usr/spu/lib -o hello_shared hello_shared_tmp.o -lcellspu -lsim
#	$(SPU_CC) -L$(CHARM_BASE_DIR)/lib -L$(CELL_SDK_DIR)/sysroot/usr/spu/lib -o hello_shared hello_shared_tmp.o -lcellspu $(CELL_SDK_DIR)/sysroot/usr/spu/lib/libc.a $(CELL_SDK_DIR)/sysroot/usr/spu/lib/libsim.a
#	$(SPU_CC) -L$(CHARM_BASE_DIR)/lib -L$(CELL_SDK_DIR)/sysroot/usr/spu/lib -o hello_shared hello_shared_tmp.o $(SPERT_LIBS)
	$(SPU_LD) -L$(CHARM_BASE_DIR)/lib -o hello_shared hello_shared_tmp.o $(SPERT_LIBS)
	$(PPU_EMBEDSPU) spert_main hello_shared hello_shared.o

hello.decl.h: hello.ci
	$(CHARMC)  hello.ci

hello.o: hello.C hello.decl.h
	$(CHARMC) -c hello.C

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun hello_shared

