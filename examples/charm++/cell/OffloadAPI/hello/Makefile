CHARM_BASE_DIR = ../../../../..
CHARM_BIN_DIR = $(CHARM_BASE_DIR)/bin
CHARMC = $(CHARM_BIN_DIR)/charmc $(OPTS)

include $(CHARM_BASE_DIR)/tmp/Makefile.cell

OBJS = hello.o

all: hello

hello: $(OBJS) hello_shared.o
	$(PPU_CXX) -L$(CHARM_BASE_DIR)/lib -o hello hello_shared.o $(OBJS) -lspe2 -lcellppu

hello_shared.o: hello_shared.cpp hello_shared.h
	cp hello_shared.cpp hello_shared.c
	$(SPU_CC) -c -I$(CHARM_BASE_DIR)/include -o hello_shared_tmp.o hello_shared.c
	$(SPU_LD) -L$(CHARM_BASE_DIR)/lib -o hello_shared hello_shared_tmp.o $(SPERT_LIBS)
	$(PPU_EMBEDSPU) spert_main hello_shared hello_shared.o

hello.o: hello.cpp hello_shared.h
	$(PPU_CC) -I$(CHARM_BASE_DIR)/include -c hello.cpp -o hello.o

clean:
	rm -f *.decl.h *.def.h conv-host *.o hello charmrun hello_shared hello_shared.c
